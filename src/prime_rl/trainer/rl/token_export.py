import atexit
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from prime_rl.configs.trainer import IPOLossConfig, TrainerConfig
from prime_rl.trainer.rl.loss import compute_importance_ratio_and_mismatch_kl

SCHEMA_VERSION = 1


class DisabledTokenExporter:
    def export(self, *args: Any, **kwargs: Any) -> None:
        return

    def mark_stable(self) -> None:
        return

    def close(self) -> None:
        return


class TokenExporter:
    def __init__(
        self,
        output_dir: Path,
        rank: int,
    ) -> None:
        self.rank = rank
        self.output_dir = output_dir / "token_exports"
        self._closed = False
        self._initialized_files: set[tuple[int, int]] = set()
        self._sequences_by_file: dict[tuple[int, int], int] = {}
        self._pending_stable_dirs: set[Path] = set()
        atexit.register(self.close)

    def export(
        self,
        step: int,
        micro_step: int,
        micro_batch: Mapping[str, Any],
        model_output: Mapping[str, Tensor],
        sequence_lengths: list[int],
        loss_config: Any,
    ) -> None:
        columns = _export_columns(micro_batch, model_output, loss_config)
        _check_lengths(columns)
        file_key = (step, self.rank)

        start = 0
        for micro_sequence_idx, length in enumerate(sequence_lengths):
            raw_end = start + length
            end = _trim_padding(columns, start, raw_end)
            if end > start and any(columns["loss_mask"][start:end]):
                self._write(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "step": step,
                        "rank": self.rank,
                        "micro_step": micro_step,
                        "micro_sequence_idx": micro_sequence_idx,
                        "export_sequence_idx": self._sequences_by_file.get(file_key, 0),
                        "env_name": _first_non_empty(columns["env_names"][start:end]),
                        **_slice_columns(columns, start, end),
                    },
                    step,
                )
                self._sequences_by_file[file_key] = self._sequences_by_file.get(file_key, 0) + 1
            start = raw_end

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

    def mark_stable(self) -> None:
        # The caller barriers first so a STABLE only lands after every rank flushed.
        while self._pending_stable_dirs:
            stable_dir = self._pending_stable_dirs.pop()
            (stable_dir / "STABLE").touch()

    def _export_dir(self, step: int) -> Path:
        return self.output_dir / f"step_{step}"

    def _export_file(self, step: int) -> Path:
        if self._closed:
            raise RuntimeError(f"Token exporter is closed for {self.output_dir}")

        step_dir = self._export_dir(step)
        try:
            step_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            if not step_dir.is_dir():
                raise
        return step_dir / f"rank_{self.rank}.jsonl"

    def _write(self, record: dict[str, Any], step: int) -> None:
        if self._closed:
            raise RuntimeError(f"Token exporter is closed for {self.output_dir}")

        file_key = (step, self.rank)
        mode = "a" if file_key in self._initialized_files else "w"
        export_file = self._export_file(step)
        with export_file.open(mode, encoding="utf-8") as file:
            file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self._initialized_files.add(file_key)
        self._pending_stable_dirs.add(export_file.parent)


def setup_token_exporter(
    config: TrainerConfig, parallel_dims: Any, world: Any, logger: Any
) -> TokenExporter | DisabledTokenExporter:
    if not config.enable_token_export:
        return DisabledTokenExporter()
    if parallel_dims.cp_enabled and parallel_dims.world_mesh["cp"].get_local_rank() != 0:
        return DisabledTokenExporter()

    exporter = TokenExporter(config.output_dir, world.rank)
    logger.info(f"Writing token exports under {exporter.output_dir}")
    return exporter


def _export_columns(
    micro_batch: Mapping[str, Any], model_output: Mapping[str, Tensor], loss_config: Any
) -> dict[str, list[Any]]:
    token_ids = _tensor_to_ints(micro_batch["input_ids"])
    seq_len = len(token_ids)
    trainer_logprobs = model_output["logprobs"]
    export_tensors = _compute_export_tensors(micro_batch, trainer_logprobs, loss_config)

    return {
        "token_ids": token_ids,
        "position_ids": _tensor_to_ints(micro_batch["position_ids"]),
        "loss_mask": _tensor_to_bools(micro_batch["loss_mask"]),
        "advantages": _tensor_to_floats(micro_batch["advantages"]),
        "inference_logprobs": _tensor_to_floats(micro_batch["inference_logprobs"]),
        "trainer_logprobs": _tensor_to_floats(trainer_logprobs),
        "entropy": _tensor_to_floats(model_output["entropy"]),
        "mismatch_kl": _optional_tensor_to_floats(export_tensors["mismatch_kl"], seq_len),
        "log_importance_ratio": _optional_tensor_to_floats(export_tensors["log_importance_ratio"], seq_len),
        "importance_ratio": _optional_tensor_to_floats(export_tensors["importance_ratio"], seq_len),
        "prob_delta": _optional_tensor_to_floats(export_tensors["prob_delta"], seq_len),
        "is_masked": _optional_tensor_to_bools(export_tensors["is_masked"], seq_len),
        "is_masked_high": _optional_tensor_to_bools(export_tensors["is_masked_high"], seq_len),
        "is_masked_low": _optional_tensor_to_bools(export_tensors["is_masked_low"], seq_len),
        # Component weight streams; ``None`` columns mean the defaults (rl 1.0
        # on the loss mask, no ce/ref_kl component).
        "rl_weights": _optional_tensor_to_floats(micro_batch.get("rl_weights"), seq_len),
        "ce_weights": _optional_tensor_to_floats(micro_batch.get("ce_weights"), seq_len),
        "ref_kl_weights": _optional_tensor_to_floats(micro_batch.get("ref_kl_weights"), seq_len),
        "env_names": list(micro_batch["env_names"]),
    }


def _compute_export_tensors(
    micro_batch: Mapping[str, Any], trainer_logprobs: Tensor, loss_config: Any
) -> dict[str, Tensor | None]:
    fields: dict[str, Tensor | None] = {
        "log_importance_ratio": None,
        "importance_ratio": None,
        "mismatch_kl": None,
        "prob_delta": None,
        "is_masked": None,
        "is_masked_high": None,
        "is_masked_low": None,
    }
    # Ratio-based fields are meaningless when no token has sampling logprobs
    # (e.g. pure CE batches distilling frozen-model tokens): no rl member
    # (stream present but all-zero) and no ref_kl member.
    rl_weights = micro_batch.get("rl_weights")
    ref_kl_weights = micro_batch.get("ref_kl_weights")
    no_rl = rl_weights is not None and not bool((rl_weights != 0).any())
    no_ref_kl = ref_kl_weights is None or not bool((ref_kl_weights != 0).any())
    if no_rl and no_ref_kl:
        return fields

    inference_logprobs = micro_batch["inference_logprobs"].to(trainer_logprobs.device)
    loss_mask = micro_batch["loss_mask"].to(trainer_logprobs.device)
    with torch.no_grad():
        log_ratio, ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(trainer_logprobs, inference_logprobs)
        prob_delta = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
        fields["log_importance_ratio"] = log_ratio
        fields["importance_ratio"] = ratio
        fields["mismatch_kl"] = mismatch_kl
        fields["prob_delta"] = prob_delta
        if isinstance(loss_config, IPOLossConfig):
            invalid_high = prob_delta > loss_config.eps
            invalid_low = prob_delta < -loss_config.eps
            fields["is_masked"] = loss_mask & (invalid_high | invalid_low)
            fields["is_masked_high"] = loss_mask & invalid_high
            fields["is_masked_low"] = loss_mask & invalid_low
    return fields


def _tensor_to_ints(tensor: Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _tensor_to_bools(tensor: Tensor) -> list[bool]:
    return [bool(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _tensor_to_floats(tensor: Tensor) -> list[float | None]:
    values = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1).tolist()
    return [_json_float(value) for value in values]


def _optional_tensor_to_floats(tensor: Tensor | None, seq_len: int) -> list[float | None]:
    if tensor is None:
        return [None] * seq_len
    return _tensor_to_floats(tensor)


def _optional_tensor_to_bools(tensor: Tensor | None, seq_len: int) -> list[bool | None]:
    if tensor is None:
        return [None] * seq_len
    return _tensor_to_bools(tensor)


def _check_lengths(columns: Mapping[str, Sequence[Any]]) -> None:
    lengths = {key: len(values) for key, values in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Token export fields must have aligned lengths, got {lengths}")


def _slice_columns(columns: Mapping[str, Sequence[Any]], start: int, end: int) -> dict[str, list[Any]]:
    return {key: list(values[start:end]) for key, values in columns.items() if key != "env_names"}


def _trim_padding(columns: Mapping[str, Sequence[Any]], start: int, end: int) -> int:
    env_names = columns["env_names"]
    loss_mask = columns["loss_mask"]
    while end > start and env_names[end - 1] == "" and not loss_mask[end - 1]:
        end -= 1
    return end


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _first_non_empty(values: Sequence[str]) -> str | None:
    for value in values:
        if value:
            return value
    return None

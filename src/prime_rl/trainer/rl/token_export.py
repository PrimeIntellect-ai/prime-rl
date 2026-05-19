import atexit
import json
import math
import queue
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from prime_rl.configs.trainer import DefaultLossConfig, TokenExportConfig, TrainerConfig
from prime_rl.trainer.rl.loss import compute_importance_ratio_and_mismatch_kl

SCHEMA_VERSION = 1
_STOP = object()


class AsyncJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._error: BaseException | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=f"token-export-writer:{path}", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def write(self, record: dict[str, Any]) -> None:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError(f"Token export writer is closed for {self.path}")
        self._queue.put(record)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_STOP)
        self._thread.join()
        self._raise_if_failed()

    def _run(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                while True:
                    record = self._queue.get()
                    try:
                        if record is _STOP:
                            break
                        f.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                    finally:
                        self._queue.task_done()
        except BaseException as exc:
            self._error = exc

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Token export writer failed for {self.path}") from self._error


class DisabledTokenExporter:
    def export(self, *args: Any, **kwargs: Any) -> None:
        return

    def close(self) -> None:
        return


class TokenExporter:
    def __init__(
        self,
        config: TokenExportConfig,
        output_dir: Path,
        rank: int,
    ) -> None:
        self.config = config
        self.rank = rank
        self.path = self._resolve_path(config.path, output_dir, rank)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = AsyncJsonlWriter(self.path)
        self._current_step: int | None = None
        self._sequences_this_step = 0

    def export(
        self,
        step: int,
        micro_step: int,
        micro_batch: Mapping[str, Any],
        model_output: Mapping[str, Tensor],
        response_lengths: list[int],
        loss_config: Any,
    ) -> None:
        if self._current_step != step:
            self._current_step = step
            self._sequences_this_step = 0

        flat = self._flatten_micro_batch(
            micro_batch,
            model_output["logprobs"],
            model_output["entropy"],
            _compute_export_tensors(micro_batch, model_output["logprobs"], loss_config),
        )
        start = 0
        for micro_sequence_idx, length in enumerate(response_lengths):
            end = start + length
            record = self._build_record(
                step=step,
                micro_step=micro_step,
                micro_sequence_idx=micro_sequence_idx,
                flat=flat,
                start=start,
                end=end,
                sft_loss=bool(micro_batch["sft_loss"]),
            )
            start = end
            if record is None:
                continue
            self._writer.write(record)
            self._sequences_this_step += 1

    def close(self) -> None:
        self._writer.close()

    @staticmethod
    def _resolve_path(path: Path | None, output_dir: Path, rank: int) -> Path:
        if path is None:
            return output_dir / "token_exports" / f"rank_{rank}.jsonl"
        if path.is_absolute():
            return path
        return output_dir / path

    def _flatten_micro_batch(
        self,
        micro_batch: Mapping[str, Any],
        trainer_logprobs: Tensor,
        entropy: Tensor,
        export_tensors: Mapping[str, Tensor | None],
    ) -> dict[str, list[Any]]:
        input_ids = _tensor_to_ints(micro_batch["input_ids"])
        seq_len = len(input_ids)
        rewards_tensor = micro_batch.get("rewards")

        flat = {
            "input_ids": input_ids,
            "position_ids": _tensor_to_ints(micro_batch["position_ids"]),
            "loss_mask": _tensor_to_bools(micro_batch["loss_mask"]),
            "advantages": _tensor_to_floats(micro_batch["advantages"]),
            "rewards": _optional_tensor_to_floats(rewards_tensor, seq_len),
            "inference_logprobs": _tensor_to_floats(micro_batch["inference_logprobs"]),
            "trainer_logprobs": _tensor_to_floats(trainer_logprobs),
            "entropy": _tensor_to_floats(entropy),
            "mismatch_kl": _optional_tensor_to_floats(export_tensors["mismatch_kl"], seq_len),
            "log_importance_ratio": _optional_tensor_to_floats(export_tensors["log_importance_ratio"], seq_len),
            "importance_ratio": _optional_tensor_to_floats(export_tensors["importance_ratio"], seq_len),
            "prob_delta": _optional_tensor_to_floats(export_tensors["prob_delta"], seq_len),
            "is_masked": _optional_tensor_to_bools(export_tensors["is_masked"], seq_len),
            "is_masked_high": _optional_tensor_to_bools(export_tensors["is_masked_high"], seq_len),
            "is_masked_low": _optional_tensor_to_bools(export_tensors["is_masked_low"], seq_len),
            "env_names": list(micro_batch["env_names"]),
        }
        lengths = {key: len(values) for key, values in flat.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Token export fields must have aligned lengths, got {lengths}")
        return flat

    def _build_record(
        self,
        *,
        step: int,
        micro_step: int,
        micro_sequence_idx: int,
        flat: dict[str, list[Any]],
        start: int,
        end: int,
        sft_loss: bool,
    ) -> dict[str, Any] | None:
        end = _trim_padding(flat, start, end)
        if start >= end:
            return None

        loss_mask = flat["loss_mask"][start:end]
        if not any(loss_mask):
            return None

        token_ids = flat["input_ids"][start:end]
        tokens = []
        for local_idx, absolute_idx in enumerate(range(start, end)):
            log_importance_ratio = _json_float(flat["log_importance_ratio"][absolute_idx])
            token = {
                "index": local_idx,
                "id": token_ids[local_idx],
                "position": flat["position_ids"][absolute_idx],
                "loss_mask": loss_mask[local_idx],
                "advantage": _json_float(flat["advantages"][absolute_idx]),
                "reward": _json_float(flat["rewards"][absolute_idx]),
                "entropy": _json_float(flat["entropy"][absolute_idx]),
                "mismatch_kl": _json_float(flat["mismatch_kl"][absolute_idx]),
                "log_importance_ratio": log_importance_ratio,
                "sample_kl_trainer_to_inference": log_importance_ratio,
                "sample_kl_inference_to_trainer": -log_importance_ratio if log_importance_ratio is not None else None,
                "importance_ratio": _json_float(flat["importance_ratio"][absolute_idx]),
                "prob_delta": _json_float(flat["prob_delta"][absolute_idx]),
                "inference_logprob": _json_float(flat["inference_logprobs"][absolute_idx]),
                "trainer_logprob": _json_float(flat["trainer_logprobs"][absolute_idx]),
                "is_masked": flat["is_masked"][absolute_idx],
                "is_masked_high": flat["is_masked_high"][absolute_idx],
                "is_masked_low": flat["is_masked_low"][absolute_idx],
            }
            token["inference_prob"] = _prob_from_logprob(token["inference_logprob"])
            token["trainer_prob"] = _prob_from_logprob(token["trainer_logprob"])
            tokens.append(token)

        env_name = _first_non_empty(flat["env_names"][start:end])
        return {
            "schema_version": SCHEMA_VERSION,
            "step": step,
            "rank": self.rank,
            "micro_step": micro_step,
            "micro_sequence_idx": micro_sequence_idx,
            "export_sequence_idx": self._sequences_this_step,
            "env_name": env_name,
            "sft_loss": sft_loss,
            "tokens": tokens,
        }


def setup_token_exporter(
    config: TrainerConfig, parallel_dims: Any, world: Any, logger: Any
) -> TokenExporter | DisabledTokenExporter:
    token_export_config = config.experimental.token_export
    if token_export_config is None:
        return DisabledTokenExporter()
    if parallel_dims.cp_enabled and parallel_dims.world_mesh["cp"].get_local_rank() != 0:
        return DisabledTokenExporter()

    exporter = TokenExporter(token_export_config, config.output_dir, world.rank)
    logger.info(f"Writing token exports to {exporter.path}")
    return exporter


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
    if micro_batch["sft_loss"]:
        return fields

    inference_logprobs = micro_batch["inference_logprobs"].to(trainer_logprobs.device)
    loss_mask = micro_batch["loss_mask"].to(trainer_logprobs.device)
    advantages = micro_batch["advantages"].to(trainer_logprobs.device)
    with torch.no_grad():
        log_ratio, ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(trainer_logprobs, inference_logprobs)
        prob_delta = torch.exp(trainer_logprobs) - torch.exp(inference_logprobs)
        fields.update(
            {
                "log_importance_ratio": log_ratio,
                "importance_ratio": ratio,
                "mismatch_kl": mismatch_kl,
                "prob_delta": prob_delta,
            }
        )
        if isinstance(loss_config, DefaultLossConfig):
            invalid_high = prob_delta > loss_config.dppo_mask_high
            invalid_low = prob_delta < -loss_config.dppo_mask_low
            positive_advantages = advantages > 0
            negative_advantages = advantages < 0
            invalid = torch.where(positive_advantages, invalid_high, invalid_low)
            fields.update(
                {
                    "is_masked": loss_mask & invalid,
                    "is_masked_high": loss_mask & positive_advantages & invalid_high,
                    "is_masked_low": loss_mask & negative_advantages & invalid_low,
                }
            )
    return fields


def _tensor_to_ints(tensor: Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _tensor_to_bools(tensor: Tensor) -> list[bool]:
    return [bool(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _tensor_to_floats(tensor: Tensor) -> list[float]:
    values = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1).tolist()
    return [float(value) for value in values]


def _optional_tensor_to_floats(tensor: Tensor | None, seq_len: int) -> list[float | None]:
    if tensor is None:
        return [None] * seq_len
    return _tensor_to_floats(tensor)


def _optional_tensor_to_bools(tensor: Tensor | None, seq_len: int) -> list[bool | None]:
    if tensor is None:
        return [None] * seq_len
    return _tensor_to_bools(tensor)


def _trim_padding(flat: dict[str, list[Any]], start: int, end: int) -> int:
    env_names = flat["env_names"]
    loss_mask = flat["loss_mask"]
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


def _prob_from_logprob(logprob: float | None) -> float | None:
    if logprob is None:
        return None
    if logprob > 709:
        return None
    return math.exp(logprob)


def _first_non_empty(values: list[str]) -> str | None:
    for value in values:
        if value:
            return value
    return None

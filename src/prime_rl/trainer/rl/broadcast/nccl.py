import os
import pickle
import time
from pathlib import Path
from typing import Generator, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.trainer import NCCLWeightBroadcastConfig
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.trainer.weights import get_max_layer_num
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path

NCCL_READY_MARKER = "NCCL_READY"

KERNEL_WEIGHT_TRANSFER = os.environ.get("PRIME_RL_KERNEL_WEIGHT_TRANSFER", "0") == "1"


def broadcast_integer(integer: int, communicator: PyNcclCommunicator) -> None:
    """Broadcast an integer to a process group using NCCL communicator."""
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    communicator.broadcast(integer_tensor, src=0)


def broadcast_state_dict(state_dict: dict[str, Tensor], communicator: PyNcclCommunicator) -> None:
    """Broadcast a state dict to NCCL process group using the PyNcclCommunicator."""
    # Group tensors by dtype
    dtype_groups: dict[torch.dtype, list[tuple[str, Tensor]]] = {}
    for key, value in state_dict.items():
        assert not isinstance(value, DTensor), (
            "DTensor is not supported for broadcast, should have been converted to tensor already"
        )
        dtype = value.dtype
        if dtype not in dtype_groups:
            dtype_groups[dtype] = []
        dtype_groups[dtype].append((key, value))

    # Build metadata: for each dtype group, store keys and shapes
    metadata = {}
    for dtype, items in dtype_groups.items():
        metadata[dtype] = [(key, value.shape, value.numel()) for key, value in items]

    # Send metadata
    state = pickle.dumps(metadata)
    size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.ByteTensor(list(state)).cuda()
    communicator.broadcast(state_tensor, src=0)

    # Concatenate and broadcast tensors grouped by dtype
    for dtype, items in dtype_groups.items():
        # Flatten all tensors and concatenate
        flat_tensors = [value.flatten() for _, value in items]
        concatenated = torch.cat(flat_tensors)
        communicator.broadcast(concatenated, src=0)
        del concatenated
        # Clean up individual tensors
        for _, value in items:
            del value


def filter_state_dict_by_layers(
    state_dict: dict[str, torch.Tensor], num_layers: int
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    """Yield a generator of state dicts for each layer as well as the remaining weights."""
    yield 0, {key: value for key, value in state_dict.items() if "model.layers" not in key}

    for i in range(1, num_layers + 1):  # +1 because layer indices start from 1
        yield (
            i,
            {
                key: value
                for key, value in state_dict.items()
                if key.startswith(f"model.layers.{i}.") or key == f"model.layers.{i}"
            },
        )


def _quantize_to_fp8_blockwise(weight: Tensor, block_size: int = 128) -> tuple[Tensor, Tensor]:
    """Quantize a 2D weight tensor to FP8 e4m3 with block-wise scaling."""
    rows, cols = weight.shape
    br = bc = block_size
    pad_r = (br - rows % br) % br
    pad_c = (bc - cols % bc) % bc
    if pad_r > 0 or pad_c > 0:
        padded = torch.zeros(rows + pad_r, cols + pad_c, dtype=weight.dtype, device=weight.device)
        padded[:rows, :cols] = weight
    else:
        padded = weight.clone()
    pr, pc = padded.shape
    blocks = padded.reshape(pr // br, br, pc // bc, bc).permute(0, 2, 1, 3)
    max_abs = blocks.float().abs().amax(dim=(2, 3))
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (max_abs / fp8_max).clamp(min=1e-12)
    blocks_fp8 = (blocks.float() / scale[:, :, None, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return blocks_fp8.permute(0, 2, 1, 3).reshape(pr, pc)[:rows, :cols].contiguous(), scale.float().contiguous()


def _convert_layer_to_kernel_format(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    config: object,
    quantize_fp8: bool = False,
) -> dict[str, Tensor]:
    """Convert a single layer's PrimeRL state dict to vLLM kernel format.

    Handles: fusing q_a_proj+kv_a_proj_with_mqa, gate+up projections,
    stacking MoE experts, and optional FP8 quantization.
    """
    out: dict[str, Tensor] = {}
    p = f"model.layers.{layer_idx}"

    def add(name: str, tensor: Tensor) -> None:
        out[name] = tensor

    def add_maybe_fp8(name: str, tensor: Tensor) -> None:
        if quantize_fp8 and tensor.ndim == 2:
            fp8_w, scale = _quantize_to_fp8_blockwise(tensor.cuda())
            out[name] = fp8_w
            out[name[: -len(".weight")] + ".weight_scale_inv" if name.endswith(".weight") else name + "_scale_inv"] = (
                scale
            )
        else:
            out[name] = tensor

    # Norms
    for key in [f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight"]:
        if key in state_dict:
            add(key, state_dict[key])

    # Attention: fuse q_a_proj + kv_a_proj_with_mqa → fused_qkv_a_proj
    q_a_key = f"{p}.self_attn.q_a_proj.weight"
    kv_a_key = f"{p}.self_attn.kv_a_proj_with_mqa.weight"
    if q_a_key in state_dict and kv_a_key in state_dict:
        fused = torch.cat([state_dict[q_a_key], state_dict[kv_a_key]], dim=0)
        add_maybe_fp8(f"{p}.self_attn.fused_qkv_a_proj.weight", fused)

    for suffix in ["q_a_layernorm.weight", "kv_a_layernorm.weight"]:
        key = f"{p}.self_attn.{suffix}"
        if key in state_dict:
            add(key, state_dict[key])

    for suffix in ["q_b_proj.weight", "kv_b_proj.weight", "o_proj.weight"]:
        key = f"{p}.self_attn.{suffix}"
        if key in state_dict:
            add_maybe_fp8(key, state_dict[key])

    # Indexer
    for suffix in ["indexer.wq_b.weight", "indexer.wk.weight"]:
        key = f"{p}.self_attn.{suffix}"
        if key in state_dict:
            add_maybe_fp8(key, state_dict[key])
    for suffix in ["indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.weights_proj.weight"]:
        key = f"{p}.self_attn.{suffix}"
        if key in state_dict:
            add(key, state_dict[key])

    # Dense MLP: fuse gate_proj + up_proj → gate_up_proj
    gate_key = f"{p}.mlp.gate_proj.weight"
    up_key = f"{p}.mlp.up_proj.weight"
    if gate_key in state_dict and up_key in state_dict:
        gate_up = torch.cat([state_dict[gate_key], state_dict[up_key]], dim=0)
        add_maybe_fp8(f"{p}.mlp.gate_up_proj.weight", gate_up)
        add_maybe_fp8(f"{p}.mlp.down_proj.weight", state_dict[f"{p}.mlp.down_proj.weight"])

    # MoE: router
    router_key = f"{p}.mlp.router.gate.weight"
    if router_key in state_dict:
        add(f"{p}.mlp.gate.weight", state_dict[router_key])
    expert_bias_key = f"{p}.mlp.expert_bias"
    if expert_bias_key in state_dict:
        add(f"{p}.mlp.gate.e_score_correction_bias", state_dict[expert_bias_key])

    # MoE: routed experts w1+w3 → w13, w2
    w1_key = f"{p}.mlp.experts.w1"
    if w1_key in state_dict:
        w1 = state_dict[w1_key].cuda()
        w3 = state_dict[f"{p}.mlp.experts.w3"].cuda()
        w2 = state_dict[f"{p}.mlp.experts.w2"].cuda()
        w13 = torch.cat([w1, w3], dim=1)
        n_experts = w1.shape[0]

        if quantize_fp8:
            w13_fp8, w13_s, w2_fp8, w2_s = [], [], [], []
            for j in range(n_experts):
                f8, s = _quantize_to_fp8_blockwise(w13[j])
                w13_fp8.append(f8)
                w13_s.append(s)
                f8, s = _quantize_to_fp8_blockwise(w2[j])
                w2_fp8.append(f8)
                w2_s.append(s)
            out[f"{p}.mlp.experts.w13_weight"] = torch.stack(w13_fp8)
            out[f"{p}.mlp.experts.w13_weight_scale_inv"] = torch.stack(w13_s)
            out[f"{p}.mlp.experts.w2_weight"] = torch.stack(w2_fp8)
            out[f"{p}.mlp.experts.w2_weight_scale_inv"] = torch.stack(w2_s)
        else:
            out[f"{p}.mlp.experts.w13_weight"] = w13
            out[f"{p}.mlp.experts.w2_weight"] = w2

    # MoE: shared experts w1+w3 → gate_up_proj, w2 → down_proj
    sw1_key = f"{p}.mlp.shared_expert.w1"
    if sw1_key in state_dict:
        sw1 = state_dict[sw1_key].cuda()
        sw3 = state_dict[f"{p}.mlp.shared_expert.w3"].cuda()
        sw2 = state_dict[f"{p}.mlp.shared_expert.w2"].cuda()
        if sw1.dim() == 3:
            sw1, sw3, sw2 = sw1.squeeze(0), sw3.squeeze(0), sw2.squeeze(0)
        add_maybe_fp8(f"{p}.mlp.shared_experts.gate_up_proj.weight", torch.cat([sw1, sw3], dim=0))
        add_maybe_fp8(f"{p}.mlp.shared_experts.down_proj.weight", sw2)

    return out


class NCCLWeightBroadcastSender:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = dtype

        if self.world.is_master:
            # Trainer is on rank 0 in process group with all inference GPUs
            pg = StatelessProcessGroup.create(
                host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
            self.logger.debug("NCCL broadcast initialized on master rank")
        else:
            self.logger.debug("NCCL broadcast initialized on non-master rank (no communicator)")

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL."""
        state_dict = model.state_dict()
        num_layers = get_max_layer_num(state_dict)
        num_state_dict_to_send = num_layers + 1  # we send all layer plus the remaining weights

        if self.world.is_master:
            broadcast_integer(num_state_dict_to_send, self.communicator)

        self.logger.debug(f"Broadcasting {num_state_dict_to_send} layer state dicts")

        if True:
            self._broadcast_kernel_format(model, state_dict, num_layers)
        else:
            self._broadcast_checkpoint_format(model, state_dict, num_layers)

    def _resolve_dtensors(self, sd: dict[str, Tensor]) -> dict[str, Tensor]:
        """Resolve DTensors to full tensors in-place and return the dict."""
        for key, value in list(sd.items()):
            if isinstance(value, DTensor):
                sd[key] = cast(DTensor, value.to(self.dtype)).full_tensor()
        return sd

    def _broadcast_checkpoint_format(self, model: nn.Module, state_dict: dict[str, Tensor], num_layers: int) -> None:
        """Broadcast weights in HF checkpoint format (original path)."""
        for layer_id, layer_sd in filter_state_dict_by_layers(state_dict, num_layers):
            layer_sd = self._resolve_dtensors(layer_sd)

            if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(layer_sd):
                model.convert_layer_to_hf(layer_sd, layer_id)
            else:
                from transformers.core_model_loading import revert_weight_conversion

                layer_sd = revert_weight_conversion(model, layer_sd)

            if self.world.is_master:
                broadcast_state_dict(layer_sd, self.communicator)

    def _broadcast_kernel_format(self, model: nn.Module, state_dict: dict[str, Tensor], num_layers: int) -> None:
        """Broadcast weights in vLLM kernel format (fused, optionally FP8).

        Converts PrimeRL naming to vLLM kernel naming in-place,
        so the receiver can use param.copy_() directly.
        """
        quantize_fp8 = os.environ.get("PRIME_RL_KERNEL_WEIGHT_TRANSFER_FP8", "0") == "1"
        self.logger.debug(f"Using kernel weight transfer (quantize_fp8={quantize_fp8})")

        # Non-layer weights (embeddings, norm, lm_head) — resolve DTensors for this slice only
        non_layer_sd = {k: v for k, v in state_dict.items() if "model.layers" not in k}
        non_layer_sd = self._resolve_dtensors(non_layer_sd)
        if self.world.is_master:
            broadcast_state_dict(non_layer_sd, self.communicator)
        del non_layer_sd

        # Per-layer kernel-format conversion — resolve DTensors per layer to avoid OOM
        for layer_idx in range(num_layers):
            layer_sd = {k: v for k, v in state_dict.items() if k.startswith(f"model.layers.{layer_idx}.")}
            layer_sd = self._resolve_dtensors(layer_sd)
            kernel_sd = _convert_layer_to_kernel_format(layer_sd, layer_idx, model.config, quantize_fp8=quantize_fp8)
            del layer_sd
            if self.world.is_master:
                broadcast_state_dict(kernel_sd, self.communicator)
            del kernel_sd


class NCCLWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine using NCCL."""

    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        device: int | str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(output_dir)
        self.logger = get_logger()
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.nccl_broadcast_sender = NCCLWeightBroadcastSender(
            config.host, config.port, 0, config.inference_world_size + 1, device, config.timeout, dtype
        )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via NCCL")
        start_time = time.perf_counter()
        notified_runs: list[tuple[int, Path]] = []
        if self.world.is_master:
            notified_runs = self._notify_orchestrator()
            # Wait for inference workers to signal readiness before starting NCCL broadcast
            self._wait_for_nccl_ready(notified_runs)
        self.nccl_broadcast_sender.broadcast_weights(model, step)
        self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _notify_orchestrator(self) -> list[tuple[int, Path]]:
        """Notify the orchestrator to initiate weight broadcast.

        Returns:
            List of (run_idx, save_dir) tuples for runs that were notified.
        """
        notified_runs: list[tuple[int, Path]] = []
        if self.world.is_master:
            for idx in self.multi_run_manager.used_idxs:
                if not self.multi_run_manager.ready_to_update[idx]:
                    continue

                try:
                    save_dir = get_step_path(
                        get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                        self.multi_run_manager.progress[idx].step,
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)

                    stable_file = save_dir / "STABLE"
                    stable_file.touch()
                    notified_runs.append((idx, save_dir))
                except FileNotFoundError:
                    self.logger.warning(f"Run {idx} is deleted, skipping")
                except Exception as e:
                    self.logger.error(f"Error broadcasting weights for run {idx}: {e}")
                finally:
                    self.multi_run_manager.ready_to_update[idx] = False
        return notified_runs

    def _wait_for_nccl_ready(self, notified_runs: list[tuple[int, Path]]):
        """Wait for inference workers to signal they are ready to receive NCCL broadcast."""
        for idx, save_dir in notified_runs:
            nccl_ready_file = save_dir / NCCL_READY_MARKER
            self.logger.debug(f"Waiting for NCCL_READY marker at {nccl_ready_file}")
            sync_wait_for_path(nccl_ready_file, interval=0.1, log_interval=10)
            self.logger.debug(f"Inference workers ready for NCCL broadcast (run {idx})")

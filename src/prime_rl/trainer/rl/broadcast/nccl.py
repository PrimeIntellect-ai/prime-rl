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
from prime_rl.utils.vlm import get_layer_prefix

NCCL_READY_MARKER = "NCCL_READY"


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
    state_dict: dict[str, torch.Tensor], num_layers: int, layer_prefix: str
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    """Yield non-layer weights first, then each layer's weights.

    Yields (layer_idx, layer_state_dict) where layer_idx is -1 for the non-layer
    dict and the actual layer index (0, 1, ...) for layer dicts.
    """
    yield -1, {key: value for key, value in state_dict.items() if not key.startswith(layer_prefix)}

    for i in range(num_layers):
        yield (
            i,
            {key: value for key, value in state_dict.items() if key.startswith(f"{layer_prefix}{i}.")},
        )


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
        use_vllm_format_transfer: bool = False,
        quantize_fp8: bool = False,
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = dtype
        self.use_vllm_format_transfer = use_vllm_format_transfer
        self.quantize_fp8 = quantize_fp8

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
        self.logger.debug(f"Broadcasting weights for step {step}")
        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        num_state_dict_to_send = num_layers + 1  # we send all layer plus the remaining weights

        if self.world.is_master:
            self.logger.debug("Broadcasting number of state dicts to send")
            broadcast_integer(num_state_dict_to_send, self.communicator)
            torch.cuda.current_stream().synchronize()

        self.logger.debug(f"Broadcasting {num_state_dict_to_send} layer state dicts")

        if self.use_vllm_format_transfer:
            self._broadcast_kernel_format(model, state_dict, num_layers)
        else:
            self._broadcast_checkpoint_format(model, state_dict, num_layers, layer_prefix)

    def _resolve_dtensors(self, sd: dict[str, Tensor]) -> dict[str, Tensor]:
        """Resolve DTensors to full tensors in-place and return the dict."""
        for key, value in list(sd.items()):
            if isinstance(value, DTensor):
                sd[key] = cast(DTensor, value.to(self.dtype)).full_tensor()
        return sd

    def _broadcast_checkpoint_format(
        self, model: nn.Module, state_dict: dict[str, Tensor], num_layers: int, layer_prefix: str
    ) -> None:
        """Broadcast weights in HF checkpoint format (original path)."""
        for layer_id, layer_sd in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            layer_sd = self._resolve_dtensors(layer_sd)

            if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(layer_sd):
                model.convert_layer_to_hf(layer_sd, layer_id)
            else:
                from transformers.core_model_loading import revert_weight_conversion

                layer_sd = revert_weight_conversion(model, layer_sd)

            if self.world.is_master:
                broadcast_state_dict(layer_sd, self.communicator)
                # Synchronize to ensure the broadcast completes before the next
                # layer's DTensor resolution launches FSDP all-gathers. Running
                # two NCCL communicators concurrently on the same GPU deadlocks.
                torch.cuda.current_stream().synchronize()

    def _broadcast_kernel_format(self, model: nn.Module, state_dict: dict[str, Tensor], num_layers: int) -> None:
        """Broadcast weights in vLLM kernel format (fused, optionally FP8).

        Converts PrimeRL naming to vLLM kernel naming in-place,
        so the receiver can use param.copy_() directly.
        """
        assert isinstance(model, PreTrainedModelPrimeRL), (
            f"Kernel format transfer requires a PrimeRL model, got {type(model).__name__}"
        )
        quantize_fp8 = self.quantize_fp8
        self.logger.debug(f"Using kernel weight transfer (quantize_fp8={quantize_fp8})")

        # Non-layer weights (embeddings, norm, lm_head) — resolve DTensors for this slice only
        non_layer_sd = {k: v for k, v in state_dict.items() if "model.layers" not in k}
        non_layer_sd = self._resolve_dtensors(non_layer_sd)
        if self.world.is_master:
            broadcast_state_dict(non_layer_sd, self.communicator)
            torch.cuda.current_stream().synchronize()
        del non_layer_sd

        # Per-layer kernel-format conversion — resolve DTensors per layer to avoid OOM
        for layer_idx in range(num_layers):
            layer_sd = {k: v for k, v in state_dict.items() if k.startswith(f"model.layers.{layer_idx}.")}
            layer_sd = self._resolve_dtensors(layer_sd)
            kernel_sd = model.convert_layer_to_vllm_kernel(layer_sd, layer_idx, quantize_fp8=quantize_fp8)
            del layer_sd
            if self.world.is_master:
                broadcast_state_dict(kernel_sd, self.communicator)
                torch.cuda.current_stream().synchronize()
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
            config.host,
            config.port,
            0,
            config.inference_world_size + 1,
            device,
            config.timeout,
            dtype,
            use_vllm_format_transfer=config.use_vllm_format_transfer,
            quantize_fp8=config.quantize_fp8,
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

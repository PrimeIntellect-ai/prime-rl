import time
from pathlib import Path
from typing import Generator, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

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


def filter_state_dict_by_layers(
    state_dict: dict[str, torch.Tensor], num_layers: int
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    """Yield a state dict for each layer as well as the remaining non-layer weights."""
    yield 0, {key: value for key, value in state_dict.items() if "model.layers" not in key}

    for i in range(1, num_layers + 1):
        yield (
            i,
            {
                key: value
                for key, value in state_dict.items()
                if key.startswith(f"model.layers.{i}.") or key == f"model.layers.{i}"
            },
        )


class NCCLWeightBroadcastSender:
    def __init__(
        self,
        host: str,
        port: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        dtype: torch.dtype = torch.bfloat16,
        packed: bool = True,
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = dtype
        self.packed = packed
        self.communicator = None
        self._weight_metadata: dict | None = None

        if self.world.is_master:
            self.communicator = NCCLWeightTransferEngine.trainer_init({
                "master_address": host,
                "master_port": port,
                "world_size": world_size,
            })

    def _hf_weight_iterator(self, model: nn.Module) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield (name, tensor) pairs in HF checkpoint format, layer by layer."""
        state_dict = model.state_dict()
        num_layers = get_max_layer_num(state_dict)

        for layer_id, state_dict in filter_state_dict_by_layers(state_dict, num_layers):
            for key, value in list(state_dict.items()):
                if isinstance(value, DTensor):
                    value = cast(DTensor, value.to(self.dtype)).full_tensor()
                state_dict[key] = value

            if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
                model.convert_layer_to_hf(state_dict, layer_id)
            else:
                from transformers.core_model_loading import revert_weight_conversion

                state_dict = revert_weight_conversion(model, state_dict)

            yield from state_dict.items()

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast weights via NCCL. All ranks iterate for DTensor collectives, only master sends."""
        # Collect metadata on first call (all ranks must iterate for DTensor collectives)
        if self._weight_metadata is None:
            names, dtype_names, shapes = [], [], []
            for name, tensor in self._hf_weight_iterator(model):
                names.append(name)
                dtype_names.append(str(tensor.dtype).replace("torch.", ""))
                shapes.append(list(tensor.shape))
            self._weight_metadata = {"names": names, "dtype_names": dtype_names, "shapes": shapes}

        if self.world.is_master and self.communicator is not None:
            from prime_rl.inference.vllm.weight_transfer import PrimeNCCLWeightTransferEngine

            PrimeNCCLWeightTransferEngine.trainer_send_weights(
                iterator=self._hf_weight_iterator(model),
                trainer_args={"group": self.communicator, "packed": self.packed},
                metadata=self._weight_metadata,
            )
        else:
            for _ in self._hf_weight_iterator(model):
                pass


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
            config.host, config.port, config.inference_world_size + 1, device, config.timeout, dtype, config.packed
        )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        self.logger.debug("Starting broadcasting weights to inference engine via NCCL")
        start_time = time.perf_counter()
        notified_runs: list[tuple[int, Path]] = []
        if self.world.is_master:
            notified_runs = self._notify_orchestrator()
            self._wait_for_nccl_ready(notified_runs)
        self.nccl_broadcast_sender.broadcast_weights(model, step)
        self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _notify_orchestrator(self) -> list[tuple[int, Path]]:
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
        for idx, save_dir in notified_runs:
            nccl_ready_file = save_dir / NCCL_READY_MARKER
            self.logger.debug(f"Waiting for NCCL_READY marker at {nccl_ready_file}")
            sync_wait_for_path(nccl_ready_file, interval=0.1, log_interval=10)
            self.logger.debug(f"Inference workers ready for NCCL broadcast (run {idx})")

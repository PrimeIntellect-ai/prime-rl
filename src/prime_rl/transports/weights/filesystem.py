import warnings
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig, LoRAConfig
from prime_rl.orchestrator.clients import load_lora_adapter
from prime_rl.orchestrator.delta_sync import DeltaEndpointPool
from prime_rl.transports.weights.base import FINISHED_MARKER, WeightReceiver, WeightSender
from prime_rl.utils.delta import SAFETENSORS_DELTA_FILENAME, STREAMING_DELTA_FILENAME, ModelDeltaManager
from prime_rl.utils.pathing import wait_for_path
from prime_rl.utils.weights import (
    convert_state_dict_to_hf,
    gather_weights_parallel,
    resolve_fqn,
    resolve_wire_dtype,
    save_state_dict,
    save_state_dict_parallel,
)


class FileSystemWeightSender(WeightSender):
    """Broadcast weights by saving a HF-compatible checkpoint (or, for LoRA
    runs, the PEFT-shaped adapter) to a shared filesystem."""

    def __init__(
        self,
        output_dir: Path,
        config: FileSystemWeightBroadcastConfig,
        lora_config: LoRAConfig | None = None,
    ):
        super().__init__(output_dir, config.timeout)
        self.lora_config = lora_config
        self.mode = config.mode
        self.delta_index_encoding = config.delta_index_encoding
        self.delta_streaming_enabled = config.delta_streaming_enabled
        self.delta_stream_group_size = config.delta_stream_group_size
        self.retain_all_deltas = config.retain_all_deltas
        self.delta_manager = ModelDeltaManager() if self.mode == "delta" else None
        self._previous_state: dict[str, Tensor] | None = None
        if self.mode == "delta" and lora_config is not None:
            raise ValueError("filesystem delta weight broadcast does not support LoRA")
        self.logger.debug(f"Initialized filesystem weight broadcast (mode={self.mode})")

    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        if self.lora_config is not None:
            from prime_rl.trainer.lora import get_lora_state, save_lora_config

            # All ranks must participate in DTensor gathering, but only master saves
            state_dict = get_lora_state().adapter_state_dict()
            for key, value in state_dict.items():
                if isinstance(value, DTensor):
                    value = value.full_tensor()
                if self.world.is_master:
                    state_dict[key] = value.to("cpu", non_blocking=False)
            if self.world.is_master:
                self.logger.debug(f"Saving adapter to {step_dir}")
                save_state_dict(state_dict, step_dir, save_sharded=False, adapter=True)
                save_lora_config(
                    model,
                    step_dir,
                    rank=self.lora_config.rank,
                    alpha=self.lora_config.alpha,
                    dropout=self.lora_config.dropout,
                )
        elif self.mode == "full":
            dist.barrier()
            state_dict = gather_weights_parallel(model)
            state_dict = convert_state_dict_to_hf(model, state_dict)
            self.logger.debug(f"Saving weights to {step_dir}")
            save_state_dict_parallel(state_dict, step_dir)
        else:
            state_dict = self._gather_delta_state(model)
            if self.world.is_master:
                if self._previous_state is None:
                    self.logger.debug(f"Saving sparse-delta base marker to {step_dir}")
                    self._save_delta(state_dict, state_dict, step_dir)
                else:
                    self._save_delta(self._previous_state, state_dict, step_dir)
                self._previous_state = state_dict
            dist.barrier()

    def _gather_delta_state(self, model: nn.Module) -> dict[str, Tensor]:
        """Gather a complete logical HF state on rank zero for delta extraction."""
        keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
        state_dict: dict[str, Tensor] = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")
            for key, value in model.state_dict().items():
                if isinstance(value, DTensor):
                    target_dtype = resolve_wire_dtype(keep_in_fp32, key, torch.bfloat16)
                    value = cast(DTensor, value.to(target_dtype)).full_tensor()
                if self.world.is_master:
                    state_dict[resolve_fqn(model, key)] = value.to("cpu")
            dist.barrier()
        if self.world.is_master:
            return convert_state_dict_to_hf(model, state_dict)
        return {}

    def _save_delta(
        self,
        base_state: dict[str, Tensor],
        target_state: dict[str, Tensor],
        step_dir: Path,
    ) -> None:
        if self.delta_manager is None:
            raise RuntimeError("delta manager is not initialized")
        if self.delta_streaming_enabled:
            delta_path = step_dir / STREAMING_DELTA_FILENAME
            self.delta_manager.extract_sparse_delta_streaming_from_state_dicts(
                base_state,
                target_state,
                delta_path,
                group_size=self.delta_stream_group_size,
                index_encoding=self.delta_index_encoding,
                save_stats=True,
            )
        else:
            delta_path = step_dir / SAFETENSORS_DELTA_FILENAME
            self.delta_manager.extract_sparse_delta_from_state_dicts(
                base_state,
                target_state,
                delta_path,
                index_encoding=self.delta_index_encoding,
                save_stats=True,
            )
        self.logger.debug(f"Saved sparse delta to {delta_path}")

    def _clean(self, step: int) -> None:
        if self.mode == "delta" and self.retain_all_deltas:
            return
        super()._clean(step)


class FileSystemWeightReceiver(WeightReceiver):
    """Loads broadcasts from the shared filesystem. The acknowledgement lets
    the trainer start writing; the engines are only touched once the weights
    are fully on disk. An adapter broadcast (PEFT dir) is hot-swapped under
    live traffic — an in-place adapter reload is a vLLM-native op that needs
    no engine pause; a full checkpoint pauses the engines for the load."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        client_config = self.admin_plane.client_config
        self.delta_endpoints = DeltaEndpointPool(
            self.admin_plane.clients,
            lease_enabled=client_config.lease_enabled,
            recovery_enabled=client_config.lease_recovery_enabled,
            cooldown_s=client_config.lease_cooldown_s,
            health_timeout_s=client_config.lease_recovery_poll_interval_s,
            stage_num_streams=self.config.stage_num_streams,
            stage_chunk_size_bytes=self.config.stage_chunk_size_mb * 1024 * 1024,
            stage_chunk_retries=self.config.stage_chunk_retries,
            stage_retries=self.config.stage_retries,
        )

    async def receive(self, step: int) -> None:
        weights_dir = self.step_dir(step)
        self._ack(step)
        finished_path = weights_dir / FINISHED_MARKER

        if self.config.mode == "delta" and self.config.update_protocol == "stage_commit":
            await self._receive_staged_delta(step, weights_dir, finished_path)
            return

        await wait_for_path(finished_path)
        if (weights_dir / "adapter_config.json").exists():
            await load_lora_adapter(self.admin_plane, self.model_name, weights_dir)
        elif (weights_dir / STREAMING_DELTA_FILENAME).exists() or (weights_dir / SAFETENSORS_DELTA_FILENAME).exists():
            await self.admin_plane.update_weights(
                weights_dir,
                transport="filesystem",
                step=step,
                mode="delta",
            )
        else:
            await self.admin_plane.update_weights(weights_dir, transport="filesystem", step=step)

    async def _receive_staged_delta(self, step: int, weights_dir: Path, finished_path: Path) -> None:
        transport = self.config.stage_transport
        upload = transport != "shared_fs"
        upload_method = {
            "shared_fs": "multipart",
            "http_upload": "multipart",
            "chunked_upload": "chunked",
            "streaming_upload": "streaming",
        }[transport]

        background_stream = transport == "streaming_upload" and self.config.background_stage
        if not background_stream:
            await wait_for_path(finished_path)

        version = str(step)
        await self.delta_endpoints.stage(
            weights_dir,
            version=version,
            base_version=self.delta_endpoints.active_version,
            upload=upload,
            upload_method=upload_method,
            done_path=finished_path if background_stream else None,
        )
        await self.delta_endpoints.commit(version)

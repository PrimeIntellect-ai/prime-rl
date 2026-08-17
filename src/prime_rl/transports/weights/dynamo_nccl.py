from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, cast

import httpx
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import NCCLWeightBroadcastConfig
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.utils import get_world
from prime_rl.transports.weights.base import WeightBroadcast
from prime_rl.utils.client import NCCL_READY_MARKER
from prime_rl.utils.dynamo import (
    DynamoDiscoveryPending,
    DynamoVLLMWeightSyncClient,
    DynamoWorker,
    client_headers,
    discover_dynamo_workers,
    topology_fingerprint,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix


def _filter_state_dict_by_layers(
    state_dict: dict[str, Tensor], num_layers: int, layer_prefix: str
) -> Iterator[tuple[int, dict[str, Tensor]]]:
    yield -1, {name: tensor for name, tensor in state_dict.items() if not name.startswith(layer_prefix)}
    for layer_idx in range(num_layers):
        yield (
            layer_idx,
            {name: tensor for name, tensor in state_dict.items() if name.startswith(f"{layer_prefix}{layer_idx}.")},
        )


def _preprocess_layer_checkpoint(model: nn.Module, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
        model.convert_layer_to_hf(state_dict, layer_idx)
        return state_dict

    from transformers.core_model_loading import revert_weight_conversion

    return revert_weight_conversion(model, state_dict)


class PrimeWeightSource:
    """Expose Prime checkpoint-format tensors through vLLM's WeightSource protocol."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.model: nn.Module | None = None
        self.dtype = dtype
        self._metadata = None

    def set_model(self, model: nn.Module) -> None:
        if model is not self.model:
            self._metadata = None
        self.model = model

    def _items(self) -> Iterator[tuple[str, Tensor]]:
        if self.model is None:
            raise RuntimeError("Prime weight source has no model")
        from prime_rl.trainer.conversion_utils import get_max_layer_num

        state_dict = self.model.state_dict()
        layer_prefix = get_layer_prefix(self.model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        for layer_idx, layer_state_dict in _filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            resolved = {}
            for name, tensor in layer_state_dict.items():
                if isinstance(tensor, DTensor):
                    tensor = cast(DTensor, tensor.to(self.dtype)).full_tensor()
                resolved[name] = tensor
            yield from _preprocess_layer_checkpoint(self.model, resolved, layer_idx).items()

    def metadata(self):
        from vllm.distributed.weight_transfer import ParamMeta

        if self._metadata is None:
            if self.model is None:
                raise RuntimeError("Prime weight source has no model")
            from prime_rl.trainer.conversion_utils import get_max_layer_num

            state_dict = {
                name: torch.empty(
                    tuple(tensor.shape),
                    dtype=self.dtype if isinstance(tensor, DTensor) else tensor.dtype,
                    device="meta",
                )
                for name, tensor in self.model.state_dict().items()
            }
            layer_prefix = get_layer_prefix(self.model.config)
            num_layers = get_max_layer_num(state_dict, layer_prefix)
            metadata = []
            for layer_idx, layer_state_dict in _filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
                converted = _preprocess_layer_checkpoint(self.model, layer_state_dict, layer_idx)
                metadata.extend(
                    ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in converted.items()
                )
            self._metadata = metadata
        return list(self._metadata)

    def __iter__(self) -> Iterator[tuple[str, Tensor]]:
        yield from self._items()


def _discover(config: NCCLWeightBroadcastConfig) -> tuple[DynamoWorker, ...]:
    assert config.dynamo is not None
    headers = client_headers(config.dynamo.headers, config.dynamo.headers_from_env, config.dynamo.api_key_var)
    deadline = time.monotonic() + config.timeout
    previous_fingerprint = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            workers = discover_dynamo_workers(
                config.dynamo.discovery_url,
                config.dynamo.model_name,
                headers=headers,
                timeout=min(30.0, max(1.0, deadline - time.monotonic())),
            )
            if sum(worker.world_size for worker in workers) != config.inference_world_size:
                raise DynamoDiscoveryPending("Dynamo worker capacity does not match Prime inference capacity")
            fingerprint = topology_fingerprint(workers)
            if fingerprint == previous_fingerprint:
                return workers
            previous_fingerprint = fingerprint
        except (DynamoDiscoveryPending, httpx.TransportError) as error:
            previous_fingerprint = None
            last_error = error
        time.sleep(1)
    raise TimeoutError("Dynamo workers did not become ready before trainer initialization") from last_error


class DynamoNCCLWeightBroadcast(WeightBroadcast):
    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(output_dir)
        if config.dynamo is None:
            raise ValueError("Dynamo native NCCL requires Dynamo discovery configuration")

        from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

        from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

        self.world = get_world()
        workers = _discover(config)
        headers = client_headers(config.dynamo.headers, config.dynamo.headers_from_env, config.dynamo.api_key_var)
        self.client = DynamoVLLMWeightSyncClient(workers, headers, config.timeout)
        self.source = PrimeWeightSource(dtype)
        disable_nccl_p2p_if_unavailable()
        self.engine = WeightTransferTrainerFactory.trainer_init(
            NCCLTrainerInitInfo(
                master_address=config.host,
                master_port=config.port,
                world_size=1 + sum(worker.world_size for worker in workers),
                rank=self.world.rank,
            ),
            client=self.client,
            source=self.source,
        )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        self.source.set_model(model)
        save_dir = get_step_path(get_broadcast_dir(self.output_dir), step)
        if self.world.is_master:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "STABLE").touch()
            sync_wait_for_path(save_dir / NCCL_READY_MARKER, interval=0.1, log_interval=10)
        if self.world.world_size > 1:
            dist.barrier()
        self.engine.send_weights()
        if self.world.is_master:
            self.client.update_weight_version(str(step))
        get_logger().debug(f"Dynamo workers committed weight version {step}")

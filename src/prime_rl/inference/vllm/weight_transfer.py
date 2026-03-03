"""NCCL weight transfer engine that broadcasts weight metadata (names, dtypes, shapes)
as a JSON header via NCCL before the weight tensors.

Registered as "nccl_prime" backend in vLLM's WeightTransferEngineFactory.
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
)

BACKEND_NAME = "nccl_prime"


@dataclass
class PrimeWeightTransferConfig(WeightTransferConfig):
    backend: str = BACKEND_NAME


class PrimeNCCLWeightTransferEngine(NCCLWeightTransferEngine):
    """Extends NCCLWeightTransferEngine with in-band metadata via NCCL."""

    def receive_weights(
        self,
        update_info: NCCLWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        metadata = _nccl_receive_json(self.model_update_group)
        update_info.names = metadata["names"]
        update_info.dtype_names = metadata["dtype_names"]
        update_info.shapes = metadata["shapes"]
        super().receive_weights(update_info, load_weights)

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | NCCLTrainerSendWeightsArgs,
        metadata: dict | None = None,
    ) -> None:
        if isinstance(trainer_args, dict):
            trainer_args = NCCLTrainerSendWeightsArgs(**trainer_args)
        if metadata is not None:
            _nccl_send_json(trainer_args.group, metadata)
        NCCLWeightTransferEngine.trainer_send_weights(iterator, trainer_args)


def _nccl_send_json(group, data: dict, src: int = 0) -> None:
    raw = json.dumps(data).encode("utf-8")
    size = torch.tensor([len(raw)], dtype=torch.int64, device="cuda")
    group.broadcast(size, src=src, stream=torch.cuda.current_stream())
    payload = torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()
    group.broadcast(payload, src=src, stream=torch.cuda.current_stream())


def _nccl_receive_json(group, src: int = 0) -> dict:
    size = torch.zeros(1, dtype=torch.int64, device="cuda")
    group.broadcast(size, src=src, stream=torch.cuda.current_stream())
    payload = torch.zeros(int(size.item()), dtype=torch.uint8, device="cuda")
    group.broadcast(payload, src=src, stream=torch.cuda.current_stream())
    return json.loads(bytes(payload.cpu().numpy()))


def register():
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    WeightTransferEngineFactory.register_engine(BACKEND_NAME, PrimeNCCLWeightTransferEngine)

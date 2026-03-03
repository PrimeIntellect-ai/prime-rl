"""NCCL weight transfer engine with in-band metadata broadcasting.

Extends vLLM's NCCLWeightTransferEngine to send weight metadata (names, dtypes,
shapes) as a JSON header via NCCL before the weight tensors.

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
    """WeightTransferConfig extended with the nccl_prime backend."""

    backend: str = BACKEND_NAME


class PrimeNCCLWeightTransferEngine(NCCLWeightTransferEngine):
    """NCCL weight transfer with in-band metadata.

    Sends/receives a JSON metadata header via NCCL before the weight tensors,
    so the receiver auto-discovers the weight layout without out-of-band communication.
    """

    def receive_weights(
        self,
        update_info: NCCLWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Receive metadata header via NCCL, then delegate to parent for weight tensors."""
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
        """Broadcast metadata header via NCCL, then delegate to parent for weight tensors."""
        if isinstance(trainer_args, dict):
            trainer_args = NCCLTrainerSendWeightsArgs(**trainer_args)
        if metadata is not None:
            _nccl_send_json(trainer_args.group, metadata)
        NCCLWeightTransferEngine.trainer_send_weights(iterator, trainer_args)


def _nccl_send_json(group, data: dict, src: int = 0) -> None:
    """Broadcast a dict as JSON bytes via NCCL."""
    raw = json.dumps(data).encode("utf-8")
    size = torch.tensor([len(raw)], dtype=torch.int64, device="cuda")
    group.broadcast(size, src=src, stream=torch.cuda.current_stream())
    payload = torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()
    group.broadcast(payload, src=src, stream=torch.cuda.current_stream())


def _nccl_receive_json(group, src: int = 0) -> dict:
    """Receive a JSON dict via NCCL broadcast."""
    size = torch.zeros(1, dtype=torch.int64, device="cuda")
    group.broadcast(size, src=src, stream=torch.cuda.current_stream())
    payload = torch.zeros(int(size.item()), dtype=torch.uint8, device="cuda")
    group.broadcast(payload, src=src, stream=torch.cuda.current_stream())
    return json.loads(bytes(payload.cpu().numpy()))


def register():
    """Register the nccl_prime engine in vLLM's factory."""
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    WeightTransferEngineFactory.register_engine(BACKEND_NAME, PrimeNCCLWeightTransferEngine)

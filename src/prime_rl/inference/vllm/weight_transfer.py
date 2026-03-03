"""NCCL weight transfer engine with in-band metadata broadcasting.

Extends vLLM's NCCLWeightTransferEngine to broadcast weight metadata (names, dtypes,
shapes) as a JSON header via NCCL before the weight tensors. This eliminates the need
for out-of-band metadata passing (files, HTTP) between trainer and inference.

Registered as "nccl_prime" backend in vLLM's WeightTransferEngineFactory.
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
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


@dataclass
class PrimeNCCLUpdateInfo(NCCLWeightTransferUpdateInfo):
    """Update info that receives metadata via NCCL instead of requiring it upfront."""

    names: list[str] = field(default_factory=list)
    dtype_names: list[str] = field(default_factory=list)
    shapes: list[list[int]] = field(default_factory=list)

    def __post_init__(self):
        pass


@dataclass
class PrimeNCCLTrainerSendWeightsArgs(NCCLTrainerSendWeightsArgs):
    """Trainer args extended with explicit metadata to broadcast before weights."""

    metadata: dict | None = None


class PrimeNCCLWeightTransferEngine(NCCLWeightTransferEngine):
    """NCCL weight transfer engine with in-band metadata.

    Sends/receives a JSON metadata header (names, dtype_names, shapes) via NCCL
    before the weight tensors. This allows the receiver to auto-discover the
    weight layout without out-of-band communication.
    """

    update_info_cls = PrimeNCCLUpdateInfo

    def receive_weights(
        self,
        update_info: PrimeNCCLUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Receive metadata header via NCCL, then delegate to parent for weight tensors."""
        metadata = _receive_metadata(self.model_update_group, src=0)
        update_info.names = metadata["names"]
        update_info.dtype_names = metadata["dtype_names"]
        update_info.shapes = metadata["shapes"]
        super().receive_weights(update_info, load_weights)

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | PrimeNCCLTrainerSendWeightsArgs,
    ) -> None:
        """Broadcast metadata header via NCCL, then delegate to parent for weight tensors."""
        if isinstance(trainer_args, dict):
            args = PrimeNCCLTrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args

        if args.metadata is not None:
            _broadcast_metadata(args.group, args.metadata, src=args.src)

        NCCLWeightTransferEngine.trainer_send_weights(iterator, args)


def _broadcast_metadata(group, metadata: dict, src: int = 0) -> None:
    """Broadcast weight metadata as a JSON header via NCCL."""
    metadata_bytes = json.dumps(metadata).encode("utf-8")
    size_tensor = torch.tensor([len(metadata_bytes)], dtype=torch.int64, device="cuda")
    group.broadcast(size_tensor, src=src, stream=torch.cuda.current_stream())
    metadata_tensor = torch.frombuffer(bytearray(metadata_bytes), dtype=torch.uint8).cuda()
    group.broadcast(metadata_tensor, src=src, stream=torch.cuda.current_stream())


def _receive_metadata(group, src: int = 0) -> dict:
    """Receive weight metadata from a JSON header via NCCL."""
    size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
    group.broadcast(size_tensor, src=src, stream=torch.cuda.current_stream())
    metadata_tensor = torch.zeros(int(size_tensor.item()), dtype=torch.uint8, device="cuda")
    group.broadcast(metadata_tensor, src=src, stream=torch.cuda.current_stream())
    return json.loads(bytes(metadata_tensor.cpu().numpy()))


def register():
    """Register the nccl_prime engine in vLLM's factory."""
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    WeightTransferEngineFactory.register_engine(BACKEND_NAME, PrimeNCCLWeightTransferEngine)

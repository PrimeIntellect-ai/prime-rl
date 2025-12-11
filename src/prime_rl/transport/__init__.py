from pathlib import Path

from prime_rl.transport.base import TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.config import TransportConfigType
from prime_rl.transport.filesystem import (
    FileSystemMicroBatchReceiver,
    FileSystemMicroBatchSender,
    FileSystemTrainingBatchReceiver,
    FileSystemTrainingBatchSender,
)
from prime_rl.transport.types import MicroBatch, TrainingBatch, TrainingExample


def setup_transport_sender(output_dir: Path, transport: TransportConfigType) -> TrainingBatchSender:
    if transport.type == "filesystem":
        return FileSystemTrainingBatchSender(output_dir)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_transport_receiver(output_dir: Path, transport: TransportConfigType) -> TrainingBatchReceiver:
    if transport.type == "filesystem":
        return FileSystemTrainingBatchReceiver(output_dir)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


__all__ = [
    "FileSystemTrainingBatchSender",
    "FileSystemTrainingBatchReceiver",
    "FileSystemMicroBatchSender",
    "FileSystemMicroBatchReceiver",
    "TrainingExample",
    "TrainingBatch",
    "MicroBatch",
    "setup_transport_sender",
    "setup_transport_receiver",
]

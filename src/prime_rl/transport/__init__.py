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


def setup_training_batch_sender(output_dir: Path, transport: TransportConfigType) -> TrainingBatchSender:
    if transport.type == "filesystem":
        return FileSystemTrainingBatchSender(output_dir)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_training_batch_receiver(output_dir: Path, transport: TransportConfigType) -> TrainingBatchReceiver:
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
    "setup_training_batch_sender",
    "setup_training_batch_receiver",
]

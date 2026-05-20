from pathlib import Path
from typing import TYPE_CHECKING

from prime_rl.configs.shared import TransportConfig
from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.types import MicroBatch, TrainingBatch, TrainingSample

if TYPE_CHECKING:
    from prime_rl.transport.filesystem import (
        FileSystemMicroBatchReceiver,
        FileSystemMicroBatchSender,
        FileSystemTrainingBatchReceiver,
        FileSystemTrainingBatchSender,
    )
    from prime_rl.transport.ray import (
        RayMicroBatchReceiver,
        RayMicroBatchSender,
        RayTrainingBatchReceiver,
        RayTrainingBatchSender,
    )
    from prime_rl.transport.zmq import (
        ZMQMicroBatchReceiver,
        ZMQMicroBatchSender,
        ZMQTrainingBatchReceiver,
        ZMQTrainingBatchSender,
    )


def __getattr__(name: str):
    # Keep backend classes lazy: importing one transport submodule executes this
    # package __init__, and should not pull every backend dependency.
    if name == "FileSystemTrainingBatchSender":
        from prime_rl.transport.filesystem import FileSystemTrainingBatchSender

        return FileSystemTrainingBatchSender
    if name == "FileSystemTrainingBatchReceiver":
        from prime_rl.transport.filesystem import FileSystemTrainingBatchReceiver

        return FileSystemTrainingBatchReceiver
    if name == "FileSystemMicroBatchSender":
        from prime_rl.transport.filesystem import FileSystemMicroBatchSender

        return FileSystemMicroBatchSender
    if name == "FileSystemMicroBatchReceiver":
        from prime_rl.transport.filesystem import FileSystemMicroBatchReceiver

        return FileSystemMicroBatchReceiver
    if name == "ZMQTrainingBatchSender":
        from prime_rl.transport.zmq import ZMQTrainingBatchSender

        return ZMQTrainingBatchSender
    if name == "ZMQTrainingBatchReceiver":
        from prime_rl.transport.zmq import ZMQTrainingBatchReceiver

        return ZMQTrainingBatchReceiver
    if name == "ZMQMicroBatchSender":
        from prime_rl.transport.zmq import ZMQMicroBatchSender

        return ZMQMicroBatchSender
    if name == "ZMQMicroBatchReceiver":
        from prime_rl.transport.zmq import ZMQMicroBatchReceiver

        return ZMQMicroBatchReceiver
    if name == "RayTrainingBatchSender":
        from prime_rl.transport.ray import RayTrainingBatchSender

        return RayTrainingBatchSender
    if name == "RayTrainingBatchReceiver":
        from prime_rl.transport.ray import RayTrainingBatchReceiver

        return RayTrainingBatchReceiver
    if name == "RayMicroBatchSender":
        from prime_rl.transport.ray import RayMicroBatchSender

        return RayMicroBatchSender
    if name == "RayMicroBatchReceiver":
        from prime_rl.transport.ray import RayMicroBatchReceiver

        return RayMicroBatchReceiver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def setup_training_batch_sender(output_dir: Path, transport: TransportConfig) -> TrainingBatchSender:
    if transport.type == "filesystem":
        from prime_rl.transport.filesystem import FileSystemTrainingBatchSender

        return FileSystemTrainingBatchSender(output_dir)
    elif transport.type == "zmq":
        from prime_rl.transport.zmq import ZMQTrainingBatchSender

        return ZMQTrainingBatchSender(output_dir, transport)
    elif transport.type == "ray":
        from prime_rl.transport.ray import RayTrainingBatchSender

        return RayTrainingBatchSender(output_dir, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_training_batch_receiver(transport: TransportConfig) -> TrainingBatchReceiver:
    if transport.type == "filesystem":
        from prime_rl.transport.filesystem import FileSystemTrainingBatchReceiver

        return FileSystemTrainingBatchReceiver()
    elif transport.type == "zmq":
        from prime_rl.transport.zmq import ZMQTrainingBatchReceiver

        return ZMQTrainingBatchReceiver(transport)
    elif transport.type == "ray":
        from prime_rl.transport.ray import RayTrainingBatchReceiver

        return RayTrainingBatchReceiver(transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_micro_batch_sender(
    output_dir: Path, data_world_size: int, current_step: int, transport: TransportConfig
) -> MicroBatchSender:
    if transport.type == "filesystem":
        from prime_rl.transport.filesystem import FileSystemMicroBatchSender

        return FileSystemMicroBatchSender(output_dir, data_world_size, current_step)
    elif transport.type == "zmq":
        from prime_rl.transport.zmq import ZMQMicroBatchSender

        return ZMQMicroBatchSender(output_dir, data_world_size, current_step, transport)
    elif transport.type == "ray":
        from prime_rl.transport.ray import RayMicroBatchSender

        return RayMicroBatchSender(output_dir, data_world_size, current_step, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_micro_batch_receiver(
    output_dir: Path, data_rank: int, current_step: int, transport: TransportConfig
) -> MicroBatchReceiver:
    if transport.type == "filesystem":
        from prime_rl.transport.filesystem import FileSystemMicroBatchReceiver

        return FileSystemMicroBatchReceiver(output_dir, data_rank, current_step)
    elif transport.type == "zmq":
        from prime_rl.transport.zmq import ZMQMicroBatchReceiver

        return ZMQMicroBatchReceiver(output_dir, data_rank, current_step, transport)
    elif transport.type == "ray":
        from prime_rl.transport.ray import RayMicroBatchReceiver

        return RayMicroBatchReceiver(output_dir, data_rank, current_step, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


__all__ = [
    "FileSystemTrainingBatchSender",
    "FileSystemTrainingBatchReceiver",
    "FileSystemMicroBatchSender",
    "FileSystemMicroBatchReceiver",
    "ZMQTrainingBatchSender",
    "ZMQTrainingBatchReceiver",
    "ZMQMicroBatchSender",
    "ZMQMicroBatchReceiver",
    "RayTrainingBatchSender",
    "RayTrainingBatchReceiver",
    "RayMicroBatchSender",
    "RayMicroBatchReceiver",
    "MicroBatchReceiver",
    "MicroBatchSender",
    "TrainingSample",
    "TrainingBatch",
    "MicroBatch",
    "setup_training_batch_sender",
    "setup_training_batch_receiver",
    "setup_micro_batch_sender",
    "setup_micro_batch_receiver",
]

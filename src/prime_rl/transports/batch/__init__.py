from pathlib import Path

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.types import (
    MicroBatch,
    MicroBatchPlan,
    MMImageRef,
    MMRefs,
    RoutedExperts,
    SamplerSupport,
    StepBatch,
    TrainingSample,
)
from prime_rl.transports.batch.zmq import (
    ZMQBatchReceiver,
    ZMQBatchSender,
)


def setup_batch_sender(output_dir: Path, current_step: int, transport: ZMQTransportConfig) -> BatchSender:
    return ZMQBatchSender(output_dir, current_step, transport)


def setup_batch_receiver(
    output_dir: Path, rank: int, world_size: int, current_step: int, transport: ZMQTransportConfig
) -> BatchReceiver:
    return ZMQBatchReceiver(output_dir, rank, world_size, current_step, transport)


__all__ = [
    "ZMQBatchSender",
    "ZMQBatchReceiver",
    "BatchReceiver",
    "BatchSender",
    "MMImageRef",
    "MMRefs",
    "TrainingSample",
    "MicroBatch",
    "MicroBatchPlan",
    "SamplerSupport",
    "RoutedExperts",
    "StepBatch",
    "setup_batch_sender",
    "setup_batch_receiver",
]

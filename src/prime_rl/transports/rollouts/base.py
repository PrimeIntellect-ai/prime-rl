from abc import ABC, abstractmethod
from pathlib import Path

import msgspec

from prime_rl.transports.rollouts.types import MicroBatch
from prime_rl.utils.logger import get_logger


class MicroBatchSender(ABC):
    """Base class for sending micro batches from the orchestrator to the train workers."""

    def __init__(self, output_dir: Path, data_world_size: int):
        self.logger = get_logger()
        self.encoder = msgspec.msgpack.Encoder()
        self.output_dir = output_dir
        self.data_world_size = data_world_size

    @abstractmethod
    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        """Send grid of micro batches to the trainers."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class MicroBatchReceiver(ABC):
    """Base class for receiving micro batches from the orchestrator."""

    def __init__(self, output_dir: Path, data_rank: int):
        self.logger = get_logger()
        self.decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])
        self.output_dir = output_dir
        self.data_rank = data_rank

    @abstractmethod
    def wait(self) -> None:
        """Wait for a micro batch to be available."""
        pass

    @abstractmethod
    def can_receive(self) -> bool:
        """Check if a micro batch is available."""
        pass

    @abstractmethod
    def receive(self) -> list[MicroBatch]:
        """Receive a micro batch from the orchestrator."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass

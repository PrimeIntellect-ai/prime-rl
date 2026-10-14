from abc import ABC, abstractmethod
from pathlib import Path

import msgspec

from prime_rl.transports.batch.types import MicroBatch
from prime_rl.utils.logger import get_logger


class BatchSender(ABC):
    """Base class for sending micro batches from the orchestrator to the train workers.

    Trainer ranks pull: a step goes out only once every trainer rank requested it, so the
    trainer only ever holds the batch it is training on, and the last step sent is always the
    one the trainer is on."""

    def __init__(self, output_dir: Path, data_world_size: int):
        self.logger = get_logger()
        self.encoder = msgspec.msgpack.Encoder()
        self.output_dir = output_dir
        self.data_world_size = data_world_size

    @abstractmethod
    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> float:
        """Send grid of micro batches to the trainers once every trainer rank requested the step.
        Returns the seconds spent handing the step to the transport, excluding the wait."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class BatchReceiver(ABC):
    """Base class for receiving micro batches from the orchestrator."""

    def __init__(self, output_dir: Path, data_rank: int, rank: int, world_size: int):
        self.logger = get_logger()
        self.decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])
        self.output_dir = output_dir
        self.data_rank = data_rank
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def wait(self) -> None:
        """Request the current step from the orchestrator and block until it is available."""
        pass

    @abstractmethod
    def receive(self) -> list[MicroBatch]:
        """Receive the micro batches ``wait`` requested."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass

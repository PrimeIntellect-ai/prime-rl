from abc import ABC, abstractmethod
from pathlib import Path

import msgspec

from prime_rl.transports.batch.types import MicroBatch
from prime_rl.utils.logger import get_logger


class BatchSender(ABC):
    """Base class for sending micro batches from the orchestrator to the train workers.

    Trainer ranks pull: a rank receives a step only after asking for it, so the
    trainer never holds more than the batch it is training on. A transport holds
    at most one step that no rank has picked up yet, which is what makes
    ``send`` block.
    """

    def __init__(self, output_dir: Path, data_world_size: int):
        self.logger = get_logger()
        self.encoder = msgspec.msgpack.Encoder()
        self.output_dir = output_dir
        self.data_world_size = data_world_size

    @abstractmethod
    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        """Hand the grid of micro batches to the transport. Blocks until every
        trainer rank has picked up the previous step."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class BatchReceiver(ABC):
    """Base class for receiving micro batches from the orchestrator."""

    def __init__(self, output_dir: Path, data_rank: int):
        self.logger = get_logger()
        self.decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])
        self.output_dir = output_dir
        self.data_rank = data_rank

    @abstractmethod
    def wait(self) -> None:
        """Ask the orchestrator for the current step and block until it is available."""
        pass

    @abstractmethod
    def receive(self) -> list[MicroBatch]:
        """Read the micro batches ``wait`` asked for."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass

from abc import ABC, abstractmethod
from pathlib import Path

import msgspec

from prime_rl.transport.types import MicroBatch, TrainingBatch
from prime_rl.utils.logger import get_logger


class TrainingBatchSender(ABC):
    """Base class for sending training examples from orchestrator to trainer."""

    def __init__(self, output_dir: Path):
        self.logger = get_logger()
        self.encoder = msgspec.msgpack.Encoder()
        self.output_dir = output_dir

    @abstractmethod
    def send(self, batch: TrainingBatch) -> None:
        """Send a batch of training examples to the trainer(s).

        Args:
            batch: The batch of training examples with metadata.
        """
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class TrainingBatchReceiver(ABC):
    """Base class for receiving training examples from orchestrator."""

    def __init__(self) -> None:
        self.logger = get_logger()
        self.decoder = msgspec.msgpack.Decoder(type=TrainingBatch)

    @abstractmethod
    def can_receive(self) -> bool:
        """Check if any batch is available."""
        pass

    @abstractmethod
    def receive(self) -> list[TrainingBatch]:
        """Receive available batches from all runs.

        Returns:
            List of training batches with idx field set.
        """
        pass

    def reset_run(self, idx: int) -> None:
        """Reset state for a run index. Called when a run is deleted and replaced.

        Override in subclasses that maintain per-run state.
        """
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class MicroBatchSender(ABC):
    """Base class for sending micro batches to the train workers."""

    def __init__(self, output_dir: Path, data_world_size: int):
        self.logger = get_logger()
        self.encoder = msgspec.msgpack.Encoder()
        self.output_dir = output_dir
        self.data_world_size = data_world_size

    @abstractmethod
    def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        """Send grid of micro batches to the trainers."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class MicroBatchReceiver(ABC):
    """Base class for receiving micro batches from the packer."""

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
        """Receive a micro batch from the trainer."""
        pass

    def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass

"""Shared publication loop for trainer-bound batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from prime_rl.transport.base import TrainingBatchSender
from prime_rl.transport.types import TrainingBatch, TrainingSample

ContextT = TypeVar("ContextT")


class StepProgress(Protocol):
    step: int


@dataclass
class PreparedTrainingBatch(Generic[ContextT]):
    examples: list[TrainingSample]
    context: ContextT


class TrainingBatchSource(Protocol[ContextT]):
    async def next_training_batch(self, step: int) -> PreparedTrainingBatch[ContextT] | None: ...

    async def on_training_batch_sent(self, batch: PreparedTrainingBatch[ContextT], step: int) -> None: ...


class TrainingBatchProducer(Generic[ContextT]):
    """Publish batches from any source and keep its step aligned with the trainer."""

    def __init__(
        self,
        *,
        source: TrainingBatchSource[ContextT],
        sender: TrainingBatchSender,
        progress: StepProgress,
    ) -> None:
        self.source = source
        self.sender = sender
        self.progress = progress

    async def run(self) -> None:
        while True:
            step = self.progress.step
            batch = await self.source.next_training_batch(step)
            if batch is None:
                return
            if not batch.examples:
                raise ValueError(f"Training batch source returned an empty batch for step {step}")

            await self.sender.send(TrainingBatch(examples=batch.examples, step=step))
            self.progress.step += 1
            await self.source.on_training_batch_sent(batch, step)

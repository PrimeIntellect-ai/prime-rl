import shutil
import time

from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.runs import get_runs
from prime_rl.transport import (
    MicroBatchSender,
    TrainingBatch,
    TrainingExample,
    TransportConfigType,
    setup_micro_batch_sender,
    setup_training_batch_receiver,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_rollout_dir


class Packer:
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
        start_step: int = 0,
    ):
        self.logger = get_logger()
        self.runs = get_runs()
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.receiver = setup_training_batch_receiver(config)
        shutil.rmtree(get_rollout_dir(self.runs.output_dir), ignore_errors=True)
        self.sender: MicroBatchSender = setup_micro_batch_sender(
            self.runs.output_dir, dp_world_size, start_step, config
        )

    def get_batch(self) -> dict[int, TrainingBatch]:
        self.runs.check_for_changes()
        batches = self.receiver.receive()
        return {batch.run_idx: batch for batch in batches if batch.run_idx is not None}

    def has_enough_tokens(self, rollouts: dict[int, TrainingBatch]) -> bool:
        tokens = 0
        batches = 1e-5  # Avoid division by zero
        threshold = self.seq_len * self.dp_world_size
        for _rollouts in rollouts.values():
            for rollout in _rollouts.examples:
                tokens += len(rollout.prompt_ids) + len(rollout.completion_ids)
            batches += 1
            estimated_next_batch_tokens = tokens + tokens / batches
            if estimated_next_batch_tokens >= threshold:
                return True
        else:
            self.logger.warning(f"Not enough tokens to pack. Expected {threshold} tokens, got {tokens}")
            return False

    def pack(self):
        training_batches: dict[int, TrainingBatch] = self.get_batch()
        # TODO: Handle timeout case
        while not self.has_enough_tokens(training_batches):
            time.sleep(1)
            training_batches = self.get_batch()

        train_examples: list[TrainingExample] = []
        train_idxs = []
        for idx, training_batch in training_batches.items():
            self.runs.progress[idx].step += 1
            self.runs.progress[idx].total_tokens += sum(
                len(rollout.prompt_ids) + len(rollout.completion_ids) for rollout in training_batch.examples
            )
            self.runs.progress[idx].total_samples += len(training_batch.examples)
            train_examples.extend(training_batch.examples)
            train_idxs.extend([idx] * len(training_batch.examples))
            self.runs.ready_to_update[idx] = True

        # TODO: Handle different temperatures for each run
        some_temperature = next(iter(training_batches.values())).temperature
        micro_batch_grid = prepare_batch(
            rollouts=train_examples,
            temperature=some_temperature,
            seq_len=self.seq_len,
            num_train_workers=self.dp_world_size,
            # idxs=train_idxs, # Needed for lora later
        )

        self.sender.send(micro_batch_grid)

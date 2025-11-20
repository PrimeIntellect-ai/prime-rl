import time
from pathlib import Path

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.runs import get_runs
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_rollout_dir, get_step_path
from prime_rl.utils.vf import Rollout


class Packer:
    def __init__(self, dp_rank: int, seq_len: int, tokenizer: PreTrainedTokenizer):
        self.logger = get_logger()
        self.runs = get_runs()
        self.dp_rank = dp_rank
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def get_rollout_paths(self) -> list[tuple[int, Path]]:
        return [
            (idx, get_step_path(get_rollout_dir(self.runs.get_run_dir(idx)), self.runs.progress[idx].step))
            for idx in self.runs.used_idxs
        ]

    def get_batch(self) -> dict[int, list[Rollout]]:
        rollouts: dict[int, list[Rollout]] = {}
        self.runs.check_for_changes()
        for idx, rollout_path in self.get_rollout_paths():
            if rollout_path.exists():
                rollouts[idx] = torch.load(rollout_path)
        return rollouts

    def pack(self):
        rollouts: dict[int, list[Rollout]] = self.get_batch()
        while len(rollouts) == 0:
            time.sleep(1)
            rollouts = self.get_batch()

        train_rollouts = []
        for idx, rollouts in rollouts.items():
            self.runs.progress[idx].step += 1
            self.runs.progress[idx].total_tokens += 0  # TODO
            self.runs.progress[idx].total_samples += len(rollouts)
            train_rollouts.extend(rollouts)

        all_data_ranks_batches = prepare_batch(
            rollouts=train_rollouts,
            temperature=1.0,  # TODO: get from run config
            tokenizer=self.tokenizer,
            num_train_workers=self.dp_rank,
            seq_len=self.seq_len,
        )

        step_path = get_rollout_dir(self.runs.output_dir) / f"step_{progress.step}"
        step_path.mkdir(parents=True, exist_ok=True)
        for i, batches in enumerate(all_data_ranks_batches):
            batch_path = step_path / f"rank_{i}.pt"
            tmp_path = batch_path.with_suffix(".tmp")
            self.logger.debug(f"Saving rollouts for step {progress.step} for rank {i} to {batch_path}")
            torch.save(batches, tmp_path)
            tmp_path.rename(batch_path)

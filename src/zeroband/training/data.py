from pathlib import Path
import time
from typing import Any, Generator, TypedDict

from pydantic_config import BaseConfig


import torch
from torch.utils.data import IterableDataset, DataLoader

from jaxtyping import Float, Int

from pyarrow import dataset as ds

from zeroband.logger import get_logger
from zeroband.training.world_info import get_world_info


STABLE_FILE = "stable"


class DataConfig(BaseConfig):
    path: str = "datasets/fineweb-edu"
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 2
    timeout: float = 360


class FakeTokenizedDataset(IterableDataset):
    """This is a dummy dataset that generates random sequences of length seq_len and vocab_size"""

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        assert vocab_size > 3, "Vocab size must be greater than 3"
        self.step = 0

    def __iter__(self) -> Generator[dict[str, Any], Any, None]:
        while True:
            # Generate a random length between 1 and self.seq_len
            len_ = torch.randint(1, self.seq_len + 1, (1,)).item()
            input_ids = torch.randint(3, self.vocab_size, (len_,))
            advantages = torch.randn(len_)
            rewards = torch.randn(len_)
            self.step += 1
            yield {"input_ids": input_ids, "advantages": advantages, "rewards": rewards}


def _get_all_files_for_step(step_count: int, path: Path, timeout: float) -> list[Path]:
    """Get all the files for a given step. Waits until the step is created which is indicated by the stable file."""
    logger = get_logger()
    step_path = path / f"step_{step_count}"
    stable_file = step_path / STABLE_FILE

    start_time = time.time()

    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0

    wait_count = 0
    while not stable_file.exists():
        if time.time() - start_time > timeout:
            logger.info("raising timeout")
            raise TimeoutError(f"Timeout waiting for step {step_count} to be created")

        if wait_count % 50 == 0:
            logger.info(f"[data_worker:{worker_id}] Waiting for {stable_file} to be created")

        wait_count += 1

        time.sleep(0.5)

    files = list(step_path.glob("*.parquet"))
    return files


def _should_skip_index(index: int, world_size: int, rank: int, num_workers: int, workers_id: int) -> bool:
    """
    This function is used to skip the index if it is not the responsibility of the current worker.
    It take into account the number of workers as well as rank.

    Its equivalent to checking if index is in samples[rank::world_size][workers_id::num_workers]

    Returns:
        True if the index should be skipped
        False if the index should be processed

    PS: would love to remove this function and use samples[rank::world_size][workers_id::num_workers] but not sure how it would work across pq dataset
    """
    # First, check if the index belongs to this rank (distributed across world_size)
    if (index % world_size) != rank:
        return True

    # Next, compute the position within the rank's subset
    rank_position = index // world_size

    # Check if this position belongs to this worker (distributed across num_workers)
    if (rank_position % num_workers) != workers_id:
        return True

    # If we passed both checks, this index should be processed by this worker
    return False


class ParquetDataset(IterableDataset):
    """
    This call is a wrapper around parquet dataset.

    It can be updated by calling update_files with a list of files. This will thrown away all previous files.

    If the dataset is exhausted, it will wait for new files to be added.
    """

    def __init__(
        self,
        path: Path,
        batch_size: int,
        timeout: float,
        pq_read_bs: int = 64,
    ):
        self._logger = get_logger()
        self._path = path
        self._batch_size = batch_size
        self._pq_read_bs = pq_read_bs

        self._world_info = get_world_info()

        self._step_count = -1
        self._timeout = timeout

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        assert self._batch_size % (self._world_info.world_size * worker_info.num_workers) == 0, (
            "Batch size must be divisible by the number of workers time the world size"
        )
        # this assert should never be triggered because we check for it in the top config level. Keep it here for sanity

        target_sample_count_per_batch = self._batch_size // (self._world_info.world_size * worker_info.num_workers)

        self._logger.info(f"num_workers: {num_workers}, target_sample_count_per_batch: {target_sample_count_per_batch}")

        while True:
            self._step_count += 1

            sample_count = 0

            self._logger.debug(msg=f"data: Processing step {self._step_count}")

            files = _get_all_files_for_step(self._step_count, self._path, self._timeout)

            # we are NOT splitting the files across datalaoder workers and rank like we did for intellect 1
            # This is because we cannot assume that the files would have the same number of samples each.
            # What we rather do here is that all the workers go over all the files and only yield some of them
            # this is unoptimal because they all load more data that they should, but since the data is already tokenized it should not be a big deal

            dataset = ds.dataset(files, format="parquet")
            # Set up a scanner with just the required columns
            required_columns = ["output_tokens", "advantages", "rewards"]

            scanner = dataset.scanner(columns=required_columns, batch_size=self._pq_read_bs)

            counter = 0

            for j, batch in enumerate(scanner.to_batches()):
                # Check if both required columns exist in this batch

                if all(col in batch.column_names for col in required_columns):
                    output_tokens = batch["output_tokens"]
                    advantages = batch["advantages"]
                    rewards = batch["rewards"]

                    for token, advantage, reward in zip(output_tokens, advantages, rewards):
                        counter += 1
                        if not _should_skip_index(
                            index=counter,
                            world_size=self._world_info.world_size,
                            rank=self._world_info.rank,
                            num_workers=num_workers,
                            workers_id=worker_id,
                        ):
                            try:
                                ids = torch.tensor(token.as_py())
                                adv = torch.tensor(data=[advantage.as_py()] * len(ids))
                                rew = torch.tensor(data=[reward.as_py()] * len(ids))
                                data = {"input_ids": ids, "advantages": adv, "rewards": rew}
                            except Exception as e:
                                self._logger.warn(f"Error processing row {counter} sample {sample_count}: {str(e)}")
                                data = None

                            if data is not None:
                                sample_count += 1
                                yield data

                            if sample_count >= target_sample_count_per_batch:
                                break
                else:
                    self._logger.warn(f"Batch {j} does not have the required columns")

                if sample_count >= target_sample_count_per_batch:
                    # need to break out of a second time because of the nested for loop
                    break


class BatchOutput(TypedDict):
    input_ids: Int[torch.Tensor, "batch seq"]
    advantages: Float[torch.Tensor, "batch seq"]
    rewards: Float[torch.Tensor, "batch seq"]


class PaddingColate:
    def __init__(self, seq_len: int, pad_token_id: int) -> None:
        self._seq_len = seq_len
        self._pad_token_id = pad_token_id

    def __call__(self, samples: list[dict[str, torch.LongTensor]]) -> BatchOutput:
        assert samples[0].keys() == {"input_ids", "advantages", "rewards"}, f"samples[0].keys() == {samples[0].keys()}"

        inputs_ids = []
        advantages = []
        rewards = []
        for sample in samples:
            ids = sample["input_ids"]
            adv = sample["advantages"]
            rew = sample["rewards"]

            if len(ids) >= self._seq_len:
                ids = ids[: self._seq_len]
                adv = adv[: self._seq_len]
                rew = rew[: self._seq_len]
            else:
                ids = torch.cat([ids, torch.full((self._seq_len - len(ids),), fill_value=self._pad_token_id, dtype=ids.dtype)])
                adv = torch.cat([adv, torch.zeros(self._seq_len - len(adv), dtype=adv.dtype)])
                rew = torch.cat([rew, torch.zeros(self._seq_len - len(rew), dtype=rew.dtype)])

            inputs_ids.append(ids)
            advantages.append(adv)
            rewards.append(rew)

        return {
            "input_ids": torch.stack(inputs_ids, dim=0),
            "advantages": torch.stack(advantages, dim=0),
            "rewards": torch.stack(rewards, dim=0),
        }


def get_dataloader(tokenizer, micro_batch_size: int, batch_size: int, data_config: DataConfig) -> DataLoader[BatchOutput]:
    """Get a dataloader for the training dataset"""
    if data_config.fake:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, len(tokenizer))
    else:
        train_dataset = ParquetDataset(Path(data_config.path), batch_size, data_config.timeout)

    collate_fn = PaddingColate(data_config.seq_length, tokenizer.pad_token_id)  # todo adjust padding token for qwen later
    return DataLoader(train_dataset, batch_size=micro_batch_size, num_workers=data_config.num_workers, collate_fn=collate_fn)

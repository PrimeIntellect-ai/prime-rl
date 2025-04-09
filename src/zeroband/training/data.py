from pathlib import Path
import time
from typing import Any, Generator, Iterator, TypedDict

from pydantic_config import BaseConfig


import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist

from jaxtyping import Float, Int

from pyarrow import dataset as ds

from zeroband.logger import get_logger
from zeroband.training.data_prefetch import GCPPrefetcher
from zeroband.training.world_info import get_world_info
from zeroband.training import envs


STABLE_FILE = "stable"


class DataConfig(BaseConfig):
    path: str = "datasets/fineweb-edu"
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 2
    timeout: float = 3600

    local_dir: str = "/dev/shm/zeroband/data"  # only used if path is gcp


class DatasetOutput(TypedDict):
    input_ids: Int[torch.Tensor, "seq"]
    advantages: Float[torch.Tensor, "seq"]
    rewards: Float[torch.Tensor, "seq"]
    loss_mask: Int[torch.Tensor, "seq"]
    logprobs: Float[torch.Tensor, "seq"]
    seq_lens: Int[torch.Tensor, "seq"]
    length_penalties: Float[torch.Tensor, "seq"]
    target_lengths: Int[torch.Tensor, "seq"]
    task_rewards: Float[torch.Tensor, "seq"]


class FakeTokenizedDataset(IterableDataset):
    """A dummy dataset that generates random sequences with the full schema including new columns."""

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        assert vocab_size > 3, "Vocab size must be greater than 3"
        self.step = 0

    def __iter__(self) -> Generator[DatasetOutput, Any, None]:
        while True:
            world_info = get_world_info()

            # we divide by local world rank to simulate imbalanced in the data
            seq_len = self.seq_len // (1 + world_info.local_rank)

            len_ = torch.randint(1, seq_len + 1, (1,)).item()
            input_ids = torch.randint(3, self.vocab_size, (len_,))
            advantages = torch.randn(len_)
            rewards = torch.clamp(torch.randn(len_), min=0.0, max=1.0)
            task_rewards = torch.rand(len_)
            length_penalties = torch.rand(len_)
            target_length_value = torch.randint(1, self.seq_len + 1, (1,)).item()
            target_lengths = torch.tensor(target_length_value, dtype=torch.int32)
            self.step += 1
            yield {
                "input_ids": input_ids,
                "advantages": advantages,
                "rewards": rewards,
                "task_rewards": task_rewards,
                "length_penalties": length_penalties,
                "target_lengths": target_lengths,
                "loss_mask": torch.ones(len_).int(),
                "logprobs": torch.randn(len_),
            }


def _get_dataset_from_files_step(step_count: int, path: Path, timeout: float, batch_size: int, ignore_zero_advantages: bool) -> ds.Dataset:
    """Get all the files for a given step. Waits until the step is created which is indicated by the stable file."""
    logger = get_logger()
    step_path = path / f"step_{step_count}"

    start_time = time.time()

    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0

    wait_count = 0

    while True:
        files = list(step_path.glob("*.parquet"))
        if envs.TRAINING_ENABLE_ACCEPTED_CHECK:
            accepted_flags = set(i.stem for i in step_path.glob("accepted/*.parquet"))
            files = [i for i in files if i.stem in accepted_flags]

        rows = 0
        if len(files) > 0:
            try:
                dataset = ds.dataset(files, format="parquet")

                if ignore_zero_advantages:
                    dataset = dataset.filter(ds.field("advantages") != 0)

                rows = dataset.count_rows()
            except Exception as e:
                logger.warn(f"Error loading dataset for step {step_count}: {e}, files: {files}")
                rows = 0

            if rows >= batch_size:
                logger.info(f"Dataset for step {step_count} has enough samples. rows: {rows} and {len(files)} files")
                return dataset

        if time.time() - start_time > timeout:
            logger.info("raising timeout")
            raise TimeoutError(f"Timeout waiting for step {step_count} to be created")

        if wait_count % 50 == 0:
            logger.info(
                f"[data_worker:{worker_id}] Waiting for {step_path} to have enough samples. len(files): {len(files)}, Current rows: {rows}, target: {batch_size}"
            )

        wait_count += 1
        time.sleep(0.5)


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
        self, path: Path, batch_size: int, timeout: float, step_count_init: int, ignore_zero_advantages: bool, pq_read_bs: int = 64
    ):
        self._logger = get_logger()
        self._path = path
        self._batch_size = batch_size
        self._pq_read_bs = pq_read_bs

        self._world_info = get_world_info()

        self._step_count = step_count_init - 1  # we immediatly bump the step count by one later
        self._timeout = timeout

        self._ignore_zero_advantages = ignore_zero_advantages

    def __iter__(self) -> Generator[DatasetOutput, Any, None]:
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

            dataset = _get_dataset_from_files_step(
                self._step_count, self._path, self._timeout, self._batch_size, self._ignore_zero_advantages
            )

            required_columns = [
                "input_tokens",
                "output_tokens",
                "advantages",
                "rewards",
                "task_rewards",
                "length_penalties",
                "target_lengths",
                "input_logprobs",
                "output_logprobs",
            ]

            scanner = dataset.scanner(columns=required_columns, batch_size=self._pq_read_bs)
            counter = 0

            for j, batch in enumerate(scanner.to_batches()):
                if all(col in batch.column_names for col in required_columns):
                    input_tokens = batch["input_tokens"]
                    output_tokens = batch["output_tokens"]
                    advantages = batch["advantages"]
                    rewards = batch["rewards"]
                    task_rewards = batch["task_rewards"]
                    length_penalties = batch["length_penalties"]
                    target_lengths = batch["target_lengths"]
                    input_logprobs = batch["input_logprobs"]
                    output_logprobs = batch["output_logprobs"]

                    for (
                        in_token,
                        out_token,
                        in_logprob,
                        out_logprob,
                        advantage,
                        reward,
                        task_rew,
                        len_pen,
                        tgt_len,
                    ) in zip(
                        input_tokens,
                        output_tokens,
                        input_logprobs,
                        output_logprobs,
                        advantages,
                        rewards,
                        task_rewards,
                        length_penalties,
                        target_lengths,
                    ):
                        counter += 1
                        if _should_skip_index(
                            index=counter,
                            world_size=self._world_info.world_size,
                            rank=self._world_info.rank,
                            num_workers=num_workers,
                            workers_id=worker_id,
                        ):
                            continue

                        try:
                            input_ids = torch.tensor(in_token.as_py())
                            output_ids = torch.tensor(out_token.as_py())
                            in_logprobs = torch.tensor(in_logprob.as_py())
                            out_logprobs = torch.tensor(out_logprob.as_py())

                            ids = torch.cat([input_ids, output_ids], dim=0)
                            logprobs = torch.cat([in_logprobs, out_logprobs], dim=0)
                            loss_mask = torch.cat([torch.zeros(len(input_ids)), torch.ones(len(output_ids))], dim=0).int()

                            adv_value = advantage.as_py()
                            reward_value = reward.as_py()
                            task_value = task_rew.as_py()
                            len_pen_value = len_pen.as_py()

                            tgt_length_val = int(tgt_len.as_py())

                            adv = torch.tensor([adv_value] * len(ids))  # advantage
                            rew = torch.tensor([reward_value] * len(ids))  # reward
                            t_rew = torch.tensor([task_value] * len(ids))  # task reward
                            l_pen = torch.tensor([len_pen_value] * len(ids))  # length penalty

                            data = {
                                "input_ids": ids,
                                "advantages": adv,
                                "rewards": rew,
                                "task_rewards": t_rew,
                                "length_penalties": l_pen,
                                "target_lengths": tgt_length_val,
                                "loss_mask": loss_mask,
                                "logprobs": logprobs,
                            }
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
                    break


def no_collate(batch: list[DatasetOutput]) -> list[DatasetOutput]:
    return batch


class BatchOutput(TypedDict):
    input_ids: Int[torch.Tensor, "batch seq"]
    advantages: Float[torch.Tensor, "batch seq"]
    rewards: Float[torch.Tensor, "batch seq"]
    loss_mask: Int[torch.Tensor, "batch seq"]
    logprobs: Float[torch.Tensor, "batch seq"]
    seq_lens: Int[torch.Tensor, "batch"]
    length_penalties: Float[torch.Tensor, "batch"]
    # target_lengths: Int[torch.Tensor, "batch"]
    task_rewards: Float[torch.Tensor, "batch"]
    position_ids: Int[torch.Tensor, "batch seq"]


def get_dataloader(
    tokenizer, local_batch_size: int, batch_size: int, data_config: DataConfig, step_count_init: int, ignore_zero_advantages: bool
) -> tuple[DataLoader[BatchOutput], GCPPrefetcher | None]:
    """Get a dataloader for the training dataset"""

    prefetcher = None
    path = data_config.path

    if "gs" in data_config.path:
        if get_world_info().local_rank == 0:
            prefetcher = GCPPrefetcher(data_config.path, data_config.local_dir)
        path = data_config.local_dir

    if data_config.fake:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, len(tokenizer))
    else:
        train_dataset = ParquetDataset(Path(path), batch_size, data_config.timeout, step_count_init, ignore_zero_advantages)

    loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        num_workers=data_config.num_workers,
        collate_fn=no_collate,
    )
    return loader, prefetcher


def packed_batch(batch_optim: list[DatasetOutput], seq_len: int, pad_token_id: int, micro_batch_size: int) -> tuple[list[BatchOutput], int]:
    """
    this function will pack the batch into [1, seq_len] microbatch tensors with positions ids for calling fa2 with sequence packing
    """

    def pack_batch(
        batch_optim: list[DatasetOutput], seq_len: int, pad_token_id: int, micro_batch_size: int
    ) -> Generator[BatchOutput, Any, None]:
        batch_seq_len = seq_len * micro_batch_size

        required_keys = {
            "input_ids",
            "advantages",
            "rewards",
            "task_rewards",
            "length_penalties",
            # "target_lengths",
            "loss_mask",
            "logprobs",
        }

        input_ids = []
        advantages = []
        rewards = []
        task_rewards = []
        length_penalties = []
        # target_lens = []
        loss_masks = []
        logprobs = []
        seq_lens = []
        seq_len_sum = 0

        batch_iter: Iterator[tuple[int, DatasetOutput]] = enumerate(batch_optim)
        pending_sample: tuple[int, DatasetOutput] | None = None
        while True:
            # Get the next sample
            if pending_sample is not None:
                i, sample = pending_sample
                pending_sample = None
            else:
                try:
                    i, sample = next(batch_iter)
                except StopIteration:
                    break

            seq_len = len(sample["input_ids"])
            input_dtype = sample["input_ids"].dtype
            if seq_len > batch_seq_len:
                get_logger().info(f"Sample {i} too long. seq_len: {seq_len} > {batch_seq_len}. Skipping.")
                sample["input_ids"] = sample["input_ids"][:batch_seq_len]
                sample["advantages"] = sample["advantages"][:batch_seq_len]
                sample["rewards"] = sample["rewards"][:batch_seq_len]
                sample["task_rewards"] = sample["task_rewards"][:batch_seq_len]
                sample["length_penalties"] = sample["length_penalties"][:batch_seq_len]
                # sample["target_lengths"] = ...
                sample["loss_mask"] = sample["loss_mask"][:batch_seq_len]
                sample["logprobs"] = sample["logprobs"][:batch_seq_len]
                seq_len = len(sample["input_ids"])

            assert required_keys <= set(sample.keys()), f"Missing required keys. Found: {sample.keys()}, required: {required_keys}"
            assert (
                len(sample["input_ids"])
                == len(sample["advantages"])
                == len(sample["rewards"])
                == len(sample["task_rewards"])
                == len(sample["length_penalties"])
                == len(sample["loss_mask"])
                == len(sample["logprobs"])
            ), (
                f"Sample {i} has different lengths: {len(sample['input_ids'])}, {len(sample['advantages'])}, {len(sample['rewards'])}, {len(sample['task_rewards'])}, {len(sample['length_penalties'])}, {len(sample['loss_mask'])}, {len(sample['logprobs'])}"
            )

            # If the sample fits, add it to the batch.
            if (seq_len_sum + seq_len) <= batch_seq_len:
                input_ids.append(sample["input_ids"])
                advantages.append(sample["advantages"])
                rewards.append(sample["rewards"])
                task_rewards.append(sample["task_rewards"])
                length_penalties.append(sample["length_penalties"])
                # target_lens.append(sample["target_lengths"])
                loss_masks.append(sample["loss_mask"])
                logprobs.append(sample["logprobs"])
                seq_lens.append(seq_len)
                seq_len_sum += seq_len

            # Otherwise, pad the batch and yield what we've built.
            # Then on the next iteration, we process the sample that was too long again.
            else:
                pending_sample = (i, sample)

                # Pad. We don't append to seq_lens so not to attend to the padding.
                padding_len = batch_seq_len - seq_len_sum
                input_ids.append(torch.full((padding_len,), fill_value=pad_token_id, dtype=input_dtype))
                advantages.append(torch.zeros(padding_len, dtype=sample["advantages"].dtype))
                rewards.append(torch.zeros(padding_len, dtype=sample["rewards"].dtype))
                task_rewards.append(torch.zeros(padding_len, dtype=sample["task_rewards"].dtype))
                length_penalties.append(torch.zeros(padding_len, dtype=sample["length_penalties"].dtype))
                # target_lens.append(padding_len)
                loss_masks.append(torch.zeros(padding_len, dtype=sample["loss_mask"].dtype))
                logprobs.append(torch.zeros(padding_len, dtype=sample["logprobs"].dtype))
                seq_lens.append(padding_len)  # Append fake padding sequence b/c flash attention explodes otherwise or when it's all zeros.

                # Yield the microbatch and reset so we can make a new one.
                position_ids = torch.cat([torch.arange(0, sl, dtype=input_dtype) for sl in seq_lens])
                yield {
                    "input_ids": torch.cat(input_ids),
                    "advantages": torch.cat(advantages),
                    "rewards": torch.cat(rewards),
                    "task_rewards": torch.cat(task_rewards),
                    "length_penalties": torch.cat(length_penalties),
                    # "target_lengths": torch.tensor(target_lens, dtype=torch.int32),
                    "loss_mask": torch.cat(loss_masks).int(),
                    "logprobs": torch.cat(logprobs),
                    "seq_lens": torch.tensor(seq_lens),
                    "position_ids": position_ids,
                }

                input_ids = []
                advantages = []
                rewards = []
                task_rewards = []
                length_penalties = []
                # target_lens = []
                loss_masks = []
                logprobs = []
                seq_lens = []
                seq_len_sum = 0

        padding_len = batch_seq_len - seq_len_sum
        input_ids.append(torch.full((padding_len,), fill_value=pad_token_id, dtype=input_dtype))
        advantages.append(torch.zeros(padding_len, dtype=sample["advantages"].dtype))
        rewards.append(torch.zeros(padding_len, dtype=sample["rewards"].dtype))
        task_rewards.append(torch.zeros(padding_len, dtype=sample["task_rewards"].dtype))
        length_penalties.append(torch.zeros(padding_len, dtype=sample["length_penalties"].dtype))
        # target_lens.append(padding_len)
        loss_masks.append(torch.zeros(padding_len, dtype=sample["loss_mask"].dtype))
        logprobs.append(torch.zeros(padding_len, dtype=sample["logprobs"].dtype))
        seq_lens.append(padding_len)

        position_ids = torch.cat([torch.arange(0, sl, dtype=input_dtype) for sl in seq_lens])
        yield {
            "input_ids": torch.cat(input_ids),
            "advantages": torch.cat(advantages),
            "rewards": torch.cat(rewards),
            "task_rewards": torch.cat(task_rewards),
            "length_penalties": torch.cat(length_penalties),
            # "target_lengths": torch.tensor(target_lens, dtype=torch.int32),
            "loss_mask": torch.cat(loss_masks).int(),
            "logprobs": torch.cat(logprobs),
            "seq_lens": torch.tensor(seq_lens),
            "position_ids": position_ids,
        }

        return  # Batch exhausted

    batch_outputs = []
    for packed in pack_batch(batch_optim, seq_len, pad_token_id, micro_batch_size):
        packed = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in packed.items()}
        batch_outputs.append(packed)  # Re-coallated

    num_grad_acc_steps = len(batch_outputs)

    max_grad_acc_step = torch.tensor(num_grad_acc_steps).to("cuda")

    dist.all_reduce(max_grad_acc_step, op=dist.ReduceOp.MAX, group=None)

    empty_batch_count = (max_grad_acc_step - num_grad_acc_steps).item()

    for _ in range(empty_batch_count):
        empty_batch = {}

        for key, value in batch_outputs[0].items():
            empty_batch[key] = torch.zeros_like(value)

        batch_outputs.append(empty_batch)

    return batch_outputs, max_grad_acc_step.item()

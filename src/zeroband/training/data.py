import time
from pathlib import Path
from typing import Any, Generator, Literal, TypeAlias, TypedDict

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from jaxtyping import Float, Int
from pyarrow import dataset as ds
from pydantic_config import BaseConfig
from torch.utils.data import DataLoader, IterableDataset

from zeroband.training import envs
from zeroband.training.data_prefetch import STABLE_FILE, GCPPrefetcher
from zeroband.training.world_info import get_world_info
from zeroband.utils.logger import get_logger
from zeroband.utils.parquet import pa_schema


class DataConfig(BaseConfig):
    path: str = "datasets/fineweb-edu"
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 1
    timeout: float = 3600

    local_dir: str = "/dev/shm/zeroband/data"  # only used if path is gcp

    ignore_zero_advantages: bool = False  # don't use in local setup


class DatasetOutput(TypedDict):
    # token level
    input_ids: Int[torch.Tensor, "seq"]
    advantages: Float[torch.Tensor, "seq"]
    loss_mask: Int[torch.Tensor, "seq"]

    # sample level
    seq_lens: Int[torch.Tensor, "1"]
    rewards: Float[torch.Tensor, "1"]
    task_rewards: Float[torch.Tensor, "1"]
    length_penalties: Float[torch.Tensor, "1"]
    target_lengths: Int[torch.Tensor, "1"]


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
            self.step += 1

            yield {
                "input_ids": input_ids,
                "advantages": advantages,
                "rewards": 0.5,
                "loss_mask": torch.ones(len_).int(),
                "task_rewards": 0.5,
                "length_penalties": 0.5,
                "target_lengths": seq_len,
            }


def validate_schema_pa_file(file: Path):
    """Check if the schema of the parquet file is the same as the schema of the pa_schema"""
    try:
        parquet_schema = pq.read_schema(file)
        return parquet_schema.equals(pa_schema)
    except Exception as e:
        print(f"Error reading schema for file {file}: {e}")
        return False


def _get_dataset_from_files_step(
    step_count: int, path: Path, timeout: float, batch_size: int, ignore_zero_advantages: bool, use_stable_file: bool
) -> ds.Dataset:
    """Get all the files for a given step. Waits until the step is created which is indicated by the stable file."""
    logger = get_logger("TRAIN")
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
                for file in files:
                    if not validate_schema_pa_file(file):
                        raise ValueError(f"Schema of file {file} is not the same as the schema of the pa_schema")

                dataset = ds.dataset(files, format="parquet")

                if ignore_zero_advantages:
                    dataset = dataset.filter(ds.field("advantages") != 0)

                rows = dataset.count_rows()
            except Exception as e:
                logger.warn(f"Error loading dataset for step {step_count}: {e}, files: {files}")
                rows = 0

            if rows >= batch_size:
                logger.info(f"Dataset for step {step_count} has enough samples. rows: {rows} and {len(files)} files")

                if use_stable_file:
                    stable_file = step_path / STABLE_FILE
                    if stable_file.exists():
                        logger.info(f"Stable file {stable_file} exists for step {step_count}, returning dataset")
                        return dataset
                else:
                    return dataset

        if time.time() - start_time > timeout:
            logger.info("raising timeout")
            raise TimeoutError(f"Timeout waiting for step {step_count} to be created")

        if wait_count % 600 == 0:  # log every 5 minutes
            logger.info(
                f"[data_worker:{worker_id}] Waiting for {step_path} to have enough samples. len(files): {len(files)}, Current rows: {rows}, target: {batch_size}"
            )
            if use_stable_file:
                stable_file = step_path / STABLE_FILE
                if not stable_file.exists():
                    logger.info(f"Stable file {stable_file} does not exist for step {step_count}, waiting for it to be created")

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
        self,
        path: Path,
        batch_size: int,
        timeout: float,
        step_count_init: int,
        ignore_zero_advantages: bool,
        pq_read_bs: int = 64,
        use_stable_file: bool = False,
    ):
        self._logger = get_logger("TRAIN")
        self._path = path
        self._batch_size = batch_size
        self._pq_read_bs = pq_read_bs

        self._world_info = get_world_info()

        self._step_count = step_count_init - 1  # we immediately bump the step count by one later
        self._timeout = timeout

        self._ignore_zero_advantages = ignore_zero_advantages

        self._use_stable_file = use_stable_file

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
                self._step_count, self._path, self._timeout, self._batch_size, self._ignore_zero_advantages, self._use_stable_file
            )

            required_columns = [
                "input_tokens",
                "output_tokens",
                "advantages",
                "rewards",
                "task_rewards",
                "length_penalties",
                "target_lengths",
            ]

            scanner = dataset.scanner(columns=required_columns, batch_size=self._pq_read_bs)
            counter = 0

            for j, batch in enumerate(scanner.to_batches()):
                if all(col in batch.column_names for col in required_columns):
                    for (
                        in_token,
                        out_token,
                        advantage,
                        reward,
                        task_reward,
                        length_penalty,
                        target_length,
                    ) in zip(
                        batch["input_tokens"],
                        batch["output_tokens"],
                        batch["advantages"],
                        batch["rewards"],
                        batch["task_rewards"],
                        batch["length_penalties"],
                        batch["target_lengths"],
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

                            ids = torch.cat([input_ids, output_ids], dim=0)
                            loss_mask = torch.cat([torch.zeros(len(input_ids)), torch.ones(len(output_ids))], dim=0).int()

                            adv_value = advantage.as_py()
                            reward_value = reward.as_py()

                            adv = torch.tensor([adv_value] * len(ids))  # advantage

                            data = {
                                "input_ids": ids,
                                "advantages": adv,
                                "rewards": reward_value,
                                "loss_mask": loss_mask,
                                "task_rewards": task_reward.as_py(),
                                "length_penalties": length_penalty.as_py(),
                                "target_lengths": target_length.as_py(),
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


def get_dataloader(
    tokenizer,
    local_batch_size: int,
    batch_size: int,
    data_config: DataConfig,
    step_count_init: int,
) -> tuple[DataLoader[list[DatasetOutput]], GCPPrefetcher | None]:
    """Get a dataloader for the training dataset"""

    """Get a dataloader for the training dataset"""

    prefetcher = None
    path = data_config.path

    use_stable_file = False
    if "gs" in data_config.path:
        use_stable_file = True
        if get_world_info().rank == 0:
            prefetcher = GCPPrefetcher(data_config.path, data_config.local_dir)
        path = data_config.local_dir

    if data_config.fake:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, len(tokenizer))
    else:
        train_dataset = ParquetDataset(
            Path(path),
            batch_size,
            data_config.timeout,
            step_count_init,
            data_config.ignore_zero_advantages,
            use_stable_file=use_stable_file,
        )

    loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        num_workers=data_config.num_workers,
        collate_fn=no_collate,
    )
    return loader, prefetcher


class BatchOutput(TypedDict):
    # token level
    input_ids: Int[torch.Tensor, "batch seq"]
    advantages: Float[torch.Tensor, "batch seq"]
    loss_mask: Int[torch.Tensor, "batch seq"]
    position_ids: Int[torch.Tensor, "batch seq"]

    # sample level
    seq_lens: Int[torch.Tensor, "sample"]
    rewards: Float[torch.Tensor, "sample"]
    task_rewards: Float[torch.Tensor, "sample"]
    length_penalties: Float[torch.Tensor, "sample"]
    target_lengths: Int[torch.Tensor, "sample"]


### colate


def collate_fn(samples: list[DatasetOutput], max_seq_len: int, pad_token_id: int) -> BatchOutput:
    """
    This take a list of samples that should be packed together along the sequence dimension. Will add padding at the end of needed and
    clipped to max_seq_len
    """

    total_len = sum(len(sample["input_ids"]) for sample in samples)

    inputs_ids = [sample["input_ids"] for sample in samples]
    advantages = [sample["advantages"] for sample in samples]
    rewards = [sample["rewards"] for sample in samples]
    loss_masks = [sample["loss_mask"] for sample in samples]
    task_rewards = [sample["task_rewards"] for sample in samples]
    length_penalties = [sample["length_penalties"] for sample in samples]
    target_lengths = [sample["target_lengths"] for sample in samples]

    seq_lens = [len(sample["input_ids"]) for sample in samples]
    position_ids = [torch.arange(0, len(sample["input_ids"]), dtype=torch.int32) for sample in samples]

    if total_len < max_seq_len:
        padding_len = max_seq_len - total_len

        inputs_ids.append(torch.full((padding_len,), fill_value=pad_token_id, dtype=inputs_ids[0].dtype))
        advantages.append(torch.zeros(padding_len, dtype=advantages[0].dtype))
        loss_masks.append(torch.zeros(padding_len, dtype=loss_masks[0].dtype).int())
        position_ids.append(torch.arange(0, padding_len, dtype=torch.int32))

    return {
        # token level
        "input_ids": torch.cat(inputs_ids, dim=0)[:max_seq_len].unsqueeze(0),
        "advantages": torch.cat(advantages, dim=0)[:max_seq_len].unsqueeze(0),
        "loss_mask": torch.cat(loss_masks, dim=0)[:max_seq_len].unsqueeze(0),
        "position_ids": torch.cat(position_ids, dim=0)[:max_seq_len].unsqueeze(0),
        # sample level
        "rewards": torch.tensor(rewards),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32),
        "task_rewards": torch.tensor(task_rewards),
        "length_penalties": torch.tensor(length_penalties),
        "target_lengths": torch.tensor(target_lengths),
    }


### sequence packing


def pack_datatset_outputs_efficiently(batch_optim: list[DatasetOutput], max_seq_len: int) -> list[list[DatasetOutput]]:
    """
    This function will pack the batch into a single batch in a efficient manner
    """
    ## we sorted by inputs_ids

    batch_with_len = [(len(sample["input_ids"]), sample) for sample in batch_optim]

    sorted_batch = sorted(batch_with_len, key=lambda x: x[0], reverse=True)

    ## we create bins
    batches: list[list[DatasetOutput]] = []

    ## we pack the bins

    for seq_len, sample in sorted_batch:
        # Try to find a bin that can fit this sequence
        bin_found = False
        for bin_idx, bin_content in enumerate(batches):
            # Calculate current bin length
            bin_len = sum(len(s["input_ids"]) for s in bin_content)
            # Check if sequence fits in this bin
            if bin_len + seq_len <= max_seq_len:
                batches[bin_idx].append(sample)
                bin_found = True
                break

        # If no suitable bin found, create a new bin
        if not bin_found:
            batches.append([sample])

    return batches


def data_parallel_rebalancing(micro_batches: list[BatchOutput]) -> list[BatchOutput]:
    """
    This function will duplicate the first micro_batch to match the number of grad acc steps on each gpu
    Otherwise will block FSDP forward and backward all gather.
    """
    num_grad_acc_steps = len(micro_batches)

    max_grad_acc_step = num_grad_acc_steps
    if dist.is_initialized():
        max_grad_acc_step = torch.tensor(num_grad_acc_steps, dtype=torch.int32).to("cuda")
        dist.all_reduce(max_grad_acc_step, op=dist.ReduceOp.MAX, group=None)
        max_grad_acc_step = int(max_grad_acc_step.item())

    empty_batch_count = max_grad_acc_step - num_grad_acc_steps

    for _ in range(empty_batch_count):
        empty_batch = {}

        for key, value in micro_batches[0].items():
            if isinstance(value, torch.Tensor):
                empty_batch[key] = value.clone()
            else:
                empty_batch[key] = value

        micro_batches.append(empty_batch)

    return micro_batches


def packed_batch_packing(batch_optim: list[DatasetOutput], max_seq_len: int, pad_token_id: int, micro_bs: int) -> list[BatchOutput]:
    """
    this function will pack the batch into [1, seq_len] microbatch tensors with positions ids for calling fa2 with sequence packing
    """
    max_seq_len = max_seq_len * micro_bs

    batches = pack_datatset_outputs_efficiently(batch_optim, max_seq_len=max_seq_len)

    micro_batches = [collate_fn(bin, pad_token_id=pad_token_id, max_seq_len=max_seq_len) for bin in batches]

    return data_parallel_rebalancing(micro_batches)


def merge_batches_padding(batches: list[BatchOutput]) -> list[BatchOutput]:
    return {
        # token level
        "input_ids": torch.cat([b["input_ids"] for b in batches], dim=0),
        "advantages": torch.cat([b["advantages"] for b in batches], dim=0),
        "rewards": torch.cat([b["rewards"] for b in batches], dim=0),
        "loss_mask": torch.cat([b["loss_mask"] for b in batches], dim=0),
        "position_ids": torch.cat([b["position_ids"] for b in batches], dim=0),
        # sample level
        "seq_lens": torch.cat([b["seq_lens"] for b in batches]),
        "task_rewards": torch.cat([b["task_rewards"] for b in batches]),
        "length_penalties": torch.cat([b["length_penalties"] for b in batches]),
        "target_lengths": torch.cat([b["target_lengths"] for b in batches]),
    }


def packed_batch_padding(batch_optim: list[DatasetOutput], max_seq_len: int, pad_token_id: int, micro_bs: int) -> list[BatchOutput]:
    """
    This function will pad the batch to the max_seq_len
    """
    assert len(batch_optim) % micro_bs == 0, "batch_optim must be divisible by micro_bs"

    sample_padded_batch = [collate_fn([sample_batch], max_seq_len, pad_token_id) for sample_batch in batch_optim]

    micro_batches = [merge_batches_padding(sample_padded_batch[i : i + micro_bs]) for i in range(0, len(sample_padded_batch), micro_bs)]

    return micro_batches


### balancing


def pack_datatset_outputs_balancing(
    batch_optim: list[DatasetOutput], max_seq_len: int, micro_bs: int
) -> list[tuple[list[DatasetOutput], int]]:
    """
    This function will pack by batch of balanced seq length and will pad up to the max seq len per batch.
    Will create differentiely shaped batch per microbatch (and will break any compile step) but will reduce batch size
    """

    max_token_per_micro_batch = max_seq_len * micro_bs

    batch_with_len = [(len(sample["input_ids"]), sample) for sample in batch_optim]
    sorted_batch = sorted(batch_with_len, key=lambda x: x[0])

    batches_and_max_seq_len: list[tuple[list[DatasetOutput], int]] = []

    micro_batch = []
    max_seq_len_current_batch = 0

    for seq_len, sample in sorted_batch:
        # first we check if we can add this sample to the current batch
        # to do this we we need to see if the total token with this sample would exceed the max value

        maybe_max_seq_len = max(max_seq_len_current_batch, seq_len)

        if maybe_max_seq_len * (len(micro_batch) + 1) > max_token_per_micro_batch:
            # in tis case adding the sample would exceed the limit
            # so we rather cut out the current batch and start a new one
            batches_and_max_seq_len.append((micro_batch, maybe_max_seq_len))
            micro_batch = [sample]
            max_seq_len_current_batch = seq_len

        else:
            # if we still have room we can add this sample to the current batch
            max_seq_len_current_batch = maybe_max_seq_len
            micro_batch.append(sample)

    batches_and_max_seq_len.append((micro_batch, max_seq_len_current_batch))

    return batches_and_max_seq_len


def packed_batch_balancing(batch_optim: list[DatasetOutput], max_seq_len: int, pad_token_id: int, micro_bs: int) -> list[BatchOutput]:
    """
    this function will take a list of sample and try to balance by seq len the microbatches to avoid too much padding
    """

    batches_and_max_seq_len = pack_datatset_outputs_balancing(batch_optim, max_seq_len, micro_bs)

    micro_batches = []

    for batch, max_seq_len_batch in batches_and_max_seq_len:
        padded_micro_batch = []
        for sample in batch:
            collate_sample = collate_fn([sample], max_seq_len_batch, pad_token_id)
            padded_micro_batch.append(collate_sample)

        micro_batch = merge_batches_padding(padded_micro_batch)
        micro_batches.append(micro_batch)

    return data_parallel_rebalancing(micro_batches)


###########


CollateMode: TypeAlias = Literal["packing", "padding", "balancing"]


def packed_batch(
    batch_optim: list[DatasetOutput], max_seq_len: int, pad_token_id: int, micro_bs: int, collate_mode: CollateMode
) -> list[BatchOutput]:
    """
    Take a list of sample and return a list of microbatches
    """

    match collate_mode:
        case "packing":
            return packed_batch_packing(batch_optim, max_seq_len, pad_token_id, micro_bs)
        case "padding":
            return packed_batch_padding(batch_optim, max_seq_len, pad_token_id, micro_bs)
        case "balancing":
            return packed_batch_balancing(batch_optim, max_seq_len, pad_token_id, micro_bs)
        case _:
            raise ValueError(f"Invalid collate mode: {collate_mode}")

import random

import pytest
import torch
from torch.utils.data import DataLoader

from zeroband.training.data import (
    FakeTokenizedDataset,
    ParquetDataset,
    _should_skip_index,
    collate_fn,
    pack_datatset_outputs_balancing,
    pack_datatset_outputs_efficiently,
    packed_batch,
)


def test_pq_dataset(fake_rollout_files_dir):
    path = fake_rollout_files_dir(steps=[0, 1, 2, 3], num_files=4, batch_size=8)

    dataset = ParquetDataset(path, 8 * 4, timeout=2, step_count_init=0, ignore_zero_advantages=False)

    dataloader = DataLoader(dataset, batch_size=10, num_workers=2, collate_fn=lambda x: x)

    with pytest.raises(TimeoutError, match="Timeout waiting for step 4 to be created"):
        for _ in dataloader:
            ...


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("workers_id", [0, 1, 2, 3])
def test_should_skip_index(rank, workers_id):
    world_size = 4
    num_workers = 4

    full_index = list(range(100))

    expected_results = full_index[rank::world_size][workers_id::num_workers]

    results = []
    for index in full_index:
        # If we should not skip this index, add it to results
        if not _should_skip_index(index, world_size, rank, num_workers, workers_id):
            results.append(index)

    assert results == expected_results


def test_pack_datatset_outputs_efficiently():
    BS = 16

    batch = []

    dataset = FakeTokenizedDataset(64, 128)

    for i in range(BS):
        batch.append(next(iter(dataset)))

    packed_batch = pack_datatset_outputs_efficiently(batch, 64)

    assert len(packed_batch) >= 1


def test_pack_dataset_2():
    BS = 16
    SEQ_LEN = 2048

    batch = []

    for i in range(BS):
        seq_len = SEQ_LEN - 1
        input_ids = torch.randint(3, 128, (seq_len,))
        advantages = torch.randn(seq_len)
        batch.append(
            {
                "input_ids": input_ids,
                "advantages": advantages,
                "rewards": 0.5,
                "loss_mask": torch.ones(seq_len).int(),
                "logprobs": torch.randn(seq_len),
            }
        )
    packed_batch = pack_datatset_outputs_efficiently(batch, max_seq_len=seq_len)

    assert len(packed_batch) == BS


def test_pack_bin_packing():
    bin_size = 3
    SEQ_LEN = 64

    bin = []

    dataset = FakeTokenizedDataset(seq_len=SEQ_LEN, vocab_size=128)

    for i in range(bin_size):
        bin.append(next(iter(dataset)))

    micro_batch = collate_fn(bin, 2048, 128)

    assert micro_batch["input_ids"].shape == (1, 2048)


def test_packing_vs_padding():
    """
    Here we test that we don't lose any rewards or data when doing the different packing modes
    """

    BS = 32
    MICRO_BS = 4
    SEQ_LEN = 64

    batch_rollout = []

    seq_lens = [random.randint(1, SEQ_LEN) for _ in range(BS)]
    for i in range(BS):
        seq_len = seq_lens[i]
        data = {
            "input_ids": torch.ones(seq_len).int(),
            "advantages": torch.ones(seq_len),
            "loss_mask": torch.ones(seq_len).int(),
            "logprobs": torch.ones(seq_len),
            "seq_lens": torch.ones(seq_len),
            "rewards": torch.ones(1),
            "task_rewards": torch.ones(1),
            "length_penalties": torch.ones(1),
            "target_lengths": torch.ones(1),
            "task_type": "test_task",
        }

        batch_rollout.append(data)

    batch_packed = packed_batch(batch_rollout, max_seq_len=SEQ_LEN, collate_mode="packing", micro_bs=MICRO_BS, pad_token_id=0)
    batch_padded = packed_batch(batch_rollout, max_seq_len=SEQ_LEN, collate_mode="padding", micro_bs=MICRO_BS, pad_token_id=0)
    batch_balancing = packed_batch(batch_rollout, max_seq_len=SEQ_LEN, collate_mode="balancing", micro_bs=MICRO_BS, pad_token_id=0)

    total_rewards_packed = sum(batch["rewards"].sum().item() for batch in batch_packed)
    total_rewards_padded = sum(batch["rewards"].sum().item() for batch in batch_padded)
    total_rewards_balancing = sum(batch["rewards"].sum().item() for batch in batch_balancing)

    assert total_rewards_padded == total_rewards_balancing
    assert total_rewards_packed == total_rewards_balancing

    total_input_ids_packed = sum(batch["input_ids"].sum().item() for batch in batch_packed)
    total_input_ids_padded = sum(batch["input_ids"].sum().item() for batch in batch_padded)
    total_input_ids_balancing = sum(batch["input_ids"].sum().item() for batch in batch_balancing)

    assert total_input_ids_packed == total_input_ids_padded
    assert total_input_ids_balancing == total_input_ids_padded

    total_padded_tokens_packed = (
        sum(batch["input_ids"].shape[0] * batch["input_ids"].shape[1] for batch in batch_packed) - total_input_ids_packed
    )
    total_padded_tokens_padded = (
        sum(batch["input_ids"].shape[0] * batch["input_ids"].shape[1] for batch in batch_padded) - total_input_ids_padded
    )
    total_padded_tokens_balancing = (
        sum(batch["input_ids"].shape[0] * batch["input_ids"].shape[1] for batch in batch_balancing) - total_input_ids_balancing
    )

    assert total_padded_tokens_packed < total_padded_tokens_padded
    assert total_padded_tokens_balancing <= total_padded_tokens_padded


@pytest.mark.parametrize(
    "seq_lens,packed_output",
    [
        [[3, 3, 3, 3, 4, 4, 4, 4, 7, 7, 8, 8, 9, 9], [[3, 3, 3, 3], [4, 4, 4, 4], [7, 7], [8, 8], [9], [9]]],
        [[2, 2, 2, 2, 4, 4, 32], [[2, 2, 2, 2], [4, 4], [32]]],
        [[1, 1, 1, 1, 2, 2, 2, 2, 4, 32], [[1, 1, 1, 1, 2, 2, 2, 2], [4], [32]]],
    ],
)
def test_pack_datatset_outputs_balancing(seq_lens, packed_output):
    max_seq_len = 8
    micro_bs = 2

    # batch_size = 32

    samples = [{"input_ids": torch.arange(seq)} for seq in seq_lens]

    micro_batches = pack_datatset_outputs_balancing(samples, max_seq_len=max_seq_len, micro_bs=micro_bs)

    micro_batches_len = [[len(sample["input_ids"]) for sample in batch[0]] for batch in micro_batches]

    print(micro_batches_len)
    assert micro_batches_len == packed_output

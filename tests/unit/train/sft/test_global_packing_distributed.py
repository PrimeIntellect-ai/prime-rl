import copy
import json
import os
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from datasets import Dataset
from renderers.base import RenderedTokens

from prime_rl.configs.sft import PackingConfig, SFTDataConfig
from prime_rl.trainer.sft.data.broker import PackedDataLoader
from prime_rl.trainer.sft.data.dataset import SFTDataset
from prime_rl.trainer.world import reset_world


class TextRenderer:
    def __call__(self, example):
        return self

    def render(self, messages, **kwargs):
        content = [ord(char) for char in messages[-1]["content"]]
        return RenderedTokens(
            token_ids=[0, *content, 1],
            message_indices=[-1, *([len(messages) - 1] * (len(content) + 1))],
            sampled_mask=[False, *([True] * (len(content) + 1))],
        )

    def get_stop_token_ids(self):
        return [1]


class ObservedLoader(PackedDataLoader):
    def __init__(self, *args, **kwargs):
        self.communication_windows = []
        super().__init__(*args, **kwargs)

    def _exchange(self, rows):
        time.sleep(0.001 * (1 + self.rank))
        start = time.perf_counter()
        result = super()._exchange(rows)
        self.communication_windows.append((start, time.perf_counter()))
        return result


def distributed_loader_worker(rank, world_size, cp_size, backend, directory):
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world_size)
    )
    reset_world()
    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        init_method=f"file://{directory}/init",
        timeout=timedelta(seconds=120),
        device_id=torch.device("cuda", rank) if backend == "nccl" else None,
    )
    verification = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    raw = Dataset.from_list(
        [
            {
                "messages": [
                    {
                        "role": "user" if index % 11 == 0 else "assistant",
                        "content": chr(65 + index % 26) * (1 + index * 13 % 43),
                    }
                ]
            }
            for index in range(37)
        ]
    )
    config = SFTDataConfig(
        seq_len=16, batch_size=8, micro_batch_size=2, num_workers=1, packing=PackingConfig(chunk_size=2)
    )
    microsteps = config.batch_size // (world_size // cp_size * config.micro_batch_size)
    epochs = None if backend == "nccl" else 3

    def make_loader(max_epochs=epochs, settings=config, validation=False):
        dataset = SFTDataset(raw, TextRenderer(), seq_len=16, shuffle=True, seed=7, max_epochs=max_epochs)
        return ObservedLoader(dataset, settings, cp_size=cp_size, timeout_seconds=120, validation=validation)

    reference = SFTDataset(raw, TextRenderer(), seq_len=16, shuffle=True, seed=7)
    shuffled = {}

    def expected_sample(position):
        epoch, index = divmod(position, len(raw))
        if epoch not in shuffled:
            shuffled[epoch] = raw.shuffle(seed=7 + epoch, keep_in_memory=True)
        sample = reference._process(shuffled[epoch][index])
        if sample is None:
            return None
        return {key: sample[key][:32] for key in ("input_ids", "target_ids", "position_ids", "loss_mask")}

    def verify_step(rows, loader, previous):
        received = [None] * world_size
        dist.all_gather_object(received, (rows, loader.dataset_progress), group=verification)
        assert all(progress == received[0][1] for _, progress in received)
        for base in range(0, world_size, cp_size):
            for peer in range(1, cp_size):
                for left, right in zip(received[base][0], received[base + peer][0], strict=True):
                    assert left["sample_ids"] == right["sample_ids"]
                    for key in ("input_ids", "target_ids", "position_ids", "loss_mask", "seq_lens"):
                        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        unique = [rows for rows, _ in received[::cp_size]]
        positions, costs = [], []
        for round_rows in zip(*unique, strict=True):
            for row in round_rows:
                offset, lengths = 0, []
                for position in row["sample_ids"]:
                    expected = expected_sample(position)
                    length = len(expected["input_ids"])
                    for key, values in expected.items():
                        assert row[key][0, offset : offset + length].tolist() == values
                    positions.append(position)
                    lengths.append(length)
                    offset += length
                costs.append(sum(length**2 for length in lengths))
                assert row["num_tokens"] == offset
                assert not row["loss_mask"][0, offset:].any()
                assert not row["input_ids"][0, offset:].any()
                assert row["position_ids"][0, offset:].tolist() == list(range(32 - offset))
                if lengths:
                    lengths[-1] += 32 - offset
                assert row["seq_lens"].tolist() == (lengths or [32])
        end = loader.dataset_progress["step"]
        assert sorted(positions) == [
            position for position in range(previous, end) if expected_sample(position) is not None
        ]
        assert costs == sorted(costs)
        return end

    model = optimizer = None
    if backend == "nccl":
        from torch.distributed.fsdp import fully_shard

        torch.manual_seed(123)
        model = torch.nn.Sequential(
            torch.nn.Embedding(256, 512), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 128)
        ).cuda()
        for layer in model:
            if next(layer.parameters(), None) is not None:
                fully_shard(layer)
        fully_shard(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    loader = make_loader()
    position = 0
    model_windows, communication_windows = [], []
    for step in range(64 if model is not None else 3):
        time.sleep(0.002 * (rank % 2))
        rows = [next(loader) for _ in range(microsteps)]
        if model is not None:
            start = time.perf_counter()
            for row in rows:
                loss = model(row["input_ids"].cuda()).square().mean()
                assert torch.isfinite(loss)
                (loss / microsteps).backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            model_windows.append((start, time.perf_counter()))
        position = verify_step(rows, loader, position)
        if step == 2:
            state = copy.deepcopy(loader.state_dict())
            loader.future.result()
            assert loader.state_dict() == state
            loader.close()
            communication_windows.extend(loader.communication_windows)
            changed = config.model_copy(update={"num_workers": 2, "packing": PackingConfig(chunk_size=3)})
            loader = make_loader(settings=changed)
            loader.load_state_dict(state)
        if (model is None and step == 1) or (model is not None and step in [16, 32, 48]):
            batch_size = 2 if step in [1, 16] else 6
            validation_config = config.model_copy(update={"batch_size": batch_size})
            with pytest.raises(ValueError, match="divisible"):
                make_loader(settings=validation_config)
            validation = make_loader(max_epochs=1, settings=validation_config, validation=True)
            validation_microsteps = validation.num_rows // validation.dp_size
            valid_position = 0
            while (first := next(validation, None)) is not None:
                rows = [first, *(next(validation) for _ in range(validation_microsteps - 1))]
                valid_position = verify_step(rows, validation, valid_position)
            assert all(
                expected_sample(position) is None
                for position in range(valid_position, validation.dataset_progress["step"])
            )
            assert validation.dataset_progress["step"] == len(raw)
            validation.close()
    if model is None:
        while (first := next(loader, None)) is not None:
            rows = [first, *(next(loader) for _ in range(microsteps - 1))]
            position = verify_step(rows, loader, position)
        assert all(expected_sample(index) is None for index in range(position, loader.dataset_progress["step"]))
        position = loader.dataset_progress["step"]
        assert position == 3 * len(raw)
    time.sleep(0.003 * rank)
    loader.close()
    communication_windows.extend(loader.communication_windows)
    overlaps = sum(
        any(start < model_end and end > model_start for model_start, model_end in model_windows)
        for start, end in communication_windows
    )
    if model is not None:
        assert overlaps > 0
    with open(f"{directory}/rank-{rank}.json", "w") as output:
        json.dump(
            {
                "position": position,
                "exchanges": len(communication_windows),
                "overlapped_exchanges": overlaps,
                "model_steps": len(model_windows),
            },
            output,
        )
    dist.destroy_process_group(verification)
    dist.destroy_process_group()


def test_distributed_payloads_cp_and_cursor_resume(tmp_path):
    multiprocessing.spawn(distributed_loader_worker, args=(4, 2, "gloo", str(tmp_path)), nprocs=4)


@pytest.mark.gpu
def test_prefetch_overlaps_fsdp_and_optimizer_collectives(tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("Requires four GPUs")
    multiprocessing.spawn(distributed_loader_worker, args=(4, 2, "nccl", str(tmp_path)), nprocs=4)

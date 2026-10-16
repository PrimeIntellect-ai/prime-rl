import random

import pytest
import torch
from datasets import Dataset
from torch import nn

from prime_rl.configs.sft import SFTDataConfig
from prime_rl.trainer.sft.data import SFTDataset
from prime_rl.trainer.sft.master_cast_probe import load_probe_batches, probe_data_config
from prime_rl.trainer.sft.weight_dynamics import UPDATE_LAGS, WeightUpdateTracker
from prime_rl.trainer.world import reset_world

NUM_LAYERS = 4
PROBE_SEQ_LEN = 64


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(16, 24, bias=False)
        self.o_proj = nn.Linear(8, 16, bias=False)


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2, 16, 16))
        self.down_proj = nn.Parameter(torch.randn(2, 16, 8))


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = Experts()


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.mlp = Mlp()


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Layer() for _ in range(NUM_LAYERS))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()


def test_update_cosines_match_reference_at_every_lag_through_ring_wraparound():
    torch.manual_seed(0)
    model = Model()
    tracker = WeightUpdateTracker(model, NUM_LAYERS, num_elements=64)
    drift = [torch.randn_like(param) for param in tracker.params]
    updates = [[] for _ in tracker.params]
    num_steps = UPDATE_LAGS[-1] + 44
    for t in range(num_steps):
        with torch.no_grad():
            for i, param in enumerate(tracker.params):
                update = 0.3 * drift[i] + torch.randn_like(param)
                param.add_(update)
                updates[i].append(update.reshape(-1)[tracker.indices[i]])
        metrics = tracker.step()
        for i, tag in enumerate(tracker.tags):
            for lag in UPDATE_LAGS:
                key = f"weight_dynamics/{tag}/update_cos_lag{lag}"
                if t < lag:
                    assert key not in metrics
                else:
                    expected = torch.nn.functional.cosine_similarity(updates[i][t], updates[i][t - lag], dim=0)
                    assert metrics[key] == pytest.approx(expected.item(), abs=1e-5)
    assert f"weight_dynamics/{tracker.tags[0]}/update_cos_lag{UPDATE_LAGS[-1]}" in metrics


def _probe_rows(num_rows: int) -> Dataset:
    rng = random.Random(0)
    return Dataset.from_list(
        [{"messages": [{"role": "assistant", "content": "x" * rng.randint(3, 40)}]} for _ in range(num_rows)]
    )


def _rank_batches(monkeypatch, dummy_renderer, world_size: int, cp: int, num_packs: int) -> list[list[dict]]:
    batches = []
    for rank in range(world_size):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(world_size))
        reset_world()
        config = probe_data_config(SFTDataConfig(seq_len=32768, seed=rank, shuffle=False), "probe", PROBE_SEQ_LEN)
        dataset = SFTDataset(
            _probe_rows(200),
            lambda _: dummy_renderer,
            shuffle=config.shuffle,
            seed=config.seed,
            seq_len=config.seq_len,
            non_dp_size=world_size,
            max_epochs=1,
        )
        batches.append(load_probe_batches(dataset, config.seq_len, cp, num_packs, rank // cp, world_size // cp))
    reset_world()
    return batches


def _global_packs(batches: list[list[dict]], world_size: int, cp: int) -> list[dict]:
    dp_size = world_size // cp
    packs = {}
    for dp_rank in range(dp_size):
        for k, batch in enumerate(batches[dp_rank * cp]):
            packs[dp_rank + k * dp_size] = batch
    return [packs[index] for index in sorted(packs)]


@pytest.mark.parametrize("world_size, cp", [(8, 1), (8, 2), (16, 1), (16, 2)])
def test_probe_packs_do_not_depend_on_world_size_or_cp(monkeypatch, dummy_renderer, world_size, cp):
    num_packs = 16
    reference = _global_packs(_rank_batches(monkeypatch, dummy_renderer, 4, 1, num_packs), 4, 1)
    batches = _rank_batches(monkeypatch, dummy_renderer, world_size, cp, num_packs)
    for rank in range(world_size):
        assert len(batches[rank]) == num_packs // (world_size // cp)
        assert all(batch["input_ids"].shape[1] % cp == 0 for batch in batches[rank])
        for batch, cp_leader_batch in zip(batches[rank], batches[rank - rank % cp]):
            assert torch.equal(batch["input_ids"], cp_leader_batch["input_ids"])
    packs = _global_packs(batches, world_size, cp)
    assert len(packs) == len(reference) == num_packs
    for actual, expected in zip(packs, reference):
        width = expected["input_ids"].shape[1]
        assert torch.equal(actual["input_ids"][:, :width], expected["input_ids"])
        for key in ("loss_mask", "target_ids", "position_ids"):
            assert torch.equal(actual[key][:, :width], expected[key]), key
        assert not actual["loss_mask"][:, width:].any()


def test_probe_data_config_does_not_depend_on_the_training_data_config():
    configs = [
        probe_data_config(SFTDataConfig(seq_len=32768, seed=3, shuffle=False, batch_size=4), "rows", PROBE_SEQ_LEN),
        probe_data_config(
            SFTDataConfig(seq_len=8192, seed=5, batch_size=16, micro_batch_size=2), "rows", PROBE_SEQ_LEN
        ),
    ]
    assert configs[0].model_dump(exclude={"batch_size"}) == configs[1].model_dump(exclude={"batch_size"})
    assert (configs[0].name, configs[0].seq_len, configs[0].seed, configs[0].shuffle) == (
        "rows",
        PROBE_SEQ_LEN,
        0,
        True,
    )
    assert configs[0].micro_batch_size == 1


def test_probe_packs_must_split_evenly_over_data_ranks(monkeypatch, dummy_renderer):
    with pytest.raises(AssertionError, match="do not split evenly"):
        _rank_batches(monkeypatch, dummy_renderer, 8, 1, 12)

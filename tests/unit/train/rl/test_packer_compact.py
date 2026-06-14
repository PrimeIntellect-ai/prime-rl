from collections import deque
from types import SimpleNamespace

from prime_rl.trainer.rl.packer import MultiPacker
from prime_rl.transport.compact import compact_training_sample
from prime_rl.transport.types import TrainingSample


def _sample() -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
        teacher_logprobs=[0.0, 0.0, -0.1, -0.2],
        env_name="test-env",
    )


def test_validate_sample_accepts_compacted_training_sample():
    packer = MultiPacker.__new__(MultiPacker)
    packer.seq_len = 8
    sample = _sample()
    compact_training_sample(sample)

    valid, reason = packer._validate_sample(sample)

    assert valid is True
    assert reason is None


def test_count_and_select_samples_use_compacted_lengths():
    packer = MultiPacker.__new__(MultiPacker)
    packer.dp_world_size = 1
    packer.seq_len = 8
    packer._round_robin_position = 0
    packer.multi_run_manager = SimpleNamespace(
        used_idxs=[0],
        progress=[SimpleNamespace(step=0)],
    )
    sample = _sample()
    compact_training_sample(sample)
    packer.buffers = [deque([(sample, 0)])]

    assert packer._count_tokens() == 4

    selected = packer._select_samples_round_robin(token_budget=8)

    assert selected == [(0, sample, 0)]
    assert not packer.buffers[0]

import numpy as np
import pytest
import torch

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.trainer.rl.data import ReplayDataLoader
from prime_rl.trainer.world import reset_world
from prime_rl.transports.batch.replay import ReplayMeta, write_replay_meta, write_replay_step
from prime_rl.transports.batch.types import TrainingSample

SEQ_LEN = 32
NUM_RANKS = 2


def sample(length: int, first_id: int) -> TrainingSample:
    mask = np.arange(length) >= length // 2
    return TrainingSample(
        token_ids=np.arange(first_id, first_id + length, dtype=np.int32),
        mask=mask,
        logprobs=np.where(mask, -0.25, 0.0).astype(np.float32),
        temperatures=np.full(length, 0.5, dtype=np.float32),
        env_name="env",
        advantages=np.where(mask, 1.5, 0.0).astype(np.float32),
        trace_id=f"trace-{first_id}",
        branch_index=0,
    )


@pytest.fixture
def replay_dir(tmp_path):
    steps = [
        prepare_batch([sample(n, 100 * i) for i, n in enumerate(lengths)], SEQ_LEN, NUM_RANKS, sum, 8)
        for lengths in ([20, 12, 9, 30], [17, 5, 26])
    ]
    for step, grid in enumerate(steps):
        write_replay_step(tmp_path, step, grid)
    write_replay_meta(
        tmp_path, ReplayMeta(num_steps=len(steps), seq_len=SEQ_LEN, num_train_workers=NUM_RANKS, pad_to_multiple_of=8)
    )
    return tmp_path, steps


@pytest.fixture
def world(monkeypatch):
    """Rank 3 of 4: data rank 1 of 2 with two ranks (e.g. CP) per data rank."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    reset_world()
    yield
    reset_world()


@pytest.mark.parametrize("start_step,replayed_steps", [(1, [0, 1, 0]), (4, [1, 0, 1])])
def test_replay_loader_returns_its_data_rank_of_each_step_and_cycles(replay_dir, world, start_step, replayed_steps):
    directory, steps = replay_dir
    loader = ReplayDataLoader(directory, start_step, SEQ_LEN, NUM_RANKS, 4)
    assert loader.dp_rank == 1

    for step in replayed_steps:
        loader.wait_for_batch()
        batch = loader.get_batch()
        expected = steps[step][1]
        assert len(batch) == len(expected)
        for tensors, micro_batch in zip(batch, expected):
            num_tokens = len(micro_batch.input_ids)
            assert tensors["input_ids"].shape == (1, num_tokens) and tensors["input_ids"].dtype == torch.int64
            torch.testing.assert_close(tensors["input_ids"][0], torch.from_numpy(micro_batch.input_ids).long())
            torch.testing.assert_close(tensors["position_ids"][0], torch.from_numpy(micro_batch.position_ids).long())
            torch.testing.assert_close(tensors["loss_mask"][0], torch.from_numpy(micro_batch.loss_mask))
            torch.testing.assert_close(tensors["advantages"][0], torch.from_numpy(micro_batch.advantages))
            torch.testing.assert_close(
                tensors["inference_logprobs"][0], torch.from_numpy(micro_batch.inference_logprobs)
            )
            torch.testing.assert_close(tensors["temperatures"][0], torch.from_numpy(micro_batch.temperatures))
            assert tensors["seq_lens"].tolist() == micro_batch.seq_lens
            assert tensors["sequence_lengths"] == micro_batch.sequence_lengths
            assert tensors["padding"] == micro_batch.padding
            assert tensors["trace_ids"] == micro_batch.trace_ids
            assert tensors["routed_experts"] is None and tensors["top_logprobs_ids"] is None


@pytest.mark.parametrize(
    "seq_len,dp_world_size,pad_to_multiple_of,message",
    [
        (SEQ_LEN, 1, 8, "packed for 2 data ranks, but the trainer has 1"),
        (64, NUM_RANKS, 8, "packed for seq_len 32"),
        (SEQ_LEN, NUM_RANKS, 16, "padded to a multiple of 8, but the trainer requires a multiple of 16"),
    ],
)
def test_replay_loader_rejects_a_mismatched_topology(
    replay_dir, world, seq_len, dp_world_size, pad_to_multiple_of, message
):
    with pytest.raises(ValueError, match=message):
        ReplayDataLoader(replay_dir[0], 1, seq_len, dp_world_size, pad_to_multiple_of)


def test_replay_config_excludes_fake_data_score_centering_and_router_replay(tmp_path):
    with pytest.raises(ValueError, match="at most one of data.fake and data.replay"):
        TrainerConfig(data={"fake": {}, "replay": tmp_path})
    with pytest.raises(ValueError, match="requires loss.score_centering = false"):
        TrainerConfig(data={"replay": tmp_path}, loss={"type": "icepop", "score_centering": True})
    with pytest.raises(ValueError, match="requires enable_router_replay = false"):
        TrainerConfig(data={"replay": tmp_path}, loss={"type": "icepop"}, enable_router_replay=True)
    assert TrainerConfig(data={"replay": tmp_path}, loss={"type": "icepop"}).data.replay == tmp_path

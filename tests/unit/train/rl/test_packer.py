from pathlib import Path
from typing import Generator

import pytest
import tomli_w
import torch
import torch.distributed as dist

import prime_rl.trainer.runs as runs
from prime_rl.configs.shared import FileSystemTransportConfig
from prime_rl.trainer.rl.packer import MultiPacker
from prime_rl.trainer.runs import setup_multi_run_manager
from prime_rl.trainer.world import reset_world
from prime_rl.transport.types import TrainingSample


@pytest.fixture(autouse=True, scope="module")
def init_process_group() -> Generator[None, None, None]:
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12356", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def create_run_with_config(output_dir: Path, run_name: str) -> Path:
    run_dir = output_dir / run_name
    run_dir.mkdir()
    control_dir = run_dir / "control"
    control_dir.mkdir()
    config = {
        "model": {"name": "test-model"},
        "batch_size": 2,
        "group_size": 1,
        "env": [{"id": "test-env"}],
        "sampling": {"temperature": 1.0},
        # test-model isn't in MODEL_RENDERER_MAP; bypass the renderer-resolution validator.
        "renderer": "None",
    }
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(config, f)
    return run_dir


def make_training_sample() -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1],
        prompt_mask=[False],
        completion_ids=[2],
        completion_mask=[True],
        completion_logprobs=[-0.1],
        completion_temperatures=[1.0],
        env_name="test-env",
    )


def test_packer_progress_updates_once_per_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reset_world()
    runs._MULTI_RUN_MANAGER = None
    manager = setup_multi_run_manager(output_dir=tmp_path, max_runs=1, device=torch.device("cpu"))

    create_run_with_config(tmp_path, "run_test123")
    manager.discover_runs()
    run_idx = manager.id_2_idx["run_test123"]

    class DummyReceiver:
        def receive(self):
            return []

        def reset_run(self, idx: int) -> None:
            pass

    class DummySender:
        def __init__(self):
            self.sent = []

        def send(self, micro_batch_grid):
            self.sent.append(micro_batch_grid)

    sender_holder: dict[str, DummySender] = {}

    def fake_receiver(_config):
        return DummyReceiver()

    def fake_sender(_output_dir, _data_world_size, _current_step, _config):
        sender = DummySender()
        sender_holder["sender"] = sender
        return sender

    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_training_batch_receiver", fake_receiver)
    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_micro_batch_sender", fake_sender)

    packer = MultiPacker(
        dp_world_size=1,
        seq_len=4,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        start_step=0,
    )

    packer.buffers[run_idx].append((make_training_sample(), 0))
    packer.buffers[run_idx].append((make_training_sample(), 0))

    packer.pack()

    progress = manager.progress[run_idx]
    assert progress.total_samples == 2
    assert progress.total_tokens == 4
    assert progress.step == 1

    sender = sender_holder["sender"]
    assert len(sender.sent) == 1
    assert len(sender.sent[0][0]) == 1


def _mm_sample(uri: str) -> TrainingSample:
    """Deferred-MM TrainingSample (carries mm_refs, no mm_kwargs)."""
    from prime_rl.transport.types import MMRefs

    return TrainingSample(
        prompt_ids=[1],
        prompt_mask=[False],
        completion_ids=[2],
        completion_mask=[True],
        completion_logprobs=[-0.1],
        completion_temperatures=[1.0],
        env_name="test-env",
        mm_token_type_ids=[1, 1],
        mm_refs=MMRefs(
            descriptor={"mm_items": {"image": [{"image_grid_thw": [[1, 2, 3]]}]}, "mm_hashes": {"image": [uri]}},
            uris=[uri],
        ),
    )


def _packer_with_two_runs(tmp_path, monkeypatch, dp_world_size, seq_len):
    """Set up a MultiPacker over two discovered runs; capture sent grids."""
    reset_world()
    runs._MULTI_RUN_MANAGER = None
    manager = setup_multi_run_manager(output_dir=tmp_path, max_runs=2, device=torch.device("cpu"))
    create_run_with_config(tmp_path, "run_a")
    create_run_with_config(tmp_path, "run_b")
    manager.discover_runs()

    sent: list = []

    class DummyReceiver:
        def receive(self):
            return []

        def reset_run(self, idx):
            pass

    class DummySender:
        def send(self, micro_batch_grid):
            sent.append(micro_batch_grid)

    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_training_batch_receiver", lambda _c: DummyReceiver())
    monkeypatch.setattr("prime_rl.trainer.rl.packer.setup_micro_batch_sender", lambda *a, **k: DummySender())
    packer = MultiPacker(
        dp_world_size=dp_world_size,
        seq_len=seq_len,
        pad_to_multiple_of=1,
        tokenizer=None,
        config=FileSystemTransportConfig(),
        start_step=0,
    )
    return manager, packer, sent


def test_multipacker_pack_preserves_mm_refs_modality_and_run_tagging(tmp_path, monkeypatch):
    """The REAL MultiPacker.pack() path (per-run buffers → round-robin → prepare_batch
    → merged rank grids) preserves deferred-MM correctness across runs."""
    from prime_rl.trainer.batch import _is_multimodal_sample

    manager, packer, sent = _packer_with_two_runs(tmp_path, monkeypatch, dp_world_size=2, seq_len=3)
    a, b = manager.id_2_idx["run_a"], manager.id_2_idx["run_b"]
    for idx, uri in ((a, "ha"), (b, "hb")):
        packer.buffers[idx].append((_mm_sample(uri), 0))
        packer.buffers[idx].append((make_training_sample(), 0))

    packer.pack()
    assert sent, "pack() sent nothing"
    grid = sent[-1]  # list[per-rank list[MicroBatch]]
    assert len(grid) == 2

    # FSDP safety: every rank has the same modality at each micro-step index.
    for step_mbs in zip(*grid):
        assert len({_is_multimodal_sample(mb) for mb in step_mbs}) == 1

    mm_mbs = [mb for rank in grid for mb in rank if _is_multimodal_sample(mb)]
    assert mm_mbs, "no MM microbatches produced"
    real_run_idxs = set()
    for mb in mm_mbs:
        # Every MM microbatch is a standalone deferred-refs sequence (never pixels,
        # never packed with text). Dummies deep-copy the source so they keep mm_refs
        # too — they're distinguished by an all-False loss_mask.
        assert mb.mm_kwargs is None and mb.mm_refs is not None
        if any(mb.loss_mask):  # real (loss-bearing) MM → tagged to exactly one run
            tagged = [i for i, n in enumerate(mb.lora_num_tokens) if n > 0]
            assert len(tagged) == 1 and mb.lora_num_tokens[tagged[0]] == len(mb.input_ids)
            real_run_idxs.add(tagged[0])
    assert real_run_idxs == {a, b}, f"both runs' MM should be tagged; got {real_run_idxs}"


def test_multipacker_pack_mm_padding_is_zero_loss(tmp_path, monkeypatch):
    """A lone MM sample forces a dummy MM microbatch for rank padding; it must be
    zero-loss (and keep MM modality so all ranks still run the vision encoder)."""
    from prime_rl.trainer.batch import _is_multimodal_sample

    manager, packer, sent = _packer_with_two_runs(tmp_path, monkeypatch, dp_world_size=2, seq_len=2)
    a, b = manager.id_2_idx["run_a"], manager.id_2_idx["run_b"]
    packer.buffers[a].append((_mm_sample("ha"), 0))  # one MM → needs a dummy to fill 2 ranks
    packer.buffers[b].append((make_training_sample(), 0))

    packer.pack()
    assert sent
    grid = sent[-1]
    mm_mbs = [mb for rank in grid for mb in rank if _is_multimodal_sample(mb)]
    # A dummy keeps MM modality (so all ranks run the vision encoder) but is
    # zero-loss; it's identified by an all-False loss_mask, not by missing mm_refs
    # (the dummy deep-copies the source, mm_refs included).
    dummies = [mb for mb in mm_mbs if not any(mb.loss_mask)]
    assert dummies, "expected a zero-loss dummy MM padding microbatch"
    for d in dummies:
        assert all(a == 0.0 for a in d.advantages)

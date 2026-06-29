from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import tomli_w
import torch
import torch.distributed as dist

import prime_rl.trainer.runs as runs
from prime_rl.configs.shared import FileSystemTransportConfig
from prime_rl.trainer.rl.packer import MultiPacker
from prime_rl.trainer.runs import setup_multi_run_manager
from prime_rl.trainer.utils import build_bin_cost
from prime_rl.trainer.world import reset_world
from prime_rl.transport.types import EncodedTensor, TrainingSample


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
        # test-model isn't in MODEL_RENDERER_MAP; use the explicit default renderer.
        "renderer": {"name": "default"},
    }
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(config, f)
    return run_dir


def make_training_sample() -> TrainingSample:
    return TrainingSample(
        token_ids=[1, 2],
        mask=[False, True],
        logprobs=[0.0, -0.1],
        temperatures=[1.0, 1.0],
        advantages=[0.0, 1.0],
        env_name="test-env",
    )


def _encoded_tensor(data, dtype) -> EncodedTensor:
    arr = np.asarray(data, dtype=dtype)
    return EncodedTensor(dtype=str(arr.dtype), shape=list(arr.shape), data=arr.tobytes())


def _decode_encoded_tensor(encoded: EncodedTensor):
    return np.frombuffer(encoded.data, dtype=np.dtype(encoded.dtype)).reshape(encoded.shape).tolist()


def _mm_sample(value: float, env_name: str = "test-env") -> TrainingSample:
    return TrainingSample(
        token_ids=[1, 250, 2],
        mask=[False, True, True],
        logprobs=[0.0, -0.1, -0.2],
        temperatures=[1.0, 1.0, 1.0],
        env_name=env_name,
        advantages=[0.0, 1.0, 1.0],
        mm_token_type_ids=[0, 1, 0],
        mm_kwargs={
            "pixel_values": _encoded_tensor([[value, value + 1]], np.float32),
            "image_grid_thw": _encoded_tensor([[1, 2, 2]], np.int64),
        },
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
        bin_cost=build_bin_cost(None),
        start_step=0,
    )
    return manager, packer, sent


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
        bin_cost=build_bin_cost(None),
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
    micro_batch = sender.sent[0][0][0]
    assert micro_batch.run_id == "run_test123"
    assert micro_batch.run_step == 0


def test_multipacker_pack_preserves_mm_kwargs_and_run_tagging(tmp_path, monkeypatch):
    """MultiPacker packs each run separately and preserves multimodal sidecars."""
    from prime_rl.trainer.batch import _is_multimodal_sample

    manager, packer, sent = _packer_with_two_runs(tmp_path, monkeypatch, dp_world_size=2, seq_len=5)
    a, b = manager.id_2_idx["run_a"], manager.id_2_idx["run_b"]
    for idx, value in ((a, 1.0), (b, 10.0)):
        packer.buffers[idx].append((_mm_sample(value), 0))
        packer.buffers[idx].append((make_training_sample(), 0))

    packer.pack()
    assert sent, "pack() sent nothing"
    grid = sent[-1]
    assert len(grid) == 2

    mm_mbs = [mb for rank in grid for mb in rank if _is_multimodal_sample(mb)]
    assert mm_mbs, "no MM microbatches produced"
    real_run_idxs = set()
    for mb in mm_mbs:
        assert mb.mm_kwargs is not None
        if any(mb.loss_mask):
            tagged = [i for i, n in enumerate(mb.lora_num_tokens) if n > 0]
            assert len(tagged) == 1 and mb.lora_num_tokens[tagged[0]] == len(mb.input_ids)
            real_run_idxs.add(tagged[0])
    assert real_run_idxs == {a, b}, f"both runs' MM should be tagged; got {real_run_idxs}"


def test_multipacker_pack_mm_padding_is_zero_loss(tmp_path, monkeypatch):
    """A lone MM sample forces a dummy MM microbatch for rank padding; it must be zero-loss."""
    from prime_rl.trainer.batch import _is_multimodal_sample

    manager, packer, sent = _packer_with_two_runs(tmp_path, monkeypatch, dp_world_size=2, seq_len=3)
    a, b = manager.id_2_idx["run_a"], manager.id_2_idx["run_b"]
    packer.buffers[a].append((_mm_sample(1.0), 0))
    packer.buffers[b].append((make_training_sample(), 0))

    packer.pack()
    assert sent
    grid = sent[-1]
    mm_mbs = [mb for rank in grid for mb in rank if _is_multimodal_sample(mb)]
    dummies = [mb for mb in mm_mbs if not any(mb.loss_mask)]
    assert dummies, "expected a zero-loss dummy MM padding microbatch"
    for dummy in dummies:
        assert all(advantage == 0.0 for advantage in dummy.advantages)


def test_multipacker_pack_packs_mm_kwargs_within_each_run(tmp_path, monkeypatch):
    """Eager multimodal samples pack within a run but never across runs."""
    from prime_rl.trainer.batch import _is_multimodal_sample

    manager, packer, sent = _packer_with_two_runs(tmp_path, monkeypatch, dp_world_size=1, seq_len=12)
    a, b = manager.id_2_idx["run_a"], manager.id_2_idx["run_b"]
    for idx, base in ((a, 1.0), (b, 10.0)):
        packer.buffers[idx].append((_mm_sample(base), 0))
        packer.buffers[idx].append((_mm_sample(base + 2), 0))

    packer.pack()
    assert sent
    grid = sent[-1]
    real_mm_mbs = [mb for rank in grid for mb in rank if _is_multimodal_sample(mb) and any(mb.loss_mask)]

    assert len(real_mm_mbs) == 2
    for mb in real_mm_mbs:
        assert len(mb.input_ids) == 6
        assert mb.position_ids == [0, 1, 2, 0, 1, 2]
        assert mb.seq_lens == [3, 3]
        assert mb.mm_kwargs is not None
        assert mb.mm_kwargs["pixel_values"].shape == [2, 2]
        assert mb.mm_kwargs["image_grid_thw"].shape == [2, 3]
        assert len(_decode_encoded_tensor(mb.mm_kwargs["pixel_values"])) == 2
        tagged = [i for i, n in enumerate(mb.lora_num_tokens) if n > 0]
        assert len(tagged) == 1

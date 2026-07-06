"""TTT RL replay: adapter-path resolution from traces, packer bin separation by adapter,
the frozen-adapter forward hooks (exact math vs a PEFT reference), QA recycling into
ce-routed samples, and checkpoint GC lifecycles."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from prime_rl.trainer.batch import _MicroBatchBin, prepare_sample  # noqa: E402
from prime_rl.transport.types import TrainingSample  # noqa: E402


def make_sample(n: int = 4, ttt_adapter_path: str | None = None, **kwargs) -> TrainingSample:
    return TrainingSample(
        token_ids=list(range(n)),
        mask=[False] + [True] * (n - 1),
        logprobs=[0.0] * n,
        temperatures=[1.0] * n,
        env_name="e",
        advantages=[0.0] + [1.0] * (n - 1),
        ttt_adapter_path=ttt_adapter_path,
        **kwargs,
    )


# -- packer: one adapter per bin ------------------------------------------------------------


def test_bins_never_mix_adapters():
    base = prepare_sample(make_sample(), seq_len=64)
    a1 = prepare_sample(make_sample(ttt_adapter_path="/ckpt/r1/v1"), seq_len=64)
    a1_again = prepare_sample(make_sample(ttt_adapter_path="/ckpt/r1/v1"), seq_len=64)
    a2 = prepare_sample(make_sample(ttt_adapter_path="/ckpt/r1/v2"), seq_len=64)

    bin_content = _MicroBatchBin.from_sample(0, a1)
    assert bin_content.can_add(a1_again, max_seq_len=64)  # same adapter packs
    assert not bin_content.can_add(a2, max_seq_len=64)  # different version doesn't
    assert not bin_content.can_add(base, max_seq_len=64)  # base model doesn't

    base_bin = _MicroBatchBin.from_sample(0, base)
    assert not base_bin.can_add(a1, max_seq_len=64)


def test_micro_batch_carries_bin_adapter():
    from prime_rl.trainer.batch import _materialize_bin

    a1 = prepare_sample(make_sample(ttt_adapter_path="/ckpt/r1/v1"), seq_len=64)
    micro = _materialize_bin(_MicroBatchBin.from_sample(0, a1), num_loras=1)
    assert micro.ttt_adapter_path == "/ckpt/r1/v1"

    base = prepare_sample(make_sample(), seq_len=64)
    micro_base = _materialize_bin(_MicroBatchBin.from_sample(0, base), num_loras=1)
    assert micro_base.ttt_adapter_path is None


# -- replay manager: hook math equals the reference LoRA computation -------------------------


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.up_proj = torch.nn.Linear(8, 16, bias=False)

    def forward(self, x):
        return self.up_proj(self.q_proj(x))


def write_adapter(path: Path, modules: dict[str, tuple[int, int]], rank: int = 2, alpha: float = 4.0) -> None:
    """A PEFT-format checkpoint with random A/B for the given {module_path: (in, out)}."""
    import json

    import safetensors.torch

    torch.manual_seed(0)
    tensors = {}
    for module_path, (d_in, d_out) in modules.items():
        tensors[f"base_model.model.{module_path}.lora_A.weight"] = torch.randn(rank, d_in) * 0.1
        tensors[f"base_model.model.{module_path}.lora_B.weight"] = torch.randn(d_out, rank) * 0.1
    path.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(tensors, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": alpha}))


def test_replay_hooks_match_reference_and_deactivate(tmp_path):
    from prime_rl.trainer.ttt_replay import TTTReplayManager

    model = TinyModel()
    ckpt = tmp_path / "r1" / "v1"
    write_adapter(ckpt, {"q_proj": (8, 8), "up_proj": (8, 16)})

    manager = TTTReplayManager(model, torch.device("cpu"))
    x = torch.randn(3, 8)
    base_out = model(x)

    manager.activate(str(ckpt))
    replay_out = model(x)

    # Deactivation restores the exact base forward (hooks are armed no-ops) — and lets the
    # reference below use the raw submodules without the hooks contributing.
    manager.activate(None)
    assert torch.equal(model(x), base_out)

    # Reference: y = up(q(x) + s*q_lora(x)) + s*up_lora(q(x) + s*q_lora(x))
    import safetensors.torch

    raw = safetensors.torch.load_file(ckpt / "adapter_model.safetensors")
    scale = 4.0 / 2
    q_a = raw["base_model.model.q_proj.lora_A.weight"].to(torch.bfloat16)
    q_b = raw["base_model.model.q_proj.lora_B.weight"].to(torch.bfloat16)
    u_a = raw["base_model.model.up_proj.lora_A.weight"].to(torch.bfloat16)
    u_b = raw["base_model.model.up_proj.lora_B.weight"].to(torch.bfloat16)
    h = model.q_proj(x) + (x.to(torch.bfloat16) @ q_a.T @ q_b.T).to(x.dtype) * scale
    expected = model.up_proj(h) + (h.to(torch.bfloat16) @ u_a.T @ u_b.T).to(x.dtype) * scale
    assert torch.allclose(replay_out, expected, atol=1e-5)
    assert not torch.allclose(replay_out, base_out)

    # Re-activation via the cache gives the same result.
    manager.activate(str(ckpt))
    assert torch.allclose(model(x), expected, atol=1e-5)


def test_replay_gradients_flow_to_base_not_adapter(tmp_path):
    from prime_rl.trainer.ttt_replay import TTTReplayManager

    model = TinyModel()
    ckpt = tmp_path / "r1" / "v1"
    write_adapter(ckpt, {"q_proj": (8, 8)})
    manager = TTTReplayManager(model, torch.device("cpu"))
    manager.activate(str(ckpt))

    out = model(torch.randn(3, 8))
    out.sum().backward()
    assert model.q_proj.weight.grad is not None
    assert model.up_proj.weight.grad is not None
    # The adapter tensors are inference-only constants: no grads accumulate anywhere else.
    (tensors, _) = manager._cache[str(ckpt)]
    for a, b in tensors.values():
        assert a.grad is None and b.grad is None


def test_replay_unknown_module_fails_loudly(tmp_path):
    from prime_rl.trainer.ttt_replay import TTTReplayManager

    model = TinyModel()
    ckpt = tmp_path / "bad" / "v1"
    write_adapter(ckpt, {"nonexistent_proj": (8, 8)})
    manager = TTTReplayManager(model, torch.device("cpu"))
    with pytest.raises(ValueError, match="does not exist on the trainer model"):
        manager.activate(str(ckpt))


# -- trajectories: adapter path resolution + QA recycling ------------------------------------


def make_trace(node_versions: list[int | None], info: dict | None = None):
    import verifiers.v1 as vf
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.types import AssistantMessage, UserMessage

    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    parent = None
    for i, version in enumerate(node_versions):
        sampled = i % 2 == 1  # alternate user/assistant
        message = AssistantMessage(content=f"a{i}") if sampled else UserMessage(content=f"u{i}")
        trace.nodes.append(
            MessageNode(
                parent=parent,
                message=message,
                sampled=sampled,
                token_ids=[10 + i, 20 + i],
                mask=[False, sampled],
                logprobs=[-0.5] if sampled else [],
                ttt_version=version,
            )
        )
        parent = len(trace.nodes) - 1
    if info:
        trace.info.update(info)
    return trace


def test_trace_to_samples_resolves_adapter_paths():
    from prime_rl.orchestrator.trajectories import trace_to_samples

    info = {"ttt": {"updates": [{"version": 1, "ckpt_path": "/ckpts/r/v1"}]}}
    # One linear branch sampled under version 1.
    trace = make_trace([1, 1, 1, 1], info=info)
    (sample,) = trace_to_samples(trace, env_name="e")
    assert sample.ttt_adapter_path == "/ckpts/r/v1"

    # Version 0 (pre-first-compaction) and no-TTT branches carry no ref.
    (sample0,) = trace_to_samples(make_trace([0, 0]), env_name="e")
    assert sample0.ttt_adapter_path is None
    (sample_none,) = trace_to_samples(make_trace([None, None]), env_name="e")
    assert sample_none.ttt_adapter_path is None


def test_trace_to_samples_missing_ckpt_is_fatal():
    from prime_rl.orchestrator.trajectories import trace_to_samples

    trace = make_trace([2, 2], info={"ttt": {"updates": [{"version": 1, "ckpt_path": "/x"}]}})
    with pytest.raises(ValueError, match="no checkpoint path"):
        trace_to_samples(trace, env_name="e")


class FakeTokenizer:
    """Chat-template stand-in: 3 prompt tokens + one token per answer word."""

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False):
        tokens = [1, 2, 3]  # question rendering
        if len(conversation) > 1:
            tokens = tokens + [100 + i for i in range(len(conversation[1]["content"].split()))]
        return tokens


def test_qa_recycle_samples_are_ce_routed():
    from prime_rl.orchestrator.trajectories import qa_recycle_samples

    info = {
        "ttt": {
            "updates": [
                {"version": 1, "qa_pairs": [{"question": "q?", "answer": "three word answer"}]},
                {"version": 2, "qa_pairs": [{"question": "q2?", "answer": "  "}]},  # blank: skipped
            ]
        }
    }
    trace = make_trace([1, 1], info=info)
    samples = qa_recycle_samples(trace, FakeTokenizer(), env_name="e")
    (sample,) = samples
    assert sample.token_ids == [1, 2, 3, 100, 101, 102]
    assert sample.mask == [False, False, False, True, True, True]
    # ce on the answer tokens, rl weight zero everywhere: pure SFT recycling.
    assert sample.ce_weights == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert sample.rl_weights == [0.0] * 6
    assert sample.advantages is None
    assert sample.ttt_adapter_path is None


# -- checkpoint GC ---------------------------------------------------------------------------


def test_ttt_gc_lifecycle(tmp_path):
    from dataclasses import dataclass, field

    from prime_rl.orchestrator.ttt_gc import TTTCheckpointGC

    shipped_dir = tmp_path / "ttt" / "r-shipped"
    dropped_dir = tmp_path / "ttt" / "r-dropped"
    for d in (shipped_dir / "v1", dropped_dir / "v1"):
        d.mkdir(parents=True)

    @dataclass
    class FakeRollout:
        info: dict = field(default_factory=dict)

    @dataclass
    class FakeBatch:
        rollouts: list
        samples: list

    shipped_rollout = FakeRollout(info={"ttt": {"updates": [{"version": 1, "ckpt_path": str(shipped_dir / "v1")}]}})
    dropped_rollout = FakeRollout(info={"ttt": {"updates": [{"version": 1, "ckpt_path": str(dropped_dir / "v1")}]}})
    batch = FakeBatch(
        rollouts=[shipped_rollout, dropped_rollout],
        samples=[make_sample(ttt_adapter_path=str(shipped_dir / "v1"))],
    )

    gc = TTTCheckpointGC()
    gc.track_batch(step=7, batch=batch)
    assert not dropped_dir.exists()  # dropped rollout's dir removed immediately
    assert shipped_dir.exists()  # shipped rollout's dir deferred

    gc.on_new_version(6)
    assert shipped_dir.exists()  # step 7 not consumed yet
    gc.on_new_version(7)
    assert not shipped_dir.exists()

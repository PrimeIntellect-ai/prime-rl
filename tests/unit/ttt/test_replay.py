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
    # The manager loads adapters in the model's parameter dtype (fp32 for this CPU model).
    dtype = model.q_proj.weight.dtype
    q_a = raw["base_model.model.q_proj.lora_A.weight"].to(dtype)
    q_b = raw["base_model.model.q_proj.lora_B.weight"].to(dtype)
    u_a = raw["base_model.model.up_proj.lora_A.weight"].to(dtype)
    u_b = raw["base_model.model.up_proj.lora_B.weight"].to(dtype)
    h = model.q_proj(x) + (x.to(dtype) @ q_a.T @ q_b.T).to(x.dtype) * scale
    expected = model.up_proj(h) + (h.to(dtype) @ u_a.T @ u_b.T).to(x.dtype) * scale
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


def test_replay_resolves_ac_wrapped_modules(tmp_path):
    """Under [trainer.model.ac] full mode, checkpoint_wrapper inserts
    ``._checkpoint_wrapped_module.`` into FQNs — the manager must still resolve the
    plain PEFT paths and produce the same hook math as the un-wrapped model."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from prime_rl.trainer.ttt_replay import TTTReplayManager

    torch.manual_seed(1)
    ckpt = tmp_path / "r1" / "v1"
    write_adapter(ckpt, {"block.q_proj": (8, 8), "block.up_proj": (8, 16)})
    x = torch.randn(3, 8)

    class Outer(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, x):
            return self.block(x)

    # Un-wrapped reference.
    torch.manual_seed(2)
    ref_model = Outer(TinyModel())
    ref_manager = TTTReplayManager(ref_model, torch.device("cpu"))
    ref_manager.activate(str(ckpt))
    expected = ref_model(x)

    # Same weights, block wrapped with AC.
    torch.manual_seed(2)
    model = Outer(checkpoint_wrapper(TinyModel()))
    assert any("_checkpoint_wrapped_module" in name for name, _ in model.named_modules())
    manager = TTTReplayManager(model, torch.device("cpu"))
    manager.activate(str(ckpt))
    assert torch.allclose(model(x), expected, atol=1e-6)


def test_replay_hooks_survive_compile(tmp_path):
    """Hooks installed BEFORE torch.compile stay live (dynamo skips guards on hooks added
    later — the eager-install path is what makes replay correct under compile)."""
    from prime_rl.trainer.ttt_replay import TTTReplayManager

    torch.manual_seed(3)
    model = TinyModel()
    ckpt1 = tmp_path / "r1" / "v1"
    ckpt2 = tmp_path / "r1" / "v2"
    write_adapter(ckpt1, {"q_proj": (8, 8)})
    write_adapter(ckpt2, {"q_proj": (8, 8)}, alpha=8.0)  # different scale -> different output

    manager = TTTReplayManager(model, torch.device("cpu"))
    manager.install_hooks_on_all_linears()
    x = torch.randn(3, 8)
    manager.activate(str(ckpt1))
    eager_ref1 = model(x)
    manager.activate(str(ckpt2))
    eager_ref2 = model(x)
    manager.activate(None)
    base_ref = model(x)

    compiled = torch.compile(model, backend="eager")
    manager.activate(str(ckpt1))
    assert torch.allclose(compiled(x), eager_ref1, atol=1e-6)
    manager.activate(str(ckpt2))  # switching adapters must take effect under compile too
    assert torch.allclose(compiled(x), eager_ref2, atol=1e-6)
    assert not torch.allclose(eager_ref1, eager_ref2)
    manager.activate(None)
    assert torch.allclose(compiled(x), base_ref, atol=1e-6)


def test_activate_failure_does_not_poison_current(tmp_path):
    from prime_rl.trainer.ttt_replay import TTTReplayManager

    manager = TTTReplayManager(TinyModel(), torch.device("cpu"))
    missing = str(tmp_path / "nope" / "v1")
    with pytest.raises(Exception):
        manager.activate(missing)
    # A failed activate must not record the path — retrying raises again instead of
    # silently running the base model.
    with pytest.raises(Exception):
        manager.activate(missing)


def test_load_rejects_rslora_dora_and_unknown_keys(tmp_path):
    import json

    import safetensors.torch

    from prime_rl.trainer.ttt_replay import TTTReplayManager

    manager = TTTReplayManager(TinyModel(), torch.device("cpu"))

    rslora = tmp_path / "rslora" / "v1"
    write_adapter(rslora, {"q_proj": (8, 8)})
    config = json.loads((rslora / "adapter_config.json").read_text())
    config["use_rslora"] = True
    (rslora / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="use_rslora"):
        manager.activate(str(rslora))

    unknown = tmp_path / "unknown" / "v1"
    write_adapter(unknown, {"q_proj": (8, 8)})
    raw = safetensors.torch.load_file(unknown / "adapter_model.safetensors")
    raw["base_model.model.q_proj.lora_magnitude_vector"] = torch.randn(8)
    safetensors.torch.save_file(raw, unknown / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="unrecognized tensor keys"):
        manager.activate(str(unknown))


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
    # ce NLL is temperature-free MLE: T=1 always, and pre-stamped so the sink's
    # rollout-temperature fill skips it.
    assert sample.temperatures == [1.0] * 6
    assert sample.advantages is None
    assert sample.ttt_adapter_path is None


def test_trace_to_samples_stamps_qa_temperature_on_ttt_qa_branches():
    from prime_rl.orchestrator.trajectories import trace_to_samples

    trace = make_trace([None, None])
    trace.nodes[1].ttt_qa = True  # the branch's sampled node is a QA side-generation
    (sample,) = trace_to_samples(trace, env_name="e", qa_temperature=0.9)
    assert sample.temperatures == [0.9] * len(sample.token_ids)
    # None = same-as-rollout: leave [] for the sink's rollout-temperature fill.
    trace2 = make_trace([None, None])
    trace2.nodes[1].ttt_qa = True
    (sample2,) = trace_to_samples(trace2, env_name="e", qa_temperature=None)
    assert sample2.temperatures == []
    # Non-QA branches keep [] even when qa_temperature is set.
    (sample3,) = trace_to_samples(make_trace([None, None]), env_name="e", qa_temperature=0.9)
    assert sample3.temperatures == []


def test_sink_temperature_stamp_skips_prestamped_samples():
    """The sink's rollout-temperature fill (TrainSink.process_group) must only touch
    unstamped samples — replicate its predicate over pre-stamped ce/qa samples."""
    from prime_rl.transport import TrainingSample

    def make(temperatures):
        return TrainingSample(
            token_ids=[1, 2], mask=[True, True], logprobs=[0.0, 0.0], temperatures=temperatures, env_name="e"
        )

    samples = [make([]), make([1.0, 1.0]), make([0.9, 0.9])]
    for sample in samples:  # the sink's stamping loop
        if not sample.temperatures:
            sample.temperatures = [0.7] * len(sample.token_ids)
    assert [s.temperatures for s in samples] == [[0.7, 0.7], [1.0, 1.0], [0.9, 0.9]]


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
        has_error: bool = False
        is_filtered: bool = False

    @dataclass
    class FakeBatch:
        rollouts: list
        samples: list

    def rollout_for(d, **kwargs):
        return FakeRollout(info={"ttt": {"updates": [{"version": 1, "ckpt_path": str(d / "v1")}]}}, **kwargs)

    shipped_rollout = rollout_for(shipped_dir)
    dropped_rollout = rollout_for(dropped_dir, is_filtered=True)
    batch = FakeBatch(
        rollouts=[shipped_rollout, dropped_rollout],
        samples=[make_sample(ttt_adapter_path=str(shipped_dir / "v1"))],
    )

    gc = TTTCheckpointGC()
    gc.track_batch(step=7, batch=batch)
    assert not dropped_dir.exists()  # filtered rollout's dir removed immediately
    assert shipped_dir.exists()  # shipped rollout's dir deferred

    gc.on_new_version(6)
    assert shipped_dir.exists()  # step 7 not consumed yet
    gc.on_new_version(7)
    assert not shipped_dir.exists()


def test_ttt_gc_carries_window_straddlers(tmp_path):
    """``batch.rollouts`` is the ARRIVAL window, not the ship cohort: a rollout can arrive
    in step N's window and ship its samples in step N+1 (batch overflow; group finalized
    after the cut). Its adapter dirs must NOT be deleted at step N — they're carried until
    they ship (then deferred + freed on consumption) — otherwise the trainer's replay load
    hard-fails at N+1."""
    from dataclasses import dataclass, field

    from prime_rl.orchestrator.ttt_gc import TTTCheckpointGC

    straddler_dir = tmp_path / "ttt" / "r-straddler"
    (straddler_dir / "v1").mkdir(parents=True)

    @dataclass
    class FakeRollout:
        info: dict = field(default_factory=dict)
        has_error: bool = False
        is_filtered: bool = False

    @dataclass
    class FakeBatch:
        rollouts: list
        samples: list

    straddler = FakeRollout(info={"ttt": {"updates": [{"version": 1, "ckpt_path": str(straddler_dir / "v1")}]}})

    gc = TTTCheckpointGC()
    # Step 1: the rollout is in the arrival window but its samples don't ship yet.
    gc.track_batch(step=1, batch=FakeBatch(rollouts=[straddler], samples=[make_sample()]))
    assert straddler_dir.exists()  # carried, not deleted

    # Step 2: its samples ship (it is no longer in the arrival window).
    gc.track_batch(
        step=2, batch=FakeBatch(rollouts=[], samples=[make_sample(ttt_adapter_path=str(straddler_dir / "v1"))])
    )
    assert straddler_dir.exists()  # now deferred on step 2
    gc.on_new_version(2)
    assert not straddler_dir.exists()  # freed once consumed


def test_eval_rollout_ckpt_disposal(tmp_path):
    """Eval rollouts do TTT at inference time but never reach TrainSink/TTTCheckpointGC —
    their adapter dirs are dismissed immediately when the rollout arrives."""
    from dataclasses import dataclass, field

    from prime_rl.orchestrator.ttt_gc import dispose_eval_rollout_ckpts

    ckpt_dir = tmp_path / "ttt" / "r-eval"
    (ckpt_dir / "v1").mkdir(parents=True)

    @dataclass
    class FakeRollout:
        info: dict = field(default_factory=dict)

    ttt_rollout = FakeRollout(info={"ttt": {"updates": [{"version": 1, "ckpt_path": str(ckpt_dir / "v1")}]}})
    dispose_eval_rollout_ckpts(ttt_rollout)
    assert not ckpt_dir.exists()

    # No ttt info → no dirs, no error.
    dispose_eval_rollout_ckpts(FakeRollout())


def test_shared_sampled_prefix_trained_once():
    """Branches sharing a SAMPLED prefix (QA side-branches, subagent forks) grant each
    sampled node its mask exactly once — the first branch trains it, later branches carry
    it as context — so shared tokens never get N× gradient weight."""
    import verifiers.v1 as vf
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.types import AssistantMessage, UserMessage

    from prime_rl.orchestrator.trajectories import trace_to_samples

    trace = vf.Trace(task=vf.Task(idx=0, prompt="t"))
    # Shared trajectory: user -> assistant (sampled).
    trace.nodes.append(MessageNode(parent=None, message=UserMessage(content="u"), token_ids=[1], mask=[False]))
    trace.nodes.append(
        MessageNode(
            parent=0,
            message=AssistantMessage(content="a"),
            sampled=True,
            token_ids=[2, 3],
            mask=[True, True],
            logprobs=[-0.1, -0.2],
        )
    )
    # Two QA-style forks off the assistant leaf, each with its own sampled tail.
    for k in (0, 1):
        base = len(trace.nodes)
        trace.nodes.append(
            MessageNode(parent=1, message=UserMessage(content=f"q{k}"), token_ids=[10 + k], mask=[False])
        )
        trace.nodes.append(
            MessageNode(
                parent=base,
                message=AssistantMessage(content=f"ans{k}"),
                sampled=True,
                token_ids=[20 + k],
                mask=[True],
                logprobs=[-0.3],
            )
        )

    samples = trace_to_samples(trace, env_name="e")
    assert len(samples) == 2  # two leaves -> two branches
    # The shared assistant tokens [2, 3] are mask-True in exactly one sample.
    trained_counts = [sum(s.mask[1:3]) for s in samples]  # positions of tokens 2,3
    assert sorted(trained_counts) == [0, 2]
    # Each branch's own tail is trainable in its own sample.
    for s in samples:
        assert s.mask[-1] is True or s.mask[-1] == True  # noqa: E712


def test_qa_recycle_conditions_on_system_and_tools():
    """Recycled QA samples render [system, Q, A] with the rollout's tools — the same frame
    the adapter training used — with loss (ce) only on the answer."""
    from prime_rl.orchestrator.trajectories import qa_recycle_samples

    class RecordingTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False, tools=None):
            self.calls.append({"roles": [m["role"] for m in conversation], "tools": tools})
            # 2 tokens per message + 1 for the generation prompt.
            n = 2 * len(conversation) + (1 if add_generation_prompt else 0)
            return list(range(n))

    info = {
        "ttt": {
            "system_prompt": "sys",
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "updates": [{"version": 1, "qa_pairs": [{"question": "q", "answer": "a"}]}],
        }
    }
    trace = make_trace([1, 1], info=info)
    tokenizer = RecordingTokenizer()
    (sample,) = qa_recycle_samples(trace, tokenizer, env_name="e")
    # Rendered with system head + tools on both the full and the prompt-only calls.
    assert all(c["roles"][0] == "system" for c in tokenizer.calls)
    assert all(c["tools"] is not None for c in tokenizer.calls)
    # full = 6 tokens ([sys, user, assistant]), prompt = 5 ([sys, user] + gen prompt) but the
    # prefix check fails (prompt ids 0..4 == full ids 0..4) -> prompt_len 5, answer len 1.
    assert len(sample.token_ids) == 6
    assert sample.mask == [False] * 5 + [True]
    assert sample.ce_weights == [0.0] * 5 + [1.0]


# -- group-level meta-extraction --------------------------------------------------------------


def rollout_with_pairs(reward: float, pairs: list[dict]):
    from dataclasses import dataclass, field

    @dataclass
    class FakeGroupRollout:
        reward: float
        info: dict = field(default_factory=dict)
        samples: list = field(default_factory=list)

    return FakeGroupRollout(
        reward=reward,
        info={
            "ttt": {
                "system_prompt": "sys",
                "tools": [{"type": "function", "function": {"name": "search"}}],
                "updates": [{"version": 1, "qa_pairs": pairs}],
            }
        },
    )


class FakeChat:
    """AsyncOpenAI stand-in: records the prompt, returns scripted content."""

    def __init__(self, content: str, fail: bool = False):
        self._content = content
        self._fail = fail
        self.prompts: list[str] = []

        outer = self

        class _Completions:
            async def create(self, *, model, messages, max_tokens):
                if outer._fail:
                    raise RuntimeError("model down")
                outer.prompts.append(messages[0]["content"])

                class _Msg:
                    content = outer._content

                class _Choice:
                    message = _Msg()

                class _Completion:
                    choices = [_Choice()]

                return _Completion()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


@pytest.mark.asyncio
async def test_meta_extraction_contrasts_rewards():
    from verifiers.v1.ttt import QAConfig

    from prime_rl.orchestrator.qa_meta import extract_meta_lessons

    good = rollout_with_pairs(1.0, [{"type": "qa", "question": "How to find X for task T?", "answer": "use search"}])
    bad = rollout_with_pairs(0.0, [{"type": "lesson", "question": "When S rate-limits?", "answer": "give up"}])
    lesson = "<item><type>lesson</type><question>When solving research task T, what search strategy works?</question><answer>narrow queries with site: filters</answer></item>"
    chat = FakeChat(lesson)
    items = await extract_meta_lessons([good, bad], QAConfig(meta_lessons=True), chat, "m")

    (prompt,) = chat.prompts
    # Both attempts appear, with rewards and their pairs.
    assert "Attempt 1 (reward: 1.000)" in prompt
    assert "Attempt 2 (reward: 0.000)" in prompt
    assert "use search" in prompt and "give up" in prompt
    assert "SELF-CONTAINED" in prompt
    (item,) = items
    assert item["type"] == "lesson"
    assert "search strategy" in item["question"]


@pytest.mark.asyncio
async def test_meta_extraction_needs_two_with_pairs_and_fails_open():
    from verifiers.v1.ttt import QAConfig

    from prime_rl.orchestrator.qa_meta import extract_meta_lessons

    good = rollout_with_pairs(1.0, [{"question": "q", "answer": "a"}])
    empty = rollout_with_pairs(0.5, [])
    # Only one rollout has pairs -> nothing to contrast.
    assert await extract_meta_lessons([good, empty], QAConfig(), FakeChat("x"), "m") == []
    # A failing model call is enrichment, not an error.
    bad_chat = FakeChat("", fail=True)
    two = [good, rollout_with_pairs(0.0, [{"question": "q2", "answer": "a2"}])]
    assert await extract_meta_lessons(two, QAConfig(), bad_chat, "m") == []


def test_meta_lesson_samples_are_ce_routed_with_conditioning():
    from prime_rl.orchestrator.qa_meta import meta_lesson_samples

    class RecordingTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False, tools=None):
            self.calls.append({"roles": [m["role"] for m in conversation], "tools": tools})
            return list(range(2 * len(conversation) + (1 if add_generation_prompt else 0)))

    group = [rollout_with_pairs(1.0, [{"question": "q", "answer": "a"}])]
    items = [
        {"type": "lesson", "question": "When X, do?", "answer": "Y"},
        {"type": "lesson", "question": "empty", "answer": "  "},  # skipped
    ]
    tokenizer = RecordingTokenizer()
    (sample,) = meta_lesson_samples(items, group, tokenizer, env_name="e")
    assert all(c["roles"][0] == "system" for c in tokenizer.calls)  # group conditioning
    assert all(c["tools"] is not None for c in tokenizer.calls)
    assert sample.rl_weights == [0.0] * len(sample.token_ids)
    assert sample.ce_weights[-1] == 1.0 and sample.ce_weights[0] == 0.0
    assert sample.temperatures == [1.0] * len(sample.token_ids)  # ce is T-free MLE
    assert sample.ttt_adapter_path is None  # trains the live policy

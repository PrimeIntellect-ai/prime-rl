"""TTT trainer v2 (FSDP/MultiLoRA engine): the GPU-independent logic — slot registry
(claim/release/determinism), job validation, packing, and the server-side batching plumbing.
The full engine (setup_model + FSDP + packed forwards) needs CUDA and runs in the GPU
integration suite; here the trainer is built via __new__ with the registry fields only."""

import pytest

pytest.importorskip("torch")

from prime_rl.configs.ttt import (  # noqa: E402
    TTTLoRAConfig,
    TTTOptimizerConfig,
    TTTServiceConfig,
)
from prime_rl.ttt.trainer_v2 import SlotState, TTTTrainerV2, UpdateJob  # noqa: E402


def make_registry(max_slots: int = 4, **engine_overrides) -> TTTTrainerV2:
    """A TTTTrainerV2 with only the slot-registry state (no model, no distributed)."""

    class FakeRuns:
        def __init__(self):
            self.resets: list[int] = []

        def reset_run_parameters(self, idx: int) -> None:
            self.resets.append(idx)

    trainer = TTTTrainerV2.__new__(TTTTrainerV2)
    trainer.config = TTTServiceConfig(
        engine={"type": "fsdp", "max_slots": max_slots, **engine_overrides},
        lora=TTTLoRAConfig(rank=4),
        optim=TTTOptimizerConfig(),
        inference_admin_urls=[],
    )
    trainer.max_slots = max_slots
    trainer.slots = {}
    trainer.free_idxs = set(range(max_slots))
    trainer.runs = FakeRuns()
    return trainer


def job(rollout_id: str, n: int = 8, seq_no: int = 1, **kwargs) -> UpdateJob:
    return UpdateJob(
        rollout_id=rollout_id,
        adapter_name=f"ttt-{rollout_id}",
        token_ids=list(range(n)),
        loss_mask=[True] * n,
        seq_no=seq_no,
        **kwargs,
    )


def test_slot_claim_is_deterministic_and_reset():
    trainer = make_registry(max_slots=2)
    s1 = trainer._claim("r1", "ttt-r1")
    s2 = trainer._claim("r2", "ttt-r2")
    assert (s1.idx, s2.idx) == (0, 1)  # lowest free index — deterministic across ranks
    assert trainer.runs.resets == [0, 1]  # zero-init on claim (B=0 → base-identical)
    assert trainer._claim("r1", "ttt-r1") is s1  # idempotent
    assert not trainer.free_idxs

    trainer.release("r1")
    assert trainer.free_idxs == {0}
    s3 = trainer._claim("r3", "ttt-r3")
    assert s3.idx == 0  # freed slot reused, reset again
    assert trainer.runs.resets == [0, 1, 0]


def test_validate_job_is_pure():
    trainer = make_registry(max_slots=1)
    good = job("r1")
    trainer.validate_job(good)
    assert trainer.slots == {} and trainer.free_idxs == {0}  # no mutation

    with pytest.raises(ValueError, match="must align"):
        trainer.validate_job(UpdateJob("r1", "a", [1, 2], [True], seq_no=1))
    with pytest.raises(ValueError, match="expected seq_no 1"):
        trainer.validate_job(job("r1", seq_no=2))
    with pytest.raises(ValueError, match="no trainable sequences"):
        trainer.validate_job(job("r1", train_rollout=False))
    # All-context mask: no trainable target after the next-token shift, no QA fallback.
    all_context = job("r1")
    all_context.loss_mask = [True] + [False] * (len(all_context.token_ids) - 1)
    with pytest.raises(ValueError, match="no trainable target tokens"):
        trainer.validate_job(all_context)
    # Oversized job: can't fit any forward, would poison packing on every retry.
    trainer.config.engine.max_tokens_per_forward = 100
    with pytest.raises(ValueError, match="job too large"):
        trainer.validate_job(job("r1", n=101))
    trainer.config.engine.max_tokens_per_forward = 65536

    # Slot exhaustion is a validation error for NEW rollouts, not existing ones.
    trainer.slots["other"] = SlotState("other", "ttt-other", idx=0)
    trainer.free_idxs = set()
    with pytest.raises(ValueError, match="no free TTT adapter slots"):
        trainer.validate_job(job("r1"))
    trainer.slots["other"].version = 0
    trainer.validate_job(job("other"))  # existing rollout: fine


def test_pack_respects_token_cap_and_keeps_jobs_atomic():
    trainer = make_registry(max_slots=8, max_tokens_per_forward=1024)
    trainer.config.engine.max_tokens_per_forward = 100  # below the config floor, fine for the unit
    jobs = [job(f"r{i}", n=60) for i in range(3)]  # 60 tokens each, cap 100
    for j in jobs:
        j.sequences = [(j.token_ids, j.loss_mask)]
    bins = trainer._pack(jobs)
    assert len(bins) == 3  # 60+60 > 100: one job per bin
    small = [job(f"s{i}", n=30) for i in range(4)]
    for j in small:
        j.sequences = [(j.token_ids, j.loss_mask)]
    bins = trainer._pack(small)
    assert len(bins) == 2  # 3x30 <= 100 → [3, 1]
    assert sorted(len(b) for b in bins) == [1, 3]
    # Every job appears exactly once.
    seen = [j.rollout_id for b in bins for j in b]
    assert sorted(seen) == sorted(j.rollout_id for j in small)


def test_server_validation_splits_bad_jobs():
    from prime_rl.ttt.server_v2 import _Pending, _validate_and_split

    trainer = make_registry(max_slots=2)
    good = _Pending(job=job("r1"))
    bad = _Pending(job=job("r2", seq_no=5))
    valid = _validate_and_split(trainer, [good, bad])
    assert valid == [good]
    assert bad.done.is_set() and "expected seq_no 1" in bad.error
    assert not good.done.is_set()


def test_service_config_engine_dispatch():
    c = TTTServiceConfig()
    assert c.engine.type == "peft"
    c2 = TTTServiceConfig(
        engine={"type": "fsdp", "max_slots": 128, "model": {"cp": 8, "cp_style": "ulysses", "impl": "custom"}},
        model={"name": "zai-org/GLM-4.5-Air"},
        lora={"rank": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    )
    mc = c2.engine.to_model_config(c2.lora)
    assert mc.name == "zai-org/GLM-4.5-Air"
    assert mc.cp == 8 and mc.cp_style == "ulysses"
    assert mc.lora is not None and mc.lora.rank == 16
    assert mc.lora.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_fsdp_engine_rejects_context_parallelism():
    """update_batch feeds full packed sequences to every rank — CP would silently
    mis-shard; the engine must refuse at startup."""
    config = TTTServiceConfig(
        engine={"type": "fsdp", "model": {"cp": 8, "cp_style": "ulysses"}},
        lora=TTTLoRAConfig(rank=4),
    )
    with pytest.raises(ValueError, match="context parallelism"):
        TTTTrainerV2(config)


def make_cpu_trainer(max_slots: int = 4, **engine_overrides):
    """The registry trainer + just enough CPU fakes to run update_batch end-to-end: a
    graph-bearing fake forward (grads flow to a dummy param), stubbed checkpointing."""
    import torch

    from prime_rl.trainer.models.layers.lora import set_lora_num_tokens, set_multilora_scaling

    # update_batch copies into the global LORA_NUM_TOKENS in place; the real service sizes
    # it (and the paired SCALING_FACTORS) via setup_multi_run_manager, which this fake
    # skips — pin both globals to a coherent max_slots shape so the test is isolated from
    # whatever shapes other suites left behind (the setters cross-assert shapes).
    set_lora_num_tokens(None, reset_reference=True)
    set_multilora_scaling(torch.ones(max_slots, dtype=torch.float32), reset_reference=True)
    set_lora_num_tokens(torch.zeros(max_slots, dtype=torch.int32), reset_reference=True)

    trainer = make_registry(max_slots=max_slots, **engine_overrides)
    trainer.device = torch.device("cpu")
    trainer.logger = type("L", (), {"info": lambda self, *a, **k: None})()
    dummy = torch.nn.Parameter(torch.zeros(1))
    trainer._forward_logprobs = lambda input_ids, position_ids, labels: (
        dummy - torch.zeros_like(input_ids, dtype=torch.float32)
    )
    trainer._optimizer = lambda state: torch.optim.SGD([dummy], lr=0.0)
    trainer.save_checkpoint = lambda state: f"/fake/{state.rollout_id}/v{state.version}"
    return trainer


def test_update_batch_isolates_poisoned_job_and_rolls_back_slot():
    trainer = make_cpu_trainer(max_slots=4)
    healthy = [job("r1"), job("r2")]
    # Passes rank-0 validation (qa_pairs non-empty) but yields no sequences in prepare —
    # the failure lands AFTER the slot claim, exercising the rollback path.
    poisoned = UpdateJob(
        rollout_id="bad",
        adapter_name="ttt-bad",
        token_ids=list(range(8)),
        loss_mask=[True] * 8,
        seq_no=1,
        train_rollout=False,
        qa_pairs=[{"question": "q", "answer": "   "}],
    )
    results = trainer.update_batch([healthy[0], poisoned, healthy[1]])
    assert results["r1"]["version"] == 1 and results["r2"]["version"] == 1
    assert "no trainable sequences" in results["bad"]["error"]
    # The poisoned job's slot claim was rolled back: only r1/r2 hold slots.
    assert set(trainer.slots) == {"r1", "r2"}
    assert trainer.free_idxs == {2, 3}


def test_update_batch_all_failed_returns_without_forward():
    trainer = make_cpu_trainer(max_slots=2)
    trainer._forward_logprobs = lambda *a, **k: pytest.fail("forward must not run")
    bad = job("r1", seq_no=7)
    results = trainer.update_batch([bad])
    assert "expected seq_no 1" in results["r1"]["error"]
    assert trainer.slots == {} and trainer.free_idxs == {0, 1}


def test_update_batch_rejects_oversized_after_qa_tokenization():
    trainer = make_cpu_trainer(max_slots=2, max_tokens_per_forward=1024)
    trainer.config.engine.max_tokens_per_forward = 100  # below config floor, fine for the unit
    # QA pairs only get tokenized in prepare_job — fake a render that busts the cap.
    trainer._tokenize_qa = lambda pairs, sp, tools: [(list(range(200)), [False] + [True] * 199)]
    oversized = job("r1", n=8, qa_pairs=[{"question": "q", "answer": "a"}])
    results = trainer.update_batch([oversized])
    assert "job too large after QA tokenization" in results["r1"]["error"]
    assert trainer.slots == {} and trainer.free_idxs == {0, 1}  # claim rolled back


def test_update_batch_asserts_unique_rollout_ids():
    trainer = make_cpu_trainer(max_slots=2)
    with pytest.raises(AssertionError, match="duplicate rollout_ids"):
        trainer.update_batch([job("r1"), job("r1", seq_no=2)])


def test_collector_dedups_same_rollout_keeping_first():
    from prime_rl.ttt.server_v2 import _dedup_pendings, _Pending

    first = _Pending(job=job("r1", seq_no=1))
    second = _Pending(job=job("r1", seq_no=2))
    other = _Pending(job=job("r2"))
    batch, deferred = _dedup_pendings([first, second, other])
    assert batch == [first, other]  # first pending per rollout stays, order preserved
    assert deferred == [second]  # later duplicate deferred to the next batch


def test_update_batch_replays_duplicate_seq_no_from_cache():
    trainer = make_cpu_trainer(max_slots=2)
    first = trainer.update_batch([job("r1")])["r1"]
    assert first["version"] == 1
    # A retry after a lost response replays the exact same seq_no: answer from the cache —
    # no forward, no optimizer step, no version bump.
    trainer._forward_logprobs = lambda *a, **k: pytest.fail("forward must not run on replay")
    replay = trainer.update_batch([job("r1")])["r1"]
    assert replay is first  # the cached result, same ckpt_path/version
    assert trainer.slots["r1"].version == 1


def test_replay_of_older_seq_no_still_409s():
    trainer = make_cpu_trainer(max_slots=2)
    trainer.update_batch([job("r1")])
    trainer.update_batch([job("r1", seq_no=2)])
    # seq_no strictly below the slot version is NOT a replay of the last update: 409.
    with pytest.raises(ValueError, match="expected seq_no 3"):
        trainer.validate_job(job("r1", seq_no=1))
    results = trainer.update_batch([job("r1", seq_no=1)])
    assert "expected seq_no 3" in results["r1"]["error"]


def test_seq_no_equal_version_without_cache_still_409s():
    trainer = make_registry(max_slots=2)
    # A slot at version 1 with no cached result (e.g. after a restart) can't prove the
    # update applied — keep the strict 409.
    trainer.slots["r1"] = SlotState("r1", "ttt-r1", idx=0, version=1)
    trainer.free_idxs = {1}
    with pytest.raises(ValueError, match="expected seq_no 2"):
        trainer.validate_job(job("r1", seq_no=1))

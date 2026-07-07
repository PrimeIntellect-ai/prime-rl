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

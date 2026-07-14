"""TTT trainer v2 (FSDP/MultiLoRA engine): the GPU-independent logic — slot registry
(claim/release/determinism), job validation, packing, and the server-side batching plumbing.
The full engine (setup_model + FSDP + packed forwards) needs CUDA and runs in the GPU
integration suite; here the trainer is built via __new__ with the registry fields only."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from prime_rl.configs.ttt import (  # noqa: E402
    TTTLoRAConfig,
    TTTOptimizerConfig,
    TTTServiceConfig,
)
from prime_rl.ttt.trainer_v2 import (  # noqa: E402
    SlotState,
    TTTTrainerV2,
    UpdateJob,
    _patch_frozen_fused_lm_head,
)


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
    trainer.adapter_names = {}
    trainer.free_idxs = set(range(max_slots))
    trainer.runs = FakeRuns()
    trainer.vocab_size = 100_000
    trainer.world = SimpleNamespace(is_master=False)
    trainer.ckpt_root = Path("outputs/ttt-test-registry")
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

    trainer.release("r1", "ttt-r1")
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
    # Malformed qa_pairs must 409 at validation — past this point a KeyError inside
    # _tokenize_qa would escape the per-job ValueError isolation and (pre-fix) hit the
    # work loop's os._exit(1) fail-fast: one bad request killed the whole service.
    with pytest.raises(ValueError, match="malformed qa_pairs\\[0\\]"):
        trainer.validate_job(job("r1", qa_pairs=[{"answer": "a but no question"}]))
    with pytest.raises(ValueError, match="malformed qa_pairs\\[1\\]"):
        trainer.validate_job(job("r1", qa_pairs=[{"question": "q", "answer": "a"}, "not-a-dict"]))
    with pytest.raises(ValueError, match="'answer' must be a string"):
        trainer.validate_job(job("r1", qa_pairs=[{"question": "q", "answer": 42}]))


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
    assert good.job.sequences == [(good.job.token_ids, good.job.loss_mask)]
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

    # Pin the LORA_NUM_TOKENS/SCALING_FACTORS globals to a coherent max_slots shape so the
    # test is isolated from whatever shapes other suites left behind.
    set_lora_num_tokens(None, reset_reference=True)
    set_multilora_scaling(torch.ones(max_slots, dtype=torch.float32), reset_reference=True)
    set_lora_num_tokens(torch.zeros(max_slots, dtype=torch.int32), reset_reference=True)

    trainer = make_registry(max_slots=max_slots, **engine_overrides)
    trainer._tokenizer = object()  # QA render is never reached in these tests
    trainer.device = torch.device("cpu")
    trainer.parallel_dims = SimpleNamespace(ep_enabled=False)
    trainer.logger = type("L", (), {"info": lambda self, *a, **k: None})()
    dummy = torch.nn.Parameter(torch.zeros(1))
    trainer._forward_logprobs = lambda input_ids, position_ids, labels: (
        dummy - torch.zeros_like(input_ids, dtype=torch.float32)
    )
    trainer._optimizer = lambda state: torch.optim.SGD([dummy], lr=0.0)
    trainer.save_checkpoint = lambda state: f"/fake/{state.rollout_id}/v{state.version}"
    return trainer


def test_update_batch_isolates_poisoned_job_and_never_claims_its_slot():
    trainer = make_cpu_trainer(max_slots=4)
    healthy = [job("r1"), job("r2")]
    # Passes rank-0 validation (qa_pairs non-empty) but yields no sequences in prepare.
    # Slot claims now happen AFTER the per-job error net, so the poisoned job never
    # claims one — no rollback needed, isolation semantics unchanged.
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
    # The poisoned job never held a slot: only r1/r2 hold slots.
    assert set(trainer.slots) == {"r1", "r2"}
    assert trainer.free_idxs == {2, 3}


def test_claim_time_fault_escapes_update_batch():
    """A fault during the GPU-mutating slot claim (reset_run_parameters — CUDA territory)
    must NOT be downgraded to a per-job error: it escapes update_batch so the work loop's
    os._exit fail-fast preserves the rank-lockstep contract."""

    def boom(idx):
        raise RuntimeError("simulated CUDA fault in reset_run_parameters")

    trainer = make_cpu_trainer(max_slots=2)
    trainer.runs.reset_run_parameters = boom
    with pytest.raises(RuntimeError, match="simulated CUDA fault"):
        trainer.update_batch([job("r1")])


def test_slot_exhaustion_within_one_batch_is_a_per_job_error():
    """With the claim deferred past the error net, in-batch admissions must count against
    the free-slot budget — otherwise the deferred _claim would raise past the net."""
    trainer = make_cpu_trainer(max_slots=1)
    results = trainer.update_batch([job("r1"), job("r2")])
    assert results["r1"]["version"] == 1
    assert "no free TTT adapter slots" in results["r2"]["error"]
    assert set(trainer.slots) == {"r1"} and trainer.free_idxs == set()


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
    assert trainer.slots == {} and trainer.free_idxs == {0, 1}  # never claimed


def test_update_batch_asserts_unique_rollout_ids():
    trainer = make_cpu_trainer(max_slots=2)
    with pytest.raises(AssertionError, match="duplicate rollout_ids"):
        trainer.update_batch([job("r1"), job("r1", seq_no=2)])


def test_collector_dedups_same_rollout_keeping_first():
    """One collector drain keeps the FIRST pending per rollout (order preserved) and
    re-queues the same-rollout follow-up for the next batch."""
    import threading
    from queue import Queue

    from prime_rl.ttt.server_v2 import _collector_loop, _Pending

    trainer = make_registry(max_slots=4, max_batch_wait_seconds=0)
    first = _Pending(job=job("r1", seq_no=1))
    second = _Pending(job=job("r1", seq_no=1))  # retry of the same update, one drain later
    other = _Pending(job=job("r2"))
    batch_queue: Queue = Queue()
    work_queue: Queue = Queue()
    for pending in (first, second, other):
        batch_queue.put(pending)
    collector = threading.Thread(
        target=_collector_loop, args=(trainer.config, trainer, batch_queue, work_queue), daemon=True
    )
    collector.start()
    kind, jobs, valid = work_queue.get(timeout=5)
    assert kind == "update"
    assert valid == [first, other]  # first pending per rollout stays, order preserved
    kind, jobs, valid = work_queue.get(timeout=5)
    assert valid == [second]  # the deferred duplicate rides the next batch


def test_update_batch_replays_duplicate_seq_no_from_cache():
    trainer = make_cpu_trainer(max_slots=2)
    first = trainer.update_batch([job("r1")])["r1"]
    assert first["version"] == 1
    # A retry after a lost response replays the exact same seq_no: answer from the cache —
    # no forward, no optimizer step, no version bump.
    real_forward = trainer._forward_logprobs
    trainer._forward_logprobs = lambda *a, **k: pytest.fail("forward must not run on replay")
    replay = trainer.update_batch([job("r1")])["r1"]
    assert replay is first  # the cached result, same ckpt_path/version
    assert trainer.slots["r1"].version == 1

    changed = job("r1")
    changed.token_ids[-1] = 99
    assert "does not match the cached request" in trainer.update_batch([changed])["r1"]["error"]

    # seq_no strictly below the slot version is NOT a replay of the last update: 409.
    trainer._forward_logprobs = real_forward
    trainer.update_batch([job("r1", seq_no=2)])
    assert "expected seq_no 3" in trainer.update_batch([job("r1", seq_no=1)])["r1"]["error"]


def test_seq_no_equal_version_without_cache_still_409s():
    trainer = make_registry(max_slots=2)
    # A slot at version 1 with no cached result (e.g. after a restart) can't prove the
    # update applied — keep the strict 409.
    trainer.slots["r1"] = SlotState("r1", "ttt-r1", idx=0, version=1)
    trainer.free_idxs = {1}
    with pytest.raises(ValueError, match="expected seq_no 2"):
        trainer.validate_job(job("r1", seq_no=1))


def test_slot_and_release_identity_remain_bound():
    trainer = make_registry(max_slots=2)
    trainer._claim("r1", "ttt-r1")
    with pytest.raises(ValueError, match="is bound to adapter"):
        trainer._claim("r1", "ttt-other")
    with pytest.raises(ValueError, match="already bound to rollout"):
        trainer._claim("r2", "ttt-r1")

    assert trainer.release("r1", "ttt-r1") is True
    # Idempotent retry: the state is gone, but the release still succeeds with False
    # (the server unloads from vLLM unconditionally either way); never-seen rollouts too.
    assert trainer.release("r1", "ttt-r1") is False
    assert trainer.release("r-unknown", "ttt-r-unknown") is False


def poisoning_forward(trainer, poisoned_span_start_fn):
    """A forward stub that mirrors make_cpu_trainer's graph-bearing fake but injects inf
    into one job's span (identified by its packed start offset)."""
    import torch

    dummy = torch.nn.Parameter(torch.zeros(1))
    trainer._optimizer = lambda state: torch.optim.SGD([dummy], lr=0.0)

    def forward(input_ids, position_ids, labels):
        out = dummy - torch.zeros_like(input_ids, dtype=torch.float32)
        start = poisoned_span_start_fn()
        if start is not None:
            poison = torch.zeros_like(out)
            poison[0, start] = float("inf")
            out = out + poison
        return out

    trainer._forward_logprobs = forward
    return dummy


def test_update_batch_isolates_non_finite_loss_per_job():
    """One poisoned job in a 2-job bin: it gets a per-job error, the healthy job trains
    and commits, and update_batch RETURNS (no FloatingPointError escaping to the work
    loop's os._exit fail-fast — one NaN rollout must not kill the service)."""
    trainer = make_cpu_trainer(max_slots=4)
    # r1 packs first (equal sizes keep insertion order); poison r1's span [0, 8).
    poisoning_forward(trainer, lambda: 0)
    results = trainer.update_batch([job("r1"), job("r2")])
    assert results["r1"]["error"] == "non-finite loss — rollout update rejected"
    assert results["r2"]["version"] == 1  # healthy job committed
    assert trainer.slots["r2"].version == 1
    assert trainer.slots["r1"].version == 0  # no commit: replay cache stays coherent


def test_update_batch_all_jobs_non_finite_skips_bin_entirely():
    import torch

    trainer = make_cpu_trainer(max_slots=4)
    dummy = torch.nn.Parameter(torch.zeros(1))
    trainer._optimizer = lambda state: torch.optim.SGD([dummy], lr=0.0)
    trainer._forward_logprobs = lambda input_ids, position_ids, labels: (
        dummy + torch.full_like(input_ids, float("nan"), dtype=torch.float32)
    )
    results = trainer.update_batch([job("r1"), job("r2")])
    assert results["r1"]["error"] == "non-finite loss — rollout update rejected"
    assert results["r2"]["error"] == "non-finite loss — rollout update rejected"
    assert trainer.slots["r1"].version == 0 and trainer.slots["r2"].version == 0


def test_update_batch_non_finite_checkpoint_is_per_job_error():
    trainer = make_cpu_trainer(max_slots=4)

    def save(state):
        if state.rollout_id == "r1":
            raise FloatingPointError("non-finite TTT v2 checkpoint tensors")
        return f"/fake/{state.rollout_id}/v{state.version}"

    trainer.save_checkpoint = save
    results = trainer.update_batch([job("r1"), job("r2")])
    assert results["r1"]["error"] == "non-finite adapter checkpoint — rollout update rejected"
    assert results["r2"]["version"] == 1  # the other job still commits
    assert trainer.slots["r1"].version == 0


def test_frozen_fused_head_backpropagates_without_weight_gradient(monkeypatch):
    import torch

    from prime_rl.trainer.models.layers.lm_head import FusedOutputLinear

    torch.manual_seed(0)
    model = torch.nn.Module()
    head = FusedOutputLinear(in_features=4, out_features=7, chunk_size=2)
    head.weight.requires_grad_(False)
    model.add_module("lm_head", head)
    assert _patch_frozen_fused_lm_head(model)

    hidden = torch.randn(2, 3, 4, requires_grad=True)
    labels = torch.tensor([[0, 3, 6], [2, 1, 4]])
    temperature = torch.ones_like(labels, dtype=torch.float32)
    output = head(hidden, labels=labels, temperature=temperature)

    original_zeros_like = torch.zeros_like

    def reject_weight_sized_allocation(tensor, *args, **kwargs):
        if tensor is head.weight:
            pytest.fail("backward allocated a gradient for the frozen vocabulary weight")
        return original_zeros_like(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros_like", reject_weight_sized_allocation)
    output["logprobs"].sum().backward()
    assert hidden.grad is not None
    assert head.weight.grad is None


def test_recv_weights_order_drains_batch_then_receives_on_every_rank():
    """The recv_weights work order runs strictly after the in-flight update batch (the
    work loop is sequential), calls the receive on the rank, bumps base_version, and
    leaves slot params/optimizer state untouched."""
    from queue import Queue
    from threading import Event

    from prime_rl.ttt.server_v2 import _work_loop

    trainer = make_cpu_trainer(max_slots=2)
    trainer.config.weight_broadcast = None  # stubbed receive below, config unused
    trainer.base_version = 0
    calls: list = []

    original_update_batch = trainer.update_batch

    def update_batch(jobs):
        calls.append(("update", [j.rollout_id for j in jobs]))
        return original_update_batch(jobs)

    def receive_base_weights(step):
        calls.append(("recv", step))
        trainer.base_version = step

    trainer.update_batch = update_batch
    trainer.receive_base_weights = receive_base_weights

    j1 = job("r1")
    from prime_rl.ttt.server_v2 import _Pending

    pending = _Pending(job=j1)
    ack = Event()
    work_queue: Queue = Queue()
    work_queue.put(("update", [j1], [pending]))
    work_queue.put(("recv_weights", 7, ack))
    work_queue.put(("stop",))
    _work_loop(trainer, work_queue, SimpleNamespace(is_master=True, world_size=1))

    assert calls == [("update", ["r1"]), ("recv", 7)]  # batch drained BEFORE the receive
    assert ack.is_set()
    assert trainer.base_version == 7
    # Slot state survives the receive: r1's slot, optimizer binding, and version.
    assert "r1" in trainer.slots and trainer.slots["r1"].version == 1

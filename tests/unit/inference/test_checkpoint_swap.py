"""Checkpoint swaps (``POST /swap-checkpoint``): the protocol a swap drives its DP group through."""

import asyncio
from types import SimpleNamespace

import pytest
from modelexpress_rl import WeightVersionState

from prime_rl.inference.vllm import checkpoint_swap
from prime_rl.inference.vllm.checkpoint_swap import CheckpointSwapper, SwapConfig, SwapFailed, SwapRequest
from prime_rl.inference.vllm.weight_pull import mesh_key
from prime_rl.transports.weights.checkpoint_versions import checkpoint_version_uid

CONFIG = SwapConfig(
    host="mx", port=1, model_name="model", staging_buffer_bytes=None, staging_buffers_count=1, timeout=60
)
TABLES = "0123456789abcdef"


def version(step: int, created: int = 0, mesh: str = "publisher-a", tables: str = TABLES, state=None):
    return SimpleNamespace(
        version_id=checkpoint_version_uid(step, tables),
        model_name="model",
        version_number=step,
        created_at_unix_ms=created,
        state=state or WeightVersionState.READY,
        trainer_mesh_id=mesh,
        trainer_mesh_generation=1,
    )


class FakeEngines:
    """A DP group: records the protocol; workers stage live only for a mesh they installed before."""

    def __init__(self, in_flight: int = 0):
        self.calls = []
        self.meshes: set[str] = set()
        self.paused = False
        self.admitting = True
        self.healthy = True
        self.served_ = (None, None)
        self.in_flight_ = in_flight
        self.tables = TABLES
        self.fail_stage = 0
        self.fail_apply = 0
        self.stage_gate: asyncio.Event | None = None

    async def init(self, payload):
        pass

    async def served(self):
        return self.served_

    async def integer_tables(self):
        return self.tables

    async def can_stage_live(self, mesh):
        return mesh in self.meshes

    async def stage(self, version_uid, live):
        assert live != self.paused, "a live stage runs while serving, any other while paused"
        self.calls.append(("stage", live))
        if self.stage_gate is not None:
            await self.stage_gate.wait()
        if self.fail_stage:
            self.fail_stage -= 1
            raise RuntimeError("stage failed")

    async def discard(self):
        self.calls.append("discard")

    async def hold(self):
        self.admitting = False
        self.calls.append("hold")
        return self.in_flight_

    async def release(self):
        self.admitting = True
        self.calls.append("release")

    async def in_flight(self):
        return self.in_flight_

    async def abort(self):
        aborted, self.in_flight_ = self.in_flight_, 0
        self.calls.append("abort")
        return aborted

    async def pause(self):
        assert not self.admitting and self.in_flight_ == 0, "pauses only with nothing in flight"
        self.paused = True
        self.calls.append("pause")

    async def apply(self, version_uid, mesh, step):
        assert self.paused
        self.calls.append("apply")
        if self.fail_apply:
            self.fail_apply -= 1
            self.served_ = (None, None)
            raise RuntimeError("install failed")
        self.meshes.add(mesh)
        self.served_ = (step, version_uid)

    async def reset_prefix_cache(self):
        assert self.paused and self.in_flight_ == 0
        self.calls.append("reset")

    async def set_health(self, healthy, reason=None):
        self.healthy = healthy
        self.calls.append(("health", healthy))

    async def resume(self):
        self.paused = False
        self.calls.append("resume")


def swapper(engines, versions):
    swap = CheckpointSwapper(engines, CONFIG)
    by_uid = {v.version_id: v for v in versions}
    swap._control = SimpleNamespace(
        list_weight_versions=lambda model_name: list(versions), get_weight_version=lambda uid: by_uid[uid]
    )
    return swap


def run(coroutine):
    return asyncio.run(coroutine)


def test_a_first_swap_stages_paused_a_later_one_while_serving_and_a_repeat_changes_nothing():
    engines = FakeEngines()
    first, second = version(2), version(3)
    swap = swapper(engines, [first, second])
    result = run(swap.swap(SwapRequest(step=2)))
    assert result["cold"] and not result["unchanged"] and result["timings"]["stage_paused"] >= 0
    assert engines.calls == ["hold", "pause", ("stage", False), "apply", "reset", "resume", "release"]
    assert engines.served_ == (2, first.version_id)

    engines.calls.clear()
    result = run(swap.swap(SwapRequest(step=3)))
    assert not result["cold"]
    assert engines.calls == [("stage", True), "hold", "pause", "apply", "reset", "resume", "release"]

    engines.calls.clear()
    assert run(swap.swap(SwapRequest(version_uid=second.version_id))) == {
        "step": 3,
        "version_uid": second.version_id,
        "unchanged": True,
    }
    assert engines.calls == []


def test_the_newest_ready_version_of_a_step_is_the_target():
    versions = [
        version(5, created=10),
        version(5, created=30),
        version(5, created=40, state=WeightVersionState.RELEASING),
    ]
    assert run(swapper(FakeEngines(), versions).resolve(SwapRequest(step=5))) is versions[1]
    with pytest.raises(SwapFailed) as failure:
        run(swapper(FakeEngines(), versions).resolve(SwapRequest(step=6)))
    assert failure.value.status == 404


@pytest.mark.parametrize("mode", ["fail", "drain", "abort"])
def test_requests_in_flight_by_mode(mode, monkeypatch):
    engines = FakeEngines(in_flight=2)
    engines.meshes.add(mesh_key(version(2)))
    swap = swapper(engines, [version(2)])

    async def finishes(seconds):
        engines.in_flight_ = 0

    monkeypatch.setattr(checkpoint_swap.asyncio, "sleep", finishes)
    if mode == "fail":
        with pytest.raises(SwapFailed) as failure:
            run(swap.swap(SwapRequest(step=2, mode=mode)))
        assert failure.value.status == 409 and failure.value.body["in_flight"] == 2
        assert engines.calls == [("stage", True), "hold", "release", "discard"]
        return
    result = run(swap.swap(SwapRequest(step=2, mode=mode)))
    assert result["in_flight"] == 2 and result["aborted"] == (2 if mode == "abort" else 0)
    assert ("abort" in engines.calls) == (mode == "abort")
    assert engines.calls[-4:] == ["apply", "reset", "resume", "release"]


def test_a_drain_that_times_out_serves_on_unchanged():
    engines = FakeEngines(in_flight=1)
    engines.meshes.add(mesh_key(version(2)))
    swap = swapper(engines, [version(2)])
    with pytest.raises(SwapFailed) as failure:
        run(swap.swap(SwapRequest(step=2, mode="drain", drain_timeout_s=0.1)))
    assert failure.value.status == 409
    assert engines.calls == [("stage", True), "hold", "release", "discard"] and engines.served_ == (None, None)


def test_a_failed_stage_never_pauses_and_a_failed_cold_stage_resumes_on_the_old_weights():
    engines = FakeEngines()
    engines.fail_stage = 1
    swap = swapper(engines, [version(2)])
    with pytest.raises(SwapFailed) as failure:
        run(swap.swap(SwapRequest(step=2)))
    assert failure.value.status == 502
    assert engines.calls == ["hold", "pause", ("stage", False), "discard", "resume", "release"]
    assert engines.healthy and swap.broken is None

    engines.calls.clear()
    engines.meshes.add(mesh_key(version(2)))
    engines.fail_stage = 1
    with pytest.raises(SwapFailed):
        run(swap.swap(SwapRequest(step=2)))
    assert engines.calls == [("stage", True), "discard"]


def test_other_integer_tables_are_refused_before_anything_changes():
    engines = FakeEngines()
    swap = swapper(engines, [version(2, tables="ffffffffffffffff")])
    with pytest.raises(SwapFailed) as failure:
        run(swap.swap(SwapRequest(step=2)))
    assert failure.value.status == 409 and "restart" in failure.value.body["error"]
    assert engines.calls == []


def test_one_swap_at_a_time_a_duplicate_shares_its_result_and_another_target_conflicts():
    engines = FakeEngines()
    target, other = version(2), version(3)
    swap = swapper(engines, [target, other])

    async def scenario():
        engines.stage_gate = asyncio.Event()
        first = asyncio.create_task(swap.swap(SwapRequest(step=2)))
        await asyncio.sleep(0.01)
        duplicate = asyncio.create_task(swap.swap(SwapRequest(version_uid=target.version_id)))
        await asyncio.sleep(0.01)
        with pytest.raises(SwapFailed) as conflict:
            await swap.swap(SwapRequest(step=3))
        engines.stage_gate.set()
        return await first, await duplicate, conflict.value

    first, duplicate, conflict = run(scenario())
    assert first == duplicate and conflict.status == 409
    assert engines.calls.count("apply") == 1


def test_a_failed_install_stays_paused_and_unhealthy_until_a_swap_installs(monkeypatch):
    engines = FakeEngines()
    engines.fail_apply = 2
    swap = swapper(engines, [version(2), version(3)])
    monkeypatch.setattr(checkpoint_swap, "MAX_REPAIR_INTERVAL_S", 0.01)
    swap.config = SwapConfig(**{**CONFIG.__dict__, "timeout": 0})

    async def scenario():
        with pytest.raises(SwapFailed) as failure:
            await swap.swap(SwapRequest(step=2))
        assert failure.value.status == 500
        assert engines.paused and not engines.admitting and not engines.healthy and swap.broken
        await swap._repair
        # the only retry also failed and the timeout passed: still paused, unhealthy
        assert engines.paused and not engines.healthy and engines.served_ == (None, None)

        # a swap whose stage fails leaves a broken group paused: its weights are still partly written
        engines.calls.clear()
        engines.fail_stage = 1
        with pytest.raises(SwapFailed) as failure:
            await swap.swap(SwapRequest(step=3))
        assert failure.value.status == 500 and engines.paused and not engines.admitting and not engines.healthy
        assert "resume" not in engines.calls and swap.broken

        # a new swap takes over from the paused group, without pausing again
        engines.calls.clear()
        return await swap.swap(SwapRequest(step=3))

    result = run(scenario())
    assert engines.calls == [("stage", False), "apply", "reset", "resume", "release", ("health", True)]
    assert result["cold"] and engines.healthy and not engines.paused and swap.broken is None


def test_versions_convert_to_the_checkpoint_layout_only_from_prime_names():
    """A trainer publishes PrimeRL-native tensors; a checkpoint publisher publishes the checkpoint's own."""
    import torch

    from prime_rl.inference.vllm.worker.modelexpress import source_to_hf_conversion
    from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime
    from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM

    config = DeepseekV4Config(
        num_hidden_layers=1, layer_types=["sliding_attention"], n_routed_experts=2, num_hash_layers=0
    )
    checkpoint = {
        name: torch.zeros(2)
        for name in (
            "embed.weight",
            "head.weight",
            "norm.weight",
            "hc_head_fn",
            "layers.0.attn.wq_a.weight",
            "layers.0.ffn.gate.weight",
            "layers.0.ffn.shared_experts.w1.weight",
            *(f"layers.0.ffn.experts.{e}.w{w}.weight" for e in range(2) for w in (1, 2, 3)),
        )
    }
    convert = source_to_hf_conversion(config)
    native = apply_hf_to_prime(dict(checkpoint), DeepseekV4ForCausalLM.conversion_chain(config))
    assert "model.layers.0.mlp.router.gate.weight" in native and "model.layers.0.mlp.experts.gate_proj" in native
    assert set(convert(dict(native))) == set(checkpoint)
    assert convert(dict(checkpoint)).keys() == checkpoint.keys()

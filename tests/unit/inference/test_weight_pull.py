import asyncio
from types import SimpleNamespace

import pytest
from modelexpress_rl import WeightPayloadFormat, WeightVersion, WeightVersionState

from prime_rl.inference.vllm import weight_pull


def version(
    number: int,
    state: WeightVersionState = WeightVersionState.READY,
    mesh: str = "mesh-a",
    published_at: int | None = None,
) -> WeightVersion:
    return WeightVersion(
        version_id=f"{mesh}-v{number}",
        model_name="model",
        payload_format=WeightPayloadFormat.FULL_TENSOR,
        layout_signature="",
        state=state,
        created_at_unix_ms=number if published_at is None else published_at,
        trainer_mesh_generation=1,
        trainer_mesh_id=mesh,
        version_number=number,
    )


class FakeServer:
    """A server of two engines with two workers each; only the last worker of the second engine fails to stage,
    which vLLM's ``collective_rpc`` alone would not report."""

    def __init__(self, stage_errors: dict[str, str], apply_error: Exception | None = None):
        self.stage_errors = stage_errors
        self.apply_error = apply_error
        self.calls = []
        self.engine_core = SimpleNamespace(call_utility_all_async=self.call_utility_all_async)

    async def call_utility_all_async(self, utility, method, timeout, args, kwargs):
        assert utility == "collective_rpc"
        self.calls.append((method, *args))
        if method == "stage_weights_from_modelexpress":
            return [[None, None], [None, self.stage_errors.get(args[0])]]
        return [[None, None], [None, None]]

    async def collective_rpc(self, method, args=(), kwargs=None):
        self.calls.append((method, *args))
        if self.apply_error is not None:
            raise self.apply_error

    async def pause_generation(self, *, mode, clear_cache):
        self.calls.append(("pause", mode, clear_cache))

    async def update_weight_version(self, label):
        self.calls.append(("label", label))

    async def resume_generation(self):
        self.calls.append(("resume",))


def test_installs_newest_version_a_source_can_serve():
    engine = FakeServer(stage_errors={"mesh-a-v3": "no source has a free read slot"})
    control = SimpleNamespace(
        list_weight_versions=lambda model_name: [
            version(1, WeightVersionState.RELEASING),
            version(2),
            version(3),
        ]
    )

    installed = asyncio.run(weight_pull._install_newest(engine, control, "model", serving=None))

    assert installed == version(2)
    assert engine.calls == [
        ("stage_weights_from_modelexpress", "mesh-a-v3"),
        ("discard_staged_modelexpress",),
        ("stage_weights_from_modelexpress", "mesh-a-v2"),
        ("pause", "keep", True),
        ("update_weights_from_modelexpress", "mesh-a-v2"),
        ("label", "2"),
        ("resume",),
    ]


def test_failed_install_leaves_generation_paused():
    engine = FakeServer(stage_errors={}, apply_error=RuntimeError("source died mid-copy"))

    with pytest.raises(RuntimeError, match="source died mid-copy"):
        asyncio.run(weight_pull._apply(engine, version(2)))

    assert engine.calls == [("pause", "keep", True), ("update_weights_from_modelexpress", "mesh-a-v2")]


def test_versions_of_a_restarted_trainer_are_newer_whatever_their_number():
    restarted = version(2, mesh="mesh-b", published_at=7)
    control = SimpleNamespace(list_weight_versions=lambda model_name: [version(4), version(6), restarted])

    newer = asyncio.run(weight_pull._newer_versions(control, "model", serving=version(5)))

    assert newer == [restarted, version(6)]

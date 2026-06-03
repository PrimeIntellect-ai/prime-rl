"""Unit tests for ``NIXLMxV2WeightBroadcast``.

These tests exercise the per-step orchestration logic — slot fill,
publisher add_tensor threading, compile_target tagging, MoE expert
metadata threading, and HSDP barrier gating — without requiring CUDA,
NIXL, a live MX server, or a real model.

We use ``importlib.util.spec_from_file_location`` to load the production
``nixl_mx_v2.py`` against a fully-stubbed dependency graph (same pattern
as MX-side ``test_vllm_weight_transfer.py``). The test is therefore
runnable anywhere torch + pytest is present, without prime-rl needing to
be installed as a package.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_PRIME_RL_ROOT = Path(__file__).resolve().parents[4]  # prime-rl root
_BROADCAST_FILE = (
    _PRIME_RL_ROOT
    / "src"
    / "prime_rl"
    / "trainer"
    / "rl"
    / "broadcast"
    / "nixl_mx_v2.py"
)


# ----------------------------------------------------------------------------
# Stub the prime_rl + modelexpress + transformers + torch.distributed
# dependency graph the broadcast module needs at import time
# ----------------------------------------------------------------------------


def _install_stubs():
    """Insert fake modules into sys.modules so importing nixl_mx_v2 succeeds.
    Returns a dict of the live mocks for test inspection."""
    mocks: dict[str, MagicMock] = {}

    # ─── modelexpress.nemo_rl_v2 ────────────────────────────────────────
    fake_publisher_cls = MagicMock(name="MxV2TrainingPublisher_cls")
    fake_publisher = MagicMock(name="MxV2TrainingPublisher_inst")
    fake_publisher.publish.return_value = "abcd1234efgh5678"
    fake_publisher.mark_ready.return_value = True
    fake_publisher_cls.return_value = fake_publisher

    fake_layout_cls = MagicMock(name="TrainerWorldLayout_cls")
    fake_layout = MagicMock(name="TrainerWorldLayout_inst")
    fake_layout.encode.return_value = "fsdp:1,tp:1,pp:1,ep:1"
    fake_layout_cls.return_value = fake_layout

    sys.modules["modelexpress"] = types.SimpleNamespace()
    sys.modules["modelexpress.nemo_rl_v2"] = types.SimpleNamespace(
        MxV2TrainingPublisher=fake_publisher_cls,
        TrainerWorldLayout=fake_layout_cls,
    )
    mocks["publisher_cls"] = fake_publisher_cls
    mocks["publisher"] = fake_publisher
    mocks["layout_cls"] = fake_layout_cls
    mocks["layout"] = fake_layout

    # ─── transformers.AutoConfig ────────────────────────────────────────
    fake_auto_config = MagicMock(name="AutoConfig")
    fake_auto_config.from_pretrained.return_value = MagicMock(
        torch_dtype="torch.bfloat16"
    )
    sys.modules["transformers"] = types.SimpleNamespace(AutoConfig=fake_auto_config)
    mocks["auto_config"] = fake_auto_config

    # ─── prime_rl.configs.trainer.MxV2WeightBroadcastConfig ─────────────
    fake_config_cls = MagicMock(name="MxV2WeightBroadcastConfig_cls")
    pkg_configs = types.ModuleType("prime_rl.configs")
    pkg_configs_trainer = types.ModuleType("prime_rl.configs.trainer")
    pkg_configs_trainer.MxV2WeightBroadcastConfig = fake_config_cls
    sys.modules.setdefault("prime_rl", types.ModuleType("prime_rl"))
    sys.modules["prime_rl.configs"] = pkg_configs
    sys.modules["prime_rl.configs.trainer"] = pkg_configs_trainer

    # ─── prime_rl.trainer.models.PreTrainedModelPrimeRL ─────────────────
    fake_pretrained_cls = MagicMock(name="PreTrainedModelPrimeRL_cls")
    pkg_trainer_models = types.ModuleType("prime_rl.trainer.models")
    pkg_trainer_models.PreTrainedModelPrimeRL = fake_pretrained_cls
    sys.modules["prime_rl.trainer"] = types.ModuleType("prime_rl.trainer")
    sys.modules["prime_rl.trainer.models"] = pkg_trainer_models

    # ─── prime_rl.trainer.models.conversions.select_default_conversion ──
    fake_conversion = types.SimpleNamespace(
        compile_target="cutlass_fp8",
        compile_metadata={"block_size": 128, "scale_layout": "per_channel"},
    )
    fake_select_conversion = MagicMock(return_value=fake_conversion)
    pkg_conv = types.ModuleType("prime_rl.trainer.models.conversions")
    pkg_conv.select_default_conversion = fake_select_conversion
    sys.modules["prime_rl.trainer.models.conversions"] = pkg_conv
    mocks["conversion"] = fake_conversion
    mocks["select_conversion"] = fake_select_conversion

    # ─── prime_rl.trainer.parallel_dims.ParallelDims ────────────────────
    fake_parallel_dims_cls = MagicMock(name="ParallelDims_cls")
    pkg_pd = types.ModuleType("prime_rl.trainer.parallel_dims")
    pkg_pd.ParallelDims = fake_parallel_dims_cls
    sys.modules["prime_rl.trainer.parallel_dims"] = pkg_pd

    # ─── prime_rl.trainer.rl.broadcast.base.WeightBroadcast ─────────────
    class FakeWeightBroadcast:
        def __init__(self, output_dir, *args, **kwargs):
            self.output_dir = output_dir
            # Mimic real base class — set logger so subclass can use it.
            self.logger = MagicMock(name="logger")

    pkg_trainer_rl = types.ModuleType("prime_rl.trainer.rl")
    pkg_trainer_rl_broadcast = types.ModuleType("prime_rl.trainer.rl.broadcast")
    pkg_broadcast_base = types.ModuleType("prime_rl.trainer.rl.broadcast.base")
    pkg_broadcast_base.WeightBroadcast = FakeWeightBroadcast
    sys.modules["prime_rl.trainer.rl"] = pkg_trainer_rl
    sys.modules["prime_rl.trainer.rl.broadcast"] = pkg_trainer_rl_broadcast
    sys.modules["prime_rl.trainer.rl.broadcast.base"] = pkg_broadcast_base
    mocks["base_cls"] = FakeWeightBroadcast

    # ─── prime_rl.trainer.runs.get_multi_run_manager ────────────────────
    fake_run_manager = types.SimpleNamespace(used_idxs=[], ready_to_update={})
    fake_get_multi_run_manager = MagicMock(return_value=fake_run_manager)
    pkg_runs = types.ModuleType("prime_rl.trainer.runs")
    pkg_runs.get_multi_run_manager = fake_get_multi_run_manager
    sys.modules["prime_rl.trainer.runs"] = pkg_runs
    mocks["run_manager"] = fake_run_manager

    # ─── prime_rl.trainer.utils.get_world ──────────────────────────────
    fake_world = types.SimpleNamespace(rank=0, is_master=True)
    fake_get_world = MagicMock(return_value=fake_world)
    pkg_utils = types.ModuleType("prime_rl.trainer.utils")
    pkg_utils.get_world = fake_get_world
    sys.modules["prime_rl.trainer.utils"] = pkg_utils
    mocks["world"] = fake_world

    # ─── prime_rl.transport.classic_cuda_pool / nixl_agent ──────────────
    class FakeAlloc:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    pkg_transport = types.ModuleType("prime_rl.transport")
    pkg_transport_classic = types.ModuleType("prime_rl.transport.classic_cuda_pool")
    pkg_transport_classic.classic_cuda_alloc = lambda: FakeAlloc()
    pkg_transport_nixl_agent = types.ModuleType("prime_rl.transport.nixl_agent")
    pkg_transport_nixl_agent.make_agent_name = MagicMock(return_value="trainer-0")
    pkg_transport_nixl_agent.pin_ucx_rail = MagicMock()
    sys.modules["prime_rl.transport"] = pkg_transport
    sys.modules["prime_rl.transport.classic_cuda_pool"] = pkg_transport_classic
    sys.modules["prime_rl.transport.nixl_agent"] = pkg_transport_nixl_agent

    return mocks


@pytest.fixture
def broadcast_mod():
    """Load nixl_mx_v2.py under fully-stubbed deps. Yields (module, mocks)."""
    # Wipe any stale modules so each test gets a fresh patched graph.
    for k in list(sys.modules.keys()):
        if k.startswith("prime_rl") or k == "modelexpress" or k.startswith("modelexpress."):
            del sys.modules[k]
        if k == "transformers":
            del sys.modules[k]

    mocks = _install_stubs()

    # Patch torch.cuda + torch.distributed before loading the module.
    import torch
    torch.cuda.current_device = MagicMock(return_value=0)
    if hasattr(torch.distributed, "barrier"):
        original_barrier = torch.distributed.barrier
        torch.distributed.barrier = MagicMock()
    else:
        original_barrier = None
        torch.distributed.barrier = MagicMock()

    spec = importlib.util.spec_from_file_location(
        "_test_nixl_mx_v2_under_test", _BROADCAST_FILE
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    try:
        yield (mod, mocks)
    finally:
        if original_barrier is not None:
            torch.distributed.barrier = original_barrier


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _make_config(**overrides):
    defaults = dict(
        type="mx_v2",
        host="localhost",
        port=8001,
        timeout=60,
        inference_world_size=1,
        inference_model_name="bench/synthetic-1.5B",
        same_rank_only=True,
        dedup_freshest_per_rank=True,
        publish_compile_target=True,
        compile_target_filter=None,
        publish_self_as_replica=True,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_parallel_dims(
    *,
    dp_replicate_enabled: bool = False,
    is_primary: bool = True,
    fsdp_world_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    ep_size: int = 1,
):
    mesh = MagicMock(name="dp_replicate_mesh")
    mesh.get_local_rank.return_value = 0 if is_primary else 1
    pdims = MagicMock(name="ParallelDims")
    pdims.dp_replicate_enabled = dp_replicate_enabled
    pdims.dp_shard_size = fsdp_world_size
    pdims.tp_size = tp_size
    pdims.pp_size = pp_size
    pdims.ep_size = ep_size
    pdims.get_mesh = MagicMock(return_value=mesh)
    return pdims


def _make_fake_slot(*, name: str, is_expert: bool = False, num_buffers: int = 2):
    import torch

    slot = MagicMock(name=f"Slot({name})")
    slot.is_expert = is_expert
    slot.expert_axis = 0 if is_expert else 0
    slot.owned_expert_ids = (0, 1, 2, 3) if is_expert else ()
    slot.buffers = [
        (f"{name}.buf_{i}", torch.zeros(4), object()) for i in range(num_buffers)
    ]
    slot.convert = MagicMock()
    return slot


def _make_fake_model(slots):
    model = MagicMock(name="Model")
    model.build_slots = MagicMock(return_value=slots)
    model.state_dict = MagicMock(return_value={})
    return model


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_construction_does_not_initialize_publisher(broadcast_mod):
    mod, mocks = broadcast_mod
    bc = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(),
        parallel_dims=_make_parallel_dims(),
    )
    assert bc.is_initialized is False
    assert bc._publisher is None
    assert bc._model_slots is None
    mocks["publisher_cls"].assert_not_called()


def test_is_primary_hsdp_rank_gates_correctly(broadcast_mod):
    mod, _ = broadcast_mod
    bc1 = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(),
        parallel_dims=_make_parallel_dims(dp_replicate_enabled=False),
    )
    assert bc1.is_primary_hsdp_rank is True

    bc2 = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(),
        parallel_dims=_make_parallel_dims(
            dp_replicate_enabled=True, is_primary=True
        ),
    )
    assert bc2.is_primary_hsdp_rank is True

    bc3 = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(),
        parallel_dims=_make_parallel_dims(
            dp_replicate_enabled=True, is_primary=False
        ),
    )
    assert bc3.is_primary_hsdp_rank is False


def test_lazy_init_builds_publisher_with_right_args(broadcast_mod):
    mod, mocks = broadcast_mod
    bc = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(host="mx-server", port=8001),
        parallel_dims=_make_parallel_dims(
            fsdp_world_size=4, tp_size=2, pp_size=1, ep_size=2
        ),
    )
    model = _make_fake_model([_make_fake_slot(name="layer0")])
    bc.lazy_init(model)

    mocks["layout_cls"].assert_called_once()
    layout_kwargs = mocks["layout_cls"].call_args.kwargs
    assert layout_kwargs["fsdp_world_size"] == 4
    assert layout_kwargs["tp_world_size"] == 2
    assert layout_kwargs["pp_world_size"] == 1
    assert layout_kwargs["ep_world_size"] == 2

    mocks["publisher_cls"].assert_called_once()
    pub_kwargs = mocks["publisher_cls"].call_args.kwargs
    assert pub_kwargs["mx_server_url"] == "mx-server:8001"
    assert pub_kwargs["world_layout"] is mocks["layout"]

    mocks["publisher"].initialize.assert_called_once()
    init_kwargs = mocks["publisher"].initialize.call_args.kwargs
    assert init_kwargs["model_name"] == "bench/synthetic-1.5B"

    assert bc.is_initialized is True


def test_lazy_init_idempotent_on_second_call(broadcast_mod):
    mod, mocks = broadcast_mod
    bc = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(),
        parallel_dims=_make_parallel_dims(),
    )
    model = _make_fake_model([_make_fake_slot(name="layer0")])
    bc.lazy_init(model)
    bc.lazy_init(model)
    assert mocks["publisher_cls"].call_count == 1


def test_broadcast_weights_threads_compile_target_metadata(broadcast_mod):
    mod, mocks = broadcast_mod
    bc = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(publish_compile_target=True),
        parallel_dims=_make_parallel_dims(),
    )
    slots = [_make_fake_slot(name="layer0", num_buffers=2)]
    model = _make_fake_model(slots)
    bc.broadcast_weights(model, step=42)

    assert mocks["publisher"].add_tensor.call_count == 2
    for call in mocks["publisher"].add_tensor.call_args_list:
        assert call.kwargs["compile_target"] == "cutlass_fp8"
        assert call.kwargs["compile_metadata"] == {
            "block_size": 128,
            "scale_layout": "per_channel",
        }

    mocks["publisher"].publish.assert_called_once()
    assert mocks["publisher"].publish.call_args.kwargs["version"] == 42
    mocks["publisher"].mark_ready.assert_called_once()


def test_broadcast_weights_publish_compile_target_false_uses_hf_raw(broadcast_mod):
    mod, mocks = broadcast_mod
    bc = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(publish_compile_target=False),
        parallel_dims=_make_parallel_dims(),
    )
    slots = [_make_fake_slot(name="layer0", num_buffers=1)]
    model = _make_fake_model(slots)
    bc.broadcast_weights(model, step=1)

    call = mocks["publisher"].add_tensor.call_args
    assert call.kwargs["compile_target"] == "hf_raw"
    assert call.kwargs["compile_metadata"] is None


def test_broadcast_weights_threads_moe_expert_metadata(broadcast_mod):
    mod, mocks = broadcast_mod
    bc = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(),
        parallel_dims=_make_parallel_dims(),
    )
    slots = [
        _make_fake_slot(name="layer0.dense", is_expert=False, num_buffers=1),
        _make_fake_slot(name="layer0.experts", is_expert=True, num_buffers=1),
    ]
    model = _make_fake_model(slots)
    bc.broadcast_weights(model, step=7)

    calls = mocks["publisher"].add_tensor.call_args_list
    assert len(calls) == 2

    dense_call = next(
        c for c in calls if c.kwargs["name"].startswith("layer0.dense")
    )
    assert dense_call.kwargs["is_expert"] is False
    assert dense_call.kwargs["owned_expert_ids"] == ()

    expert_call = next(
        c for c in calls if c.kwargs["name"].startswith("layer0.experts")
    )
    assert expert_call.kwargs["is_expert"] is True
    assert expert_call.kwargs["expert_axis"] == 0
    assert expert_call.kwargs["owned_expert_ids"] == (0, 1, 2, 3)


def test_broadcast_weights_skips_non_primary_hsdp_rank(broadcast_mod):
    mod, mocks = broadcast_mod
    bc = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(),
        parallel_dims=_make_parallel_dims(
            dp_replicate_enabled=True, is_primary=False
        ),
    )
    model = _make_fake_model([_make_fake_slot(name="layer0")])
    bc.broadcast_weights(model, step=1)

    mocks["publisher_cls"].assert_not_called()
    mocks["publisher"].add_tensor.assert_not_called()
    mocks["publisher"].publish.assert_not_called()


def test_broadcast_weights_calls_slot_convert(broadcast_mod):
    """Each slot's `convert(state_dict)` must be invoked exactly once per
    broadcast cycle. GatheredSlot's API takes only the state_dict — the
    conversion (compile_target / quantization) is baked in at
    `from_spec` creation time, not threaded per-call."""
    mod, _ = broadcast_mod
    bc = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(),
        parallel_dims=_make_parallel_dims(),
    )
    slots = [
        _make_fake_slot(name="layer0", num_buffers=1),
        _make_fake_slot(name="layer1", num_buffers=1),
    ]
    model = _make_fake_model(slots)
    bc.broadcast_weights(model, step=3)
    for slot in slots:
        slot.convert.assert_called_once()
        # convert receives the state_dict (single positional arg).
        args = slot.convert.call_args.args
        assert isinstance(args[0], dict)


def test_shutdown_calls_publisher_shutdown_idempotent(broadcast_mod):
    mod, mocks = broadcast_mod
    bc = mod.NIXLMxV2WeightBroadcast(
        output_dir=Path("/tmp/out"),
        config=_make_config(),
        parallel_dims=_make_parallel_dims(),
    )
    model = _make_fake_model([_make_fake_slot(name="layer0", num_buffers=1)])
    bc.broadcast_weights(model, step=1)

    bc.shutdown()
    assert mocks["publisher"].shutdown.call_count == 1
    bc.shutdown()
    assert mocks["publisher"].shutdown.call_count == 1
    assert bc.is_initialized is False

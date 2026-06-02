"""Unit tests for ``NIXLMxV2WeightUpdateWorker``.

Same pattern as ``test_nixl_mx_v2.py`` — load the production module via
``importlib.util.spec_from_file_location`` against a fully-stubbed
dependency graph (vLLM, modelexpress, prime_rl.transport, etc.), so the
test runs anywhere torch + pytest is present.

The worker has two RPC entry points:

- ``init_nixl_mx_v2(host, port, rank_offset, *, publish_self_as_replica, listen_port)``
- ``update_weights_via_mx_v2(step, *, compile_target_filter, timeout_seconds, same_rank_only)``

We verify init-info construction, update-info construction, the
``load_weights`` callback path, metrics-dict shape, and the post-load
``update_mla_absorbed_weights`` hook.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_PRIME_RL_ROOT = Path(__file__).resolve().parents[5]  # prime-rl root
_WORKER_FILE = (
    _PRIME_RL_ROOT
    / "src"
    / "prime_rl"
    / "inference"
    / "vllm"
    / "worker"
    / "nixl_mx_v2.py"
)


def _install_stubs():
    """Insert fake modules so the worker file imports cleanly."""
    mocks: dict[str, MagicMock] = {}

    # ─── modelexpress.vllm_weight_transfer ──────────────────────────────
    fake_engine_cls = MagicMock(name="MxWeightTransferEngine_cls")
    fake_engine = MagicMock(name="MxWeightTransferEngine_inst")
    fake_stats = types.SimpleNamespace(
        bytes_received=536_870_912,
        tensors_received=64,
        elapsed_seconds=0.082,
        bandwidth_gbps=52.4,
        discovery_seconds=0.014,
        source_worker_rank=0,
    )
    fake_engine.last_transfer_stats = fake_stats
    fake_engine.last_discovery_seconds = 0.014
    fake_engine_cls.return_value = fake_engine

    fake_init_info_cls = MagicMock(name="MxInitInfo")
    fake_update_info_cls = MagicMock(name="MxUpdateInfo")

    sys.modules["modelexpress"] = types.SimpleNamespace()
    sys.modules["modelexpress.vllm_weight_transfer"] = types.SimpleNamespace(
        MxWeightTransferEngine=fake_engine_cls,
        MxInitInfo=fake_init_info_cls,
        MxUpdateInfo=fake_update_info_cls,
    )
    mocks["engine_cls"] = fake_engine_cls
    mocks["engine"] = fake_engine
    mocks["init_info_cls"] = fake_init_info_cls
    mocks["update_info_cls"] = fake_update_info_cls
    mocks["stats"] = fake_stats

    # ─── vllm.logger ────────────────────────────────────────────────────
    sys.modules["vllm"] = types.SimpleNamespace()
    sys.modules["vllm.logger"] = types.SimpleNamespace(
        init_logger=lambda name: MagicMock(name=f"logger({name})")
    )
    # vllm.v1.worker.gpu_worker only used inside TYPE_CHECKING so no stub needed

    # ─── prime_rl.inference.vllm.worker.weight_transfer ────────────────
    fake_update_mla = MagicMock(name="update_mla_absorbed_weights")
    sys.modules.setdefault("prime_rl", types.ModuleType("prime_rl"))
    sys.modules["prime_rl.inference"] = types.ModuleType("prime_rl.inference")
    sys.modules["prime_rl.inference.vllm"] = types.ModuleType("prime_rl.inference.vllm")
    sys.modules["prime_rl.inference.vllm.worker"] = types.ModuleType(
        "prime_rl.inference.vllm.worker"
    )
    pkg_wt = types.ModuleType("prime_rl.inference.vllm.worker.weight_transfer")
    pkg_wt.update_mla_absorbed_weights = fake_update_mla
    # `build_expert_map` is imported by the OLD worker (nixl_mx.py) — not by
    # nixl_mx_v2 — so we don't need to stub it for this test. Add a no-op
    # in case the test imports the broadcast __init__ which may pull it in.
    pkg_wt.build_expert_map = MagicMock(name="build_expert_map", return_value={})
    sys.modules["prime_rl.inference.vllm.worker.weight_transfer"] = pkg_wt
    mocks["update_mla"] = fake_update_mla

    # ─── prime_rl.transport.nixl_agent ──────────────────────────────────
    fake_make_agent_name = MagicMock(return_value="vllm-inference-r0")
    fake_pin_ucx_rail = MagicMock()
    sys.modules["prime_rl.transport"] = types.ModuleType("prime_rl.transport")
    pkg_na = types.ModuleType("prime_rl.transport.nixl_agent")
    pkg_na.make_agent_name = fake_make_agent_name
    pkg_na.pin_ucx_rail = fake_pin_ucx_rail
    sys.modules["prime_rl.transport.nixl_agent"] = pkg_na
    mocks["make_agent_name"] = fake_make_agent_name
    mocks["pin_ucx_rail"] = fake_pin_ucx_rail

    return mocks


@pytest.fixture
def worker_mod():
    """Load nixl_mx_v2.py worker under fully-stubbed deps."""
    for k in list(sys.modules.keys()):
        if k.startswith("prime_rl") or k == "modelexpress" or k.startswith("modelexpress."):
            del sys.modules[k]
        if k == "vllm" or k.startswith("vllm."):
            del sys.modules[k]

    mocks = _install_stubs()

    import torch
    if not hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize = MagicMock()
    original_synchronize = torch.cuda.synchronize
    torch.cuda.synchronize = MagicMock()

    spec = importlib.util.spec_from_file_location(
        "_test_nixl_mx_v2_worker_under_test", _WORKER_FILE
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    try:
        yield (mod, mocks)
    finally:
        torch.cuda.synchronize = original_synchronize


def _make_worker(mod, *, model_name="bench/synthetic-1.5B", device_index=0):
    """Build a worker with a mocked vLLM Worker context.

    The `raw_model` property does `assert isinstance(model, Module)` so the
    inner model has to be a real ``torch.nn.Module`` subclass. We attach a
    `load_weights` method onto it so tests can spy on the callback path.
    """
    import torch.nn as nn

    class FakeInnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.load_weights = MagicMock(name="load_weights")

    worker = mod.NIXLMxV2WeightUpdateWorker()
    worker.device = MagicMock()
    worker.device.index = device_index
    runner = MagicMock(name="ModelRunner")
    runner.model_config = MagicMock(model=model_name)
    fake_inner = FakeInnerModel()
    runner.model = MagicMock()
    runner.model.runnable = fake_inner
    worker.model_runner = runner
    return worker


def test_init_nixl_mx_v2_builds_engine_with_correct_init_info(worker_mod):
    mod, mocks = worker_mod
    worker = _make_worker(mod, device_index=2)
    worker.init_nixl_mx_v2(
        host="modelexpress-server.kavin.svc.cluster.local",
        port=8001,
        rank_offset=4,
        publish_self_as_replica=True,
        listen_port=None,
    )

    mocks["init_info_cls"].assert_called_once()
    init_kwargs = mocks["init_info_cls"].call_args.kwargs
    assert (
        init_kwargs["mx_server_url"]
        == "modelexpress-server.kavin.svc.cluster.local:8001"
    )
    assert init_kwargs["worker_rank"] == 4 + 2
    assert init_kwargs["agent_name"] == "vllm-inference-r0"
    assert init_kwargs["device_id"] == 2
    assert init_kwargs["publish_self_as_replica"] is True

    mocks["engine_cls"].assert_called_once()
    eng_kwargs = mocks["engine_cls"].call_args.kwargs
    assert "init_info" in eng_kwargs
    mocks["pin_ucx_rail"].assert_called_once_with(2)
    assert worker._global_rank == 6


def test_init_nixl_mx_v2_respects_publish_self_as_replica_false(worker_mod):
    mod, mocks = worker_mod
    worker = _make_worker(mod)
    worker.init_nixl_mx_v2(
        host="x", port=8001, rank_offset=0, publish_self_as_replica=False
    )
    init_kwargs = mocks["init_info_cls"].call_args.kwargs
    assert init_kwargs["publish_self_as_replica"] is False


def test_update_weights_via_mx_v2_dispatches_engine_receive(worker_mod):
    mod, mocks = worker_mod
    worker = _make_worker(mod)
    worker.init_nixl_mx_v2(host="x", port=8001, rank_offset=0)

    metrics = worker.update_weights_via_mx_v2(
        42,
        compile_target_filter=["cutlass_fp8", "hf_raw"],
        timeout_seconds=180.0,
        same_rank_only=True,
    )

    mocks["update_info_cls"].assert_called_once()
    upd_kwargs = mocks["update_info_cls"].call_args.kwargs
    assert upd_kwargs["version"] == 42
    assert upd_kwargs["compile_target_filter"] == {"cutlass_fp8", "hf_raw"}
    assert upd_kwargs["timeout_seconds"] == 180.0
    assert upd_kwargs["same_rank_only"] is True
    assert upd_kwargs["target_tp_layout"] is None

    mocks["engine"].receive_weights.assert_called_once()
    call = mocks["engine"].receive_weights.call_args
    assert "load_weights" in call.kwargs

    mocks["update_mla"].assert_called_once()

    assert metrics["step"] == 42
    assert metrics["bytes_received"] == 536_870_912
    assert metrics["tensors_received"] == 64
    assert metrics["bandwidth_gbps"] == pytest.approx(52.4)
    assert metrics["discovery_seconds"] == pytest.approx(0.014)
    assert metrics["source_worker_rank"] == 0


def test_update_weights_via_mx_v2_no_filter_passes_none(worker_mod):
    mod, mocks = worker_mod
    worker = _make_worker(mod)
    worker.init_nixl_mx_v2(host="x", port=8001, rank_offset=0)
    worker.update_weights_via_mx_v2(1, compile_target_filter=None)
    upd_kwargs = mocks["update_info_cls"].call_args.kwargs
    assert upd_kwargs["compile_target_filter"] is None


def test_load_weights_batch_feeds_through_vllm_model_load_weights(worker_mod):
    mod, _ = worker_mod
    worker = _make_worker(mod)
    captured_batches = []
    worker.raw_model.load_weights = MagicMock(
        side_effect=lambda batch: captured_batches.append(batch)
    )
    batch_1 = [("model.layers.0.weight", "TENSOR1")]
    batch_2 = [("model.layers.1.weight", "TENSOR2"), ("a", "T3")]
    worker._load_weights_batch(batch_1)
    worker._load_weights_batch(batch_2)
    assert captured_batches == [batch_1, batch_2]


def test_update_weights_via_mx_v2_metrics_safe_when_stats_none(worker_mod):
    mod, mocks = worker_mod
    mocks["engine"].last_transfer_stats = None
    worker = _make_worker(mod)
    worker.init_nixl_mx_v2(host="x", port=8001, rank_offset=0)

    metrics = worker.update_weights_via_mx_v2(1)
    assert metrics["bytes_received"] == 0
    assert metrics["tensors_received"] == 0
    assert metrics["bandwidth_gbps"] == 0.0
    assert metrics["source_worker_rank"] is None

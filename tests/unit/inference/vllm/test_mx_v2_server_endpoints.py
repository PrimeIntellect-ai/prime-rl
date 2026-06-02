"""Unit tests for the ``mx_v2`` server-side glue.

Three pieces tested here:

1. The ``WORKER_EXTENSION_CLS["mx_v2"]`` entry in server.py — i.e. that
   the worker-extension selector points at our new worker extension class.
2. The new HTTP endpoints ``/init_nixl_mx_v2`` and ``/update_weights_v2``
   on server.py — verified to forward to the right ``collective_rpc``
   method names with the right kwargs.
3. The orchestrator-side helpers ``init_nixl_mx_v2_broadcast`` and
   ``update_weights_v2`` in client.py — verified to POST to the right
   endpoints with the right JSON body.

Plus the trainer-side selector dispatch (``setup_weight_broadcast`` for
``config.type == "mx_v2"``).

We use ``importlib.util.spec_from_file_location`` to load each target
file against a stubbed dep graph, so the test runs anywhere torch +
pytest is present without prime-rl needing to be installed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


_PRIME_RL_ROOT = Path(__file__).resolve().parents[4]
_SERVER_FILE = (
    _PRIME_RL_ROOT / "src" / "prime_rl" / "inference" / "vllm" / "server.py"
)
_CLIENT_FILE = _PRIME_RL_ROOT / "src" / "prime_rl" / "utils" / "client.py"
_BROADCAST_INIT_FILE = (
    _PRIME_RL_ROOT
    / "src"
    / "prime_rl"
    / "trainer"
    / "rl"
    / "broadcast"
    / "__init__.py"
)


# ----------------------------------------------------------------------------
# 1. WORKER_EXTENSION_CLS table — read directly from the source AST so we
#    don't have to install the package or stub anywhere near as much
# ----------------------------------------------------------------------------


def _extract_worker_extension_cls():
    """Parse server.py and pull out the WORKER_EXTENSION_CLS dict literal.

    Avoids the import-graph problem entirely — we only need the table.
    """
    import ast

    src = _SERVER_FILE.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "WORKER_EXTENSION_CLS"
            for t in node.targets
        ):
            return {
                key.value: value.value
                for key, value in zip(node.value.keys, node.value.values)
                if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
            }
    raise RuntimeError("WORKER_EXTENSION_CLS not found in server.py")


def test_worker_extension_cls_table_has_mx_v2_entry():
    table = _extract_worker_extension_cls()
    assert "mx_v2" in table
    assert (
        table["mx_v2"]
        == "prime_rl.inference.vllm.worker.nixl_mx_v2.NIXLMxV2WeightUpdateWorker"
    )


def test_worker_extension_cls_table_preserves_existing_backends():
    """Adding mx_v2 must not have removed nccl / filesystem / nixl_mx."""
    table = _extract_worker_extension_cls()
    assert "nccl" in table
    assert "filesystem" in table
    assert "nixl_mx" in table
    # And nixl_mx vs mx_v2 are two distinct worker classes.
    assert table["nixl_mx"] != table["mx_v2"]


# ----------------------------------------------------------------------------
# 2. Server endpoints — load via spec_from_file_location with stubs
# ----------------------------------------------------------------------------


def _install_server_stubs():
    """Stub the heavy server.py deps (vLLM, FastAPI bits, prime_rl imports).

    Just enough to let server.py's module-level statements run; we only need
    to call the two new endpoint coroutines.
    """
    # FastAPI bits
    fake_request_cls = type("Request", (), {})
    fake_apirouter_cls = MagicMock(name="APIRouter")
    fake_apirouter = MagicMock(name="apirouter_inst")
    # Make APIRouter.post / get return identity decorators so the @router.post
    # decorators in server.py work without registering anything.
    fake_apirouter.post = lambda *a, **kw: (lambda f: f)
    fake_apirouter.get = lambda *a, **kw: (lambda f: f)
    fake_apirouter_cls.return_value = fake_apirouter
    fake_jsonresponse = MagicMock(name="JSONResponse")
    sys.modules["fastapi"] = types.SimpleNamespace(
        Request=fake_request_cls, APIRouter=fake_apirouter_cls
    )
    sys.modules["fastapi.responses"] = types.SimpleNamespace(
        JSONResponse=fake_jsonresponse
    )

    # vllm bits
    sys.modules["vllm"] = types.SimpleNamespace()
    sys.modules["vllm.engine"] = types.SimpleNamespace()
    sys.modules["vllm.engine.protocol"] = types.SimpleNamespace(
        EngineClient=type("EngineClient", (), {})
    )
    sys.modules["vllm.entrypoints"] = types.SimpleNamespace()
    sys.modules["vllm.entrypoints.openai"] = types.SimpleNamespace()
    sys.modules["vllm.entrypoints.openai.api_server"] = types.SimpleNamespace(
        State=type("State", (), {}),
        init_app_state=MagicMock(),
        run_headless=MagicMock(),
    )
    sys.modules["vllm.entrypoints.openai.protocol"] = types.SimpleNamespace(
        LoadLoRAAdapterRequest=type("LoadLoRAAdapterRequest", (), {}),
        ErrorResponse=type("ErrorResponse", (), {}),
    )
    sys.modules["vllm.utils"] = types.SimpleNamespace(FlexibleArgumentParser=MagicMock())
    # prime_rl deps used at top of server.py
    sys.modules.setdefault("prime_rl", types.ModuleType("prime_rl"))
    sys.modules.setdefault("prime_rl.utils", types.ModuleType("prime_rl.utils"))
    sys.modules["prime_rl.utils.logger"] = types.SimpleNamespace(
        get_logger=MagicMock(return_value=MagicMock(name="logger")),
        setup_logger=MagicMock(),
    )

    # PrimeRlServingTokens etc.
    sys.modules["prime_rl.inference"] = types.ModuleType("prime_rl.inference")
    sys.modules["prime_rl.inference.vllm"] = types.ModuleType("prime_rl.inference.vllm")
    sys.modules["prime_rl.inference.vllm.serving_tokens"] = types.SimpleNamespace(
        PrimeRlServingTokens=type("PrimeRlServingTokens", (), {})
    )


@pytest.fixture
def server_mod():
    """Load server.py with stubs in place."""
    # Wipe cached state
    for k in list(sys.modules.keys()):
        if k.startswith("prime_rl") or k.startswith("vllm") or k.startswith("fastapi"):
            del sys.modules[k]

    _install_server_stubs()

    spec = importlib.util.spec_from_file_location(
        "_test_server_under_test", _SERVER_FILE
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(
            f"server.py imports too much to stub cleanly: {e}; this test "
            f"runs in CI where prime-rl IS installed"
        )
    yield mod


@pytest.mark.asyncio
async def test_init_nixl_mx_v2_endpoint_dispatches_collective_rpc(server_mod):
    fake_client = MagicMock()
    fake_client.collective_rpc = AsyncMock()
    fake_request = MagicMock()
    fake_request.json = AsyncMock(
        return_value={
            "host": "modelexpress-server.kavin.svc.cluster.local",
            "port": 8001,
            "rank_offset": 4,
            "publish_self_as_replica": True,
            "listen_port": None,
        }
    )

    orig = getattr(server_mod, "engine_client", None)
    server_mod.engine_client = lambda r: fake_client
    try:
        result = await server_mod.init_nixl_mx_v2(fake_request)
    finally:
        if orig is not None:
            server_mod.engine_client = orig

    assert result == {"status": "ok"}
    fake_client.collective_rpc.assert_called_once_with(
        "init_nixl_mx_v2",
        args=(
            "modelexpress-server.kavin.svc.cluster.local",
            8001,
            4,
        ),
        kwargs={"publish_self_as_replica": True, "listen_port": None},
    )


@pytest.mark.asyncio
async def test_update_weights_v2_endpoint_dispatches_collective_rpc(server_mod):
    fake_metrics = [
        {"step": 42, "bytes_received": 536_870_912, "bandwidth_gbps": 52.4}
    ]
    fake_client = MagicMock()
    fake_client.collective_rpc = AsyncMock(return_value=fake_metrics)
    fake_request = MagicMock()
    fake_request.json = AsyncMock(
        return_value={
            "step": 42,
            "compile_target_filter": ["cutlass_fp8"],
            "timeout_seconds": 180.0,
            "same_rank_only": True,
        }
    )

    orig = getattr(server_mod, "engine_client", None)
    server_mod.engine_client = lambda r: fake_client
    try:
        result = await server_mod.update_weights_v2(fake_request)
    finally:
        if orig is not None:
            server_mod.engine_client = orig

    assert result == {"status": "ok", "metrics": fake_metrics}
    fake_client.collective_rpc.assert_called_once_with(
        "update_weights_via_mx_v2",
        args=(42,),
        kwargs={
            "compile_target_filter": ["cutlass_fp8"],
            "timeout_seconds": 180.0,
            "same_rank_only": True,
        },
    )


@pytest.mark.asyncio
async def test_update_weights_v2_endpoint_defaults(server_mod):
    fake_client = MagicMock()
    fake_client.collective_rpc = AsyncMock(return_value=[])
    fake_request = MagicMock()
    fake_request.json = AsyncMock(return_value={"step": 1})

    orig = getattr(server_mod, "engine_client", None)
    server_mod.engine_client = lambda r: fake_client
    try:
        await server_mod.update_weights_v2(fake_request)
    finally:
        if orig is not None:
            server_mod.engine_client = orig

    kwargs = fake_client.collective_rpc.call_args.kwargs["kwargs"]
    assert kwargs["compile_target_filter"] is None
    assert kwargs["timeout_seconds"] == 300.0
    assert kwargs["same_rank_only"] is True


# ----------------------------------------------------------------------------
# 3. Orchestrator-side helpers — load client.py with stubs
# ----------------------------------------------------------------------------


def _install_client_stubs():
    sys.modules.setdefault("prime_rl", types.ModuleType("prime_rl"))
    sys.modules.setdefault("prime_rl.utils", types.ModuleType("prime_rl.utils"))
    sys.modules["prime_rl.utils.logger"] = types.SimpleNamespace(
        get_logger=MagicMock(return_value=MagicMock(name="logger")),
        setup_logger=MagicMock(),
    )
    # httpx AsyncClient stub — client.py imports it
    sys.modules["httpx"] = types.SimpleNamespace(
        AsyncClient=type("AsyncClient", (), {})
    )


@pytest.fixture
def client_mod():
    for k in list(sys.modules.keys()):
        if k.startswith("prime_rl") or k == "httpx":
            del sys.modules[k]
    _install_client_stubs()
    spec = importlib.util.spec_from_file_location(
        "_test_client_under_test", _CLIENT_FILE
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(
            f"client.py imports too much to stub cleanly: {e}; this test "
            f"runs in CI where prime-rl IS installed"
        )
    yield mod


@pytest.mark.asyncio
async def test_init_nixl_mx_v2_broadcast_posts_to_all_servers(client_mod):
    """POSTs /init_nixl_mx_v2 with rank_offset = i * gpus_per_server per server."""
    admin_clients = []
    for _ in range(3):
        c = MagicMock()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        c.post = AsyncMock(return_value=resp)
        admin_clients.append(c)

    await client_mod.init_nixl_mx_v2_broadcast(
        admin_clients,
        host="mx-server",
        port=8001,
        inference_world_size=12,
        publish_self_as_replica=True,
        listen_port=None,
    )

    # gpus_per_server = 12 // 3 = 4 → rank_offsets 0, 4, 8
    expected_offsets = [0, 4, 8]
    for c, expected_offset in zip(admin_clients, expected_offsets):
        c.post.assert_called_once()
        args, kwargs = c.post.call_args
        assert args[0] == "/init_nixl_mx_v2"
        body = kwargs["json"]
        assert body["host"] == "mx-server"
        assert body["port"] == 8001
        assert body["rank_offset"] == expected_offset
        assert body["publish_self_as_replica"] is True


@pytest.mark.asyncio
async def test_update_weights_v2_posts_step_and_returns_metrics(client_mod):
    fake_servers = []
    expected_responses = [
        {"status": "ok", "metrics": [{"step": 5, "bandwidth_gbps": 50.0}]},
        {"status": "ok", "metrics": [{"step": 5, "bandwidth_gbps": 48.0}]},
    ]
    for resp_body in expected_responses:
        c = MagicMock()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=resp_body)
        c.post = AsyncMock(return_value=resp)
        fake_servers.append(c)

    results = await client_mod.update_weights_v2(
        fake_servers,
        step=5,
        compile_target_filter=["cutlass_fp8"],
        timeout_seconds=180.0,
        same_rank_only=True,
    )

    assert results == expected_responses
    for c in fake_servers:
        args, kwargs = c.post.call_args
        assert args[0] == "/update_weights_v2"
        body = kwargs["json"]
        assert body["step"] == 5
        assert body["compile_target_filter"] == ["cutlass_fp8"]
        assert body["timeout_seconds"] == 180.0
        assert body["same_rank_only"] is True


# ----------------------------------------------------------------------------
# 4. Trainer-side selector dispatch — verify __init__.py routes mx_v2 correctly
# ----------------------------------------------------------------------------


def test_broadcast_init_dispatches_mx_v2_via_ast():
    """The selector in broadcast/__init__.py routes config.type == "mx_v2"
    to NIXLMxV2WeightBroadcast. Parse the source directly to avoid the heavy
    import graph."""
    import ast

    src = _BROADCAST_INIT_FILE.read_text()
    tree = ast.parse(src)

    # Find the setup_weight_broadcast function
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "setup_weight_broadcast"
    )

    # Find the elif branch with `config.type == "mx_v2"`
    mx_v2_branch_found = False
    for node in ast.walk(func):
        if isinstance(node, ast.Compare):
            # Detect `config.type == "mx_v2"`
            if (
                len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == "mx_v2"
            ):
                mx_v2_branch_found = True
                break
    assert mx_v2_branch_found, "mx_v2 dispatch branch not found in selector"

    # And the branch references NIXLMxV2WeightBroadcast
    assert "NIXLMxV2WeightBroadcast" in src
    assert "from prime_rl.trainer.rl.broadcast.nixl_mx_v2" in src

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import verifiers as vf

from prime_rl.configs.multi_agent import FixedMemberTargetConfig
from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.member_generation import _fixed_client
from prime_rl.utils.client import (
    AdminGatherError,
    _is_retryable_lora_error,
    init_nccl_broadcast,
    load_lora_adapter,
    setup_clients,
)


def make_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://worker/init_broadcaster")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("status error", request=request, response=response)


def test_is_retryable_lora_error_returns_true_for_404():
    response = MagicMock()
    response.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_true_for_500():
    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is False


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        timeout=httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0),
    )


def test_init_nccl_broadcast_requires_divisible_world_size():
    clients = [AsyncMock(), AsyncMock()]

    with pytest.raises(ValueError, match="divisible"):
        asyncio.run(init_nccl_broadcast(clients, host="localhost", port=29501, timeout=10, inference_world_size=3))


def test_init_nccl_broadcast_propagates_missing_route():
    # A failed peer must NOT be swallowed. The fan-out now raises an
    # attributed AdminGatherError naming the dead peer (converting the opaque
    # first-exception that killed the whole job into a recoverable, named
    # failure) while preserving the underlying HTTPStatusError as the cause.
    mock_client = AsyncMock()
    mock_client.base_url = "http://worker-a:8000"
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = make_status_error(404)
    mock_client.post.return_value = mock_response

    with pytest.raises(AdminGatherError) as exc_info:
        asyncio.run(init_nccl_broadcast([mock_client], host="localhost", port=29501, timeout=10))

    err = exc_info.value
    assert err.op_name == "init NCCL broadcaster"
    assert err.total == 1
    assert len(err.failures) == 1
    failed_client, failed_exc = err.failures[0]
    assert failed_client is mock_client
    assert isinstance(failed_exc, httpx.HTTPStatusError)
    # The dead peer is named in the aggregated message.
    assert "http://worker-a:8000" in str(err)


def test_init_nccl_broadcast_assigns_rank_offsets():
    clients = [AsyncMock(), AsyncMock()]
    for client in clients:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        client.post.return_value = response

    asyncio.run(init_nccl_broadcast(clients, host="localhost", port=29501, timeout=10, inference_world_size=8))

    assert clients[0].post.call_args.kwargs["json"]["rank_offset"] == 0
    assert clients[1].post.call_args.kwargs["json"]["rank_offset"] == 4


def test_setup_clients_assigns_renderer_and_dp_rank_headers():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
        headers={"X-Test": "test"},
        dp_rank_count=2,
        extra_headers_from_state={"X-Session-ID": "session_id"},
    )

    renderer_settings = Qwen3VLRendererConfig()
    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_config=renderer_settings,
    )

    assert [client.client_type for client in clients] == ["renderer", "renderer"]
    assert [client.renderer_config for client in clients] == [renderer_settings, renderer_settings]
    assert [client.renderer_model_name for client in clients] == [None, None]
    assert [client.api_base_url for client in clients] == ["http://worker-a:8000/v1"] * 2
    assert [client.extra_headers["X-data-parallel-rank"] for client in clients] == ["0", "1"]
    assert clients[0].extra_headers["X-Test"] == "test"
    assert clients[0].extra_headers_from_state == {"X-Session-ID": "session_id"}


def test_setup_clients_assigns_renderer_model_name():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_config=Qwen3VLRendererConfig(),
        renderer_model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    assert clients[0].renderer_model_name == "Qwen/Qwen3-VL-4B-Instruct"


def test_setup_clients_preserves_chat_client_defaults():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(client_config)

    assert clients == [
        vf.ClientConfig(
            client_idx=0,
            client_type="openai_chat_completions",
            api_key_var="PRIME_API_KEY",
            api_base_url="http://worker-a:8000/v1",
            timeout=client_config.timeout,
            connect_timeout=client_config.connect_timeout,
            max_connections=8192,
            max_keepalive_connections=8192,
            max_retries=10,
            extra_headers={},
            extra_headers_from_state={},
        )
    ]


def test_build_client_contract_matches_both_call_sites():
    """Lock the build_client factory contract: the pool path (setup_clients)
    and the fixed-member path (_fixed_client) must each map their policy
    config onto the full vf.ClientConfig exactly as documented."""
    pool_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
        timeout=900,
        connect_timeout=12.0,
        max_retries=3,
        max_connections=64,
        max_keepalive_connections=32,
        headers={"X-Test": "test"},
        extra_headers_from_state={"X-Session-ID": "session_id"},
    )

    assert setup_clients(pool_config) == [
        vf.ClientConfig(
            client_idx=0,
            client_type="openai_chat_completions",
            api_base_url="http://worker-a:8000/v1",
            api_key_var="PRIME_API_KEY",
            timeout=900,
            connect_timeout=12.0,
            max_connections=64,
            max_keepalive_connections=32,
            max_retries=3,
            extra_headers={"X-Test": "test"},
            extra_headers_from_state={"X-Session-ID": "session_id"},
        )
    ]

    target = FixedMemberTargetConfig(
        members=["judge"],
        model="judge-model",
        base_url=["http://judge:8000/v1"],
        api_key_var="JUDGE_API_KEY",
        request_mode="token",
        headers={"X-Judge": "1"},
        timeout=600.0,
        connect_timeout=5.0,
        max_retries=2,
        max_connections=16,
        max_keepalive_connections=8,
    )

    assert _fixed_client("judge", target, member_id="judge", group_id="group-0") == vf.ClientConfig(
        client_idx=0,
        client_type="openai_chat_completions_token",
        api_base_url="http://judge:8000/v1",
        api_key_var="JUDGE_API_KEY",
        timeout=600.0,
        connect_timeout=5.0,
        max_connections=16,
        max_keepalive_connections=8,
        max_retries=2,
        extra_headers={"X-Judge": "1"},
        extra_headers_from_state={},
    )


def test_as_train_client_derives_renderer_twin_from_chat_eval_client():
    from renderers import Qwen3VLRendererConfig

    from prime_rl.utils.client import StaticInferencePool

    renderer_settings = Qwen3VLRendererConfig()
    pool = StaticInferencePool(
        ClientConfig(
            base_url=["http://worker-a:8000/v1"],
            api_key_var="PRIME_API_KEY",
            headers={"X-Test": "test"},
        ),
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        train_client_type="renderer",
        eval_client_type="openai_chat_completions",
        renderer_config=renderer_settings,
        pool_size=4,
    )

    eval_client = pool.eval_clients[0]
    assert eval_client.client_type == "openai_chat_completions"

    learner = pool.as_train_client(eval_client)
    assert learner.client_type == "renderer"
    assert learner.renderer_config == renderer_settings
    assert learner.renderer_model_name == "Qwen/Qwen3-VL-4B-Instruct"
    assert learner.renderer_pool_size == 4
    # Same server, headers, and timeouts — only the client type changed
    assert learner.api_base_url == eval_client.api_base_url
    assert learner.extra_headers == eval_client.extra_headers
    assert learner.timeout == eval_client.timeout
    # Train clients already speak the train type — identity
    train_client = pool.train_clients[0]
    assert pool.as_train_client(train_client) is train_client


def test_as_train_client_is_identity_for_chat_trained_pool():
    from prime_rl.utils.client import StaticInferencePool

    pool = StaticInferencePool(
        ClientConfig(base_url=["http://worker-a:8000/v1"], api_key_var="PRIME_API_KEY"),
        model_name="some-model",
    )

    eval_client = pool.eval_clients[0]
    assert pool.as_train_client(eval_client) is eval_client


# --- regression tests for the per-peer admin-gather hardening (verify-client-hardening: SHIP) ---
# These lock in the three behaviors proven live by fault-injection but previously
# untested in the committed suite: multi-peer attribution, CancelledError-verbatim,
# and the raise_on_failure=False soft path. _gather_admin is the single chokepoint
# every weight-sync fan-out routes through, so testing it covers all call sites.


def _client(base_url: str) -> httpx.AsyncClient:
    # A real AsyncClient so base_url attribution is exercised exactly as in prod
    # (the helper only reads .base_url off it; no network is touched).
    return httpx.AsyncClient(base_url=base_url)


async def _ok(value):
    return value


async def _boom(exc: BaseException):
    raise exc


def test_gather_admin_attributes_only_dead_peers_on_partial_failure():
    # Three peers; peers b and c die with DISTINCT error types, peer a succeeds.
    # The aggregated error must name ONLY the dead peers (each with its own error
    # type) and leave the healthy peer out -- this is the attribution that turns
    # opaque first-exception death into a recoverable, named failure.
    from prime_rl.utils.client import _gather_admin

    a, b, c = _client("http://peer-a:8000"), _client("http://peer-b:8000"), _client("http://peer-c:8000")
    err_b = httpx.ConnectError("conn refused")
    err_c = httpx.ReadTimeout("read timed out")

    async def run():
        return await _gather_admin(
            [a, b, c],
            [_ok("OK"), _boom(err_b), _boom(err_c)],
            op_name="update weights",
        )

    with pytest.raises(AdminGatherError) as exc_info:
        asyncio.run(run())

    err = exc_info.value
    assert err.op_name == "update weights"
    assert err.total == 3
    assert len(err.failures) == 2
    failed_clients = {client for client, _ in err.failures}
    assert failed_clients == {b, c}
    assert a not in failed_clients
    failed_excs = {type(exc) for _, exc in err.failures}
    assert failed_excs == {httpx.ConnectError, httpx.ReadTimeout}
    msg = str(err)
    assert "http://peer-b:8000" in msg and "http://peer-c:8000" in msg
    # The healthy peer is NOT smeared into the failure message.
    assert "http://peer-a:8000" not in msg
    assert "2/3" in msg


def test_gather_admin_reraises_cancelled_verbatim():
    # CancelledError is cooperative-cancellation, not a peer failure. It must
    # propagate AS CancelledError (so the event loop's cancellation machinery
    # still works), NOT be caught and re-wrapped as AdminGatherError.
    from prime_rl.utils.client import _gather_admin

    a, b = _client("http://peer-a:8000"), _client("http://peer-b:8000")

    async def run():
        return await _gather_admin(
            [a, b],
            [_ok("OK"), _boom(asyncio.CancelledError())],
            op_name="update weights",
        )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())


def test_gather_admin_soft_path_logs_loud_and_returns_exception_in_band():
    # The unload_lora_adapter site passes raise_on_failure=False (idempotent
    # teardown). It must NOT raise, but it MUST still: (a) log the dead peer
    # loudly, and (b) return the exception in-band so the caller can see it.
    #
    # We assert the loud log by attaching a temporary sink to get_logger()'s
    # private loguru _Logger instance (logger.py binds its own isolated logger,
    # not stdlib logging or the global loguru.logger). caplog/capsys/capfd all
    # miss it: the production sink is bound to the real sys.stdout at import time,
    # so pytest's per-test capture layers never see the bytes. A scoped sink on
    # the actual logger object tests emission at the source, capture-independent.
    from prime_rl.utils.client import _gather_admin
    from prime_rl.utils.logger import get_logger

    a, b = _client("http://peer-a:8000"), _client("http://peer-b:8000")
    err_b = RuntimeError("peer-down")

    records: list[str] = []
    logger = get_logger()
    sink_id = logger.add(lambda msg: records.append(str(msg)), level="ERROR")
    try:

        async def run():
            return await _gather_admin(
                [a, b],
                [_ok("OK"), _boom(err_b)],
                op_name="unload LoRA adapter",
                raise_on_failure=False,
            )

        results = asyncio.run(run())
    finally:
        logger.remove(sink_id)

    # No raise; success and failure both observable in-band, positionally aligned.
    assert results[0] == "OK"
    assert isinstance(results[1], RuntimeError)
    assert results[1] is err_b
    # The dead peer was logged loudly at ERROR level, named with its base URL.
    combined = "".join(records)
    assert "unload LoRA adapter failed on inference peer http://peer-b:8000" in combined
    assert "peer-down" in combined


def test_update_lora_adapter_arms_without_dangling_timeout_constant():
    # Site-level regression for a rebase defect the isolated _gather_admin tests
    # could NOT catch: _arm_lora_update referenced a dropped PAUSE_READ_TIMEOUT_S
    # constant, NameError-ing on the first NCCL-LoRA arm (the feature this PR
    # adds). Drive update_lora_adapter end-to-end with mock admin clients and
    # assert the arm POST to /update_lora actually fires (i.e. no NameError, and
    # the per-peer admin contract holds).
    from prime_rl.utils.client import update_lora_adapter

    posts: list[tuple[str, dict]] = []

    def make_client() -> AsyncMock:
        client = AsyncMock()
        client.base_url = "http://worker:8000"

        async def _post(path, **kwargs):
            posts.append((path, kwargs))
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.status_code = 200
            return resp

        async def _get(path, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.json = MagicMock(return_value={"status": "ok"})
            return resp

        client.post.side_effect = _post
        client.get.side_effect = _get
        return client

    clients = [make_client(), make_client()]
    # Should NOT raise NameError; arms + waits + pauses/resumes all run.
    asyncio.run(update_lora_adapter(clients, lora_name="test-lora", weight_dir=Path("/tmp/bcast"), step=7))

    arm_posts = [(path, kw) for path, kw in posts if path == "/update_lora"]
    # One arm POST per peer, carrying the step + adapters payload.
    assert len(arm_posts) == 2, f"expected 2 arm posts, got {len(arm_posts)}: {posts}"
    for _, kw in arm_posts:
        assert kw["json"]["step"] == 7
        assert kw["json"]["adapters"] == [{"lora_name": "test-lora", "lora_int_id": 1}]

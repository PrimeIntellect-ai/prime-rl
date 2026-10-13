import pytest

from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.clients import setup_client


def test_setup_client_forwards_the_request_gate():
    client_config = ClientConfig(
        base_url="https://api.pinference.ai/api/v1",
        api_key_var="PRIME_API_KEY",
        request_gate={"max_concurrent": 32},
    )

    client = setup_client(client_config)

    assert client.request_gate is not None
    assert client.request_gate.max_concurrent == 32
    assert client.request_gate.adaptive


def test_an_invalid_request_gate_is_rejected():
    with pytest.raises(ValueError):
        ClientConfig(base_url="https://api.pinference.ai/api/v1", request_gate={"max_concurrent": 0})


def test_the_renderer_client_is_never_gated():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(request_gate={"max_concurrent": 4})

    client = setup_client(client_config, client_type="renderer", renderer_config=Qwen3VLRendererConfig())

    assert client.request_gate is None

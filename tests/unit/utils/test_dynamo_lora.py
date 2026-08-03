import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from prime_rl.utils.dynamo import DynamoInferencePool, _parse_dynamo_workers

MODEL = "Qwen/Qwen3-0.6B"


def worker(**updates):
    value = {
        "component": "backend",
        "instance_id": 1,
        "model": MODEL,
        "admin_base_url": "http://worker:8120",
        "world_size": 1,
        "system_url": "http://worker:8181",
        "system_routes": ["update/load_lora"],
    }
    return {**value, **updates}


def pool():
    value = DynamoInferencePool.__new__(DynamoInferencePool)
    value._admin_clients = [AsyncMock()]
    value._lora_update_clients = [AsyncMock()]
    value._frontend_model_clients = [AsyncMock()]
    value._wait_for_ready_timeout = 1
    return value


def test_discovery_rejects_partial_lora_capability():
    payload = {
        "protocol_version": 1,
        "workers": [
            worker(),
            worker(component="prefill", instance_id=2, admin_base_url="http://prefill:8120", system_routes=[]),
        ],
    }

    with pytest.raises(ValueError, match="partial update/load_lora"):
        _parse_dynamo_workers(payload, MODEL)


def test_lora_update_resumes_after_publication():
    inference_pool = pool()

    with (
        patch("prime_rl.utils.dynamo._pause_engines", new=AsyncMock()) as pause,
        patch("prime_rl.utils.dynamo._load_lora_adapter", new=AsyncMock()) as load,
        patch("prime_rl.utils.dynamo._wait_for_model", new=AsyncMock()) as wait,
        patch("prime_rl.utils.dynamo._resume_engines", new=AsyncMock()) as resume,
    ):
        asyncio.run(inference_pool.update_weights(Path("/weights/adapter"), lora_name="policy", step=3))

    pause.assert_awaited_once()
    load.assert_awaited_once()
    wait.assert_awaited_once()
    resume.assert_awaited_once()


def test_lora_update_resumes_after_failure():
    inference_pool = pool()

    with (
        patch("prime_rl.utils.dynamo._pause_engines", new=AsyncMock()),
        patch("prime_rl.utils.dynamo._load_lora_adapter", new=AsyncMock(side_effect=RuntimeError("failed"))),
        patch("prime_rl.utils.dynamo._resume_engines", new=AsyncMock()) as resume,
        pytest.raises(RuntimeError, match="failed"),
    ):
        asyncio.run(inference_pool.update_weights(Path("/weights/adapter"), lora_name="policy", step=3))

    resume.assert_awaited_once()

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from prime_rl.orchestrator import vf_utils


@pytest.mark.parametrize(
    ("api_base_url", "expected"),
    [
        ("http://localhost:9000/v1", "http://localhost:9000/programs/release"),
        ("http://localhost:9000/v1/", "http://localhost:9000/programs/release"),
        ("http://localhost:9000", "http://localhost:9000/programs/release"),
    ],
)
def test_get_program_release_url_strips_v1_suffix(api_base_url, expected):
    assert vf_utils.get_program_release_url(api_base_url) == expected


@pytest.mark.parametrize("program_id", ["program-1", None])
def test_run_group_releases_program_conditionally(monkeypatch, program_id):
    env = MagicMock()
    env.run_group = AsyncMock(return_value=[])
    client = MagicMock()
    release_mock = AsyncMock()
    monkeypatch.setattr(vf_utils, "release_program", release_mock)

    asyncio.run(
        vf_utils.run_group(
            env=env,
            client=client,
            model_name="dummy-model",
            example={"task": "dummy", "prompt": "hello"},
            rollouts_per_example=2,
            sampling_args={"temperature": 1.0},
            program_id=program_id,
        )
    )

    call_kwargs = env.run_group.await_args.kwargs
    request_sampling_args = call_kwargs["sampling_args"]
    if program_id is None:
        release_mock.assert_not_awaited()
        assert "extra_body" not in request_sampling_args
    else:
        release_mock.assert_awaited_once_with(client, program_id)
        assert request_sampling_args["extra_body"]["program_id"] == program_id

import os

import httpx
import pytest
from huggingface_hub import snapshot_download

from tests.integration.conftest import REF_MODEL, REF_PORT


@pytest.mark.gpu
def test_standalone_server_reloads_weights(ref_inference):
    checkpoint = snapshot_download(REF_MODEL, local_files_only=True)
    engine = f"http://localhost:{REF_PORT + 100}"
    headers = (
        {"X-Prime-Admin-Token": os.environ["PRIME_RL_ADMIN_TOKEN"]} if "PRIME_RL_ADMIN_TOKEN" in os.environ else {}
    )
    update = httpx.post(
        f"{engine}/update_weights",
        json={"weight_dir": checkpoint, "weight_version": 1},
        headers=headers,
        timeout=120,
    )
    update.raise_for_status()
    response = httpx.post(
        f"http://localhost:{REF_PORT}/v1/completions",
        json={"model": REF_MODEL, "prompt": "Hello", "max_tokens": 1},
        timeout=60,
    )
    response.raise_for_status()
    assert response.json()["usage"]["completion_tokens"] == 1

import asyncio

import pytest
from httpx import Response
from openai import AsyncOpenAI

pytestmark = [pytest.mark.slow, pytest.mark.gpu]
pytest.importorskip("sglang")

MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"


def test_chat_and_admin(sglang_server):
    async def run():
        client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert resp.choices[0].message.content
        base = str(client.base_url)[:-4]
        r = await client.post(base + "/update_weights", cast_to=Response, body={"model_path": MODEL})
        assert r.status_code == 200
        r = await client.post(base + "/flush_cache", cast_to=Response, body={})
        assert r.status_code == 200

    asyncio.run(run())

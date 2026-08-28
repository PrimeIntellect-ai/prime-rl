"""End-to-end request-shape check for raw Qwen3-VL inference."""

from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pytest

_HF_CACHE = Path("~/.cache/huggingface/hub").expanduser()
_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def _model_cached() -> bool:
    safe = "models--" + _MODEL.replace("/", "--")
    snapshots = _HF_CACHE / safe / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(p.is_dir() for p in snapshots.iterdir())


pytestmark = pytest.mark.skipif(
    not _model_cached(),
    reason=f"{_MODEL}: HF snapshot not cached locally",
)


class _FakeOpenAI:
    """Minimal AsyncOpenAI stand-in that captures POST bodies.

    ``renderers.client.generate`` calls ``client.post(absolute_url,
    body=...)``; we capture the body for assertions and return a canned
    generate response so the parse-side of the flow runs.
    """

    def __init__(self, image_pad_id: int):
        self.calls: list[dict[str, Any]] = []
        self.base_url = "http://fake-host:8000/v1"
        self.image_pad_id = image_pad_id

    async def post(self, path, *, cast_to=dict, body=None, options=None):
        self.calls.append({"path": path, "body": body, "options": options})
        # Reply with two sampled tokens + <|im_end|>. The renderer's
        # parse_response slices the content tokens.
        prompt_ids = list(body["token_ids"])
        pad_index = prompt_ids.index(self.image_pad_id)
        prompt_ids[pad_index : pad_index + 1] = [self.image_pad_id] * 4
        payload = {
            "request_id": "qwen-vl-e2e",
            "prompt_token_ids": prompt_ids,
            "choices": [
                {
                    "index": 0,
                    "token_ids": [50, 60, 151645],
                    "logprobs": {
                        "content": [
                            {"token": "token_id:50", "logprob": -0.1},
                            {"token": "token_id:60", "logprob": -0.2},
                            {"token": "token_id:151645", "logprob": -0.3},
                        ]
                    },
                    "finish_reason": "stop",
                },
            ],
        }
        return httpx.Response(200, content=json.dumps(payload).encode())


def test_generate_qwen3_vl_sends_raw_content_and_uses_expanded_prompt_ids():
    from PIL import Image
    from renderers.base import load_tokenizer
    from renderers.client import generate
    from renderers.qwen3_vl import Qwen3VLRenderer

    tokenizer = load_tokenizer(_MODEL)
    renderer = Qwen3VLRenderer(tokenizer)

    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    fake = _FakeOpenAI(image_pad_id)

    # ── Build a user message with an image (OpenAI content-part shape). ─
    img = Image.new("RGB", (224, 224), color=(64, 128, 255))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    image_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this picture?"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    result = asyncio.run(
        generate(
            client=fake,
            renderer=renderer,
            messages=messages,
            model=_MODEL,
            sampling_params={"max_tokens": 16},
            # Explicit cap so generate() skips the /v1/models discovery round-trip.
            max_prompt_len=1_000_000,
            process_multimodal=False,
        )
    )

    assert len(fake.calls) == 1
    body = fake.calls[0]["body"]
    assert "features" not in body
    assert body["content_parts"] == [{"type": "image_url", "url": image_url}]
    assert body["token_ids"].count(image_pad_id) == 1
    assert result["renderer_prompt_ids"] == body["token_ids"]
    assert result["prompt_ids"].count(image_pad_id) == 4
    assert result["completion_ids"] == [50, 60, 151645]

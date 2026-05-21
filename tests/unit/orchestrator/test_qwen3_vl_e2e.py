"""End-to-end integration test for the Qwen3-VL renderer path.

Walks a multimodal request through the full client stack — RendererClient
→ renderers.client.generate → /inference/v1/generate features payload —
with the HTTP layer mocked, and verifies that vLLM can deserialize the
features back into engine inputs identical to what its own server-side
processor would have produced for the same messages.

This is the strongest end-to-end check we can run without a GPU. The
remaining missing piece (vLLM actually consuming the engine input,
sampling tokens, and returning them) is exercised in real rollouts.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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

    The renderer client calls ``client.post(absolute_url, body=...)``;
    we capture the body for assertions and return a canned generate
    response so the parse-side of the flow runs.
    """

    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.base_url = "http://fake-host:8000/v1"

    async def post(self, path, *, cast_to=dict, body=None, options=None):
        self.calls.append({"path": path, "body": body, "options": options})
        # Reply with two sampled tokens + <|im_end|>. The renderer's
        # parse_response slices the content tokens.
        return {
            "request_id": "qwen-vl-e2e",
            "choices": [
                {
                    "index": 0,
                    "token_ids": [50, 60, 151645],
                    "logprobs": {
                        "content": [
                            {"token": "t1", "logprob": -0.1},
                            {"token": "t2", "logprob": -0.2},
                            {"token": "t3", "logprob": -0.3},
                        ]
                    },
                    "finish_reason": "stop",
                },
            ],
        }


def test_renderer_client_qwen3_vl_e2e_features_payload_roundtrips_through_vllm():
    """Walk a Qwen3-VL multimodal turn through the renderer client and
    verify the resulting ``/inference/v1/generate`` body has a valid
    ``features`` payload that:

    1. parses through vLLM's ``GenerateRequest`` pydantic model,
    2. decodes back to ``MultiModalKwargsItem`` instances carrying
       ``pixel_values`` + ``image_grid_thw`` of the right shapes,
    3. has placeholder ranges that exactly cover the ``<|image_pad|>``
       runs in the prompt token sequence.
    """
    from PIL import Image
    from renderers.base import load_tokenizer
    from renderers.qwen3_vl import Qwen3VLRenderer
    from transformers import AutoProcessor
    from verifiers.clients.renderer_client import RendererClient
    from verifiers.types import (
        ClientConfig,
        UserMessage,
    )
    from vllm.entrypoints.serve.disagg.mm_serde import decode_mm_kwargs_item
    from vllm.entrypoints.serve.disagg.protocol import GenerateRequest

    # ── Build a real Qwen3VLRenderer with a real processor. ─────────────
    tokenizer = load_tokenizer(_MODEL)
    processor = AutoProcessor.from_pretrained(_MODEL)
    renderer = Qwen3VLRenderer(tokenizer, processor=processor)

    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # ── Manually wire a RendererClient bypassing the pool factory. ──────
    client_cfg = ClientConfig(client_type="renderer", base_url="http://fake-host:8000/v1")
    rc = object.__new__(RendererClient)
    rc._config = client_cfg
    rc._renderer = renderer
    rc._pool_size = 1
    rc._client = _FakeOpenAI()
    rc.logger = MagicMock()

    # ── Build a verifiers-shaped user message with an image. ────────────
    img = Image.new("RGB", (224, 224), color=(64, 128, 255))
    # The renderer accepts the OpenAI ``image_url`` content-part shape —
    # the same shape verifiers' UserMessage carries through.
    user = UserMessage(
        content=[
            {"type": "text", "text": "What's in this picture?"},
            # Embed the PIL image directly. The verifiers→renderer message
            # converter forwards content unchanged for our purposes.
            {"type": "image", "image": img},
        ]
    )

    # to_native_prompt converts to renderer-shaped messages.
    prompt, _ = asyncio.run(rc.to_native_prompt([user]))
    sampling = {"max_tokens": 16}

    response = asyncio.run(
        rc.get_native_response(
            prompt=prompt,
            model=_MODEL,
            sampling_args=sampling,
            tools=None,
        )
    )

    # ── The HTTP body should carry a features payload. ──────────────────
    fake = rc.client
    assert isinstance(fake, _FakeOpenAI)
    assert len(fake.calls) == 1
    body = fake.calls[0]["body"]
    assert "features" in body, "RendererClient should ship features for image content"
    features = body["features"]

    # ── Pydantic-roundtrip through vLLM's GenerateRequest model. ────────
    gen_req = GenerateRequest(
        token_ids=body["token_ids"],
        features=features,
        sampling_params=body["sampling_params"],
    )
    assert gen_req.features is not None
    assert "image" in gen_req.features.mm_hashes
    assert len(gen_req.features.mm_hashes["image"]) == 1

    # ── Placeholder anchoring: the offset/length in features must land
    #    exactly on a run of <|image_pad|> ids in the prompt. ───────────
    placeholders = gen_req.features.mm_placeholders["image"]
    assert len(placeholders) == 1
    ph = placeholders[0]
    pad_slice = body["token_ids"][ph.offset : ph.offset + ph.length]
    assert all(t == image_pad_id for t in pad_slice), (
        f"placeholder span ({ph.offset}, {ph.length}) does not cover image_pad tokens; slice={pad_slice[:8]}..."
    )

    # ── kwargs_data decodes to MultiModalKwargsItem with the right keys. ─
    assert gen_req.features.kwargs_data is not None
    encoded_items = gen_req.features.kwargs_data["image"]
    assert len(encoded_items) == 1
    item = decode_mm_kwargs_item(encoded_items[0])
    assert set(item.keys()) == {"pixel_values", "image_grid_thw"}

    # The image_grid_thw must match what the HF processor would have
    # produced for the same PIL image — strongest signal that the engine
    # sees the same image features the trainer will.
    direct_proc_out = processor.image_processor(images=[img], return_tensors="pt")
    expected_grid = direct_proc_out["image_grid_thw"][0].tolist()
    assert item["image_grid_thw"].data.tolist() == expected_grid

    # ── Response parsed through renderer's parse_response. ──────────────
    assert response["completion_ids"] == [50, 60, 151645]
    # multi_modal_data surfaces on the result so the caller can persist it.
    assert response["multi_modal_data"] is not None
    assert len(response["multi_modal_data"].mm_items["image"]) == 1

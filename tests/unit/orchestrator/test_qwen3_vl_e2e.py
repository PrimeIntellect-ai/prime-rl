"""Qwen3-VL image features must match between inference and the training record."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
_HF_CACHE = Path("~/.cache/huggingface/hub").expanduser()
pytestmark = pytest.mark.skipif(
    not (_HF_CACHE / ("models--" + _MODEL.replace("/", "--")) / "snapshots").is_dir(),
    reason=f"{_MODEL}: HF snapshot not cached locally",
)


def test_generate_qwen3_vl_e2e_features_payload_roundtrips_through_vllm():
    from PIL import Image
    from renderers import Qwen3VLRendererConfig, load_tokenizer
    from renderers.qwen3_vl import Qwen3VLRenderer
    from transformers import AutoProcessor
    from verifiers.v1.dialects import ChatDialect
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    from prime_rl.inference.vllm.serving_chat import PrimeRlServingChat

    tokenizer = load_tokenizer(_MODEL)
    processor = AutoProcessor.from_pretrained(_MODEL)
    renderer = Qwen3VLRenderer(tokenizer, processor=processor)
    image = Image.new("RGB", (224, 224), color=(64, 128, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this picture?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()},
                },
            ],
        }
    ]
    request = ChatCompletionRequest(model=_MODEL, messages=messages, return_token_ids=True)
    serving = object.__new__(PrimeRlServingChat)
    serving.model_config = SimpleNamespace(tokenizer=_MODEL, max_model_len=1_000_000)
    serving.chat_template = None
    serving.chat_template_content_format = "auto"
    serving.default_chat_template_kwargs = {}
    serving.training_renderer_lock = Lock()
    serving.training_tokenizer = tokenizer
    serving.training_renderer = renderer
    serving.training_template_kwargs = {}
    serving.training_renderer_config = Qwen3VLRendererConfig()
    _, (engine_input,) = serving._render_training_prompt(request)

    assert engine_input["type"] == "multimodal"
    assert len(engine_input["mm_hashes"]["image"]) == 1
    (placeholder,) = engine_input["mm_placeholders"]["image"]
    pad_ids = engine_input["prompt_token_ids"][placeholder.offset : placeholder.offset + placeholder.length]
    assert pad_ids and all(token == tokenizer.convert_tokens_to_ids("<|image_pad|>") for token in pad_ids)
    (item,) = engine_input["mm_kwargs"]["image"]
    assert set(item) == {"pixel_values", "image_grid_thw"}
    expected = processor.image_processor(images=[image], return_tensors="pt")
    assert item["image_grid_thw"].data.tolist() == expected["image_grid_thw"][0].tolist()

    dialect = ChatDialect()
    response = dialect.parse_response(
        dialect.validate_response(
            {
                "id": "vlm-record",
                "object": "chat.completion",
                "created": 0,
                "model": _MODEL,
                "prompt_token_ids": engine_input["prompt_token_ids"],
                "training_metadata": request._prime_training_metadata,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "A picture."},
                        "token_ids": [50, 60, 151645],
                        "logprobs": {
                            "content": [{"token": f"token_id:{token}", "logprob": -0.1} for token in [50, 60, 151645]]
                        },
                    }
                ],
            }
        )
    )
    assert response.tokens is not None and response.tokens.multi_modal_data is not None
    (training_item,) = response.tokens.multi_modal_data.mm_items["image"]
    np.testing.assert_array_equal(training_item["pixel_values"], expected["pixel_values"].numpy())
    np.testing.assert_array_equal(training_item["image_grid_thw"], expected["image_grid_thw"].numpy())
    assert len(response.tokens.is_content) == len(engine_input["prompt_token_ids"])

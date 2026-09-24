"""Qwen3-VL training records retain inline images and exact placeholder ranges."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
_HF_CACHE = Path("~/.cache/huggingface/hub").expanduser()
pytestmark = pytest.mark.skipif(
    not (_HF_CACHE / ("models--" + _MODEL.replace("/", "--")) / "snapshots").is_dir(),
    reason=f"{_MODEL}: HF snapshot not cached locally",
)


def test_generate_qwen3_vl_e2e_preserves_inline_images_and_expanded_prompt_ids():
    import verifiers.v1 as vf
    from PIL import Image
    from renderers import Qwen3VLRendererConfig
    from renderers.base import load_tokenizer
    from renderers.qwen3_vl import Qwen3VLRenderer
    from transformers import AutoProcessor
    from verifiers.v1.dialects import ChatDialect
    from verifiers.v1.graph import prepare_turn
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    from prime_rl.inference.vllm.serving_chat import PrimeRlServingChat
    from prime_rl.orchestrator.trajectories import trace_to_samples

    tokenizer = load_tokenizer(_MODEL)
    processor = AutoProcessor.from_pretrained(_MODEL)
    renderer = Qwen3VLRenderer(tokenizer, processor=processor)
    image = Image.new("RGB", (224, 224), color=(64, 128, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this picture?"},
                {"type": "image_url", "image_url": {"url": image_url}},
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
    assert response.tokens is not None
    assert "multi_modal_data" not in request._prime_training_metadata
    assert response.tokens.mm_placeholders == [(placeholder.offset, placeholder.length)]
    assert len(response.tokens.is_content) == len(engine_input["prompt_token_ids"])
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="image")),
    )
    prompt, _ = dialect.parse_request({"messages": messages})
    prepare_turn(trace, prompt.messages).commit(response)
    (sample,) = trace_to_samples(trace)
    assert sample.token_ids == [*engine_input["prompt_token_ids"], 50, 60, 151645]
    assert sample.mm_refs is not None
    (image_ref,) = sample.mm_refs.images
    assert (image_ref.url, image_ref.offset, image_ref.length) == (image_url, placeholder.offset, placeholder.length)

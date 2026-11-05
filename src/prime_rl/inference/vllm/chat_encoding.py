"""Render chat prompts with a local DeepSeek-style chat encoding.

DeepSeek models ship their chat format as Python (``encoding/encoding_*.py``) instead of a Jinja
template, and vLLM bundles a copy of it for each DeepSeek tokenizer mode. With
``[inference] chat_encoding`` set, those tokenizer modes call ``encode_messages`` from the local
file, so a format that vLLM does not bundle can be served.
"""

import importlib.util
import json
from typing import Any


def _wire_message(message: dict[str, Any]) -> dict[str, Any]:
    """A message in the OpenAI wire form that shipped encodings take. vLLM's parsed messages
    carry content as text blocks and tool call arguments as dicts."""
    message = dict(message)
    content = message.get("content")
    if isinstance(content, list):
        if any(block.get("type") != "text" for block in content):
            raise ValueError("A chat encoding renders text-only messages.")
        # vLLM's bundled encodings join text blocks the same way.
        message["content"] = "\n\n".join(block["text"] for block in content)
    if message.get("tool_calls"):
        message["tool_calls"] = [
            {**call, "function": {**call["function"], "arguments": _json(call["function"]["arguments"])}}
            for call in message["tool_calls"]
        ]
    return message


def _json(arguments: Any) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)


def patch_chat_encoding(path: str) -> None:
    from vllm.tokenizers import deepseek_v4, deepseek_v32, deepseek_v41

    spec = importlib.util.spec_from_file_location("prime_rl_chat_encoding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def encode_messages(messages: list[dict[str, Any]], **kwargs) -> str:
        return module.encode_messages([_wire_message(message) for message in messages], **kwargs)

    for tokenizer_module in (deepseek_v32, deepseek_v4, deepseek_v41):
        tokenizer_module.encode_messages = encode_messages

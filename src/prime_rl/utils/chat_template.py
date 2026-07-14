import json
from typing import Any, Callable

from transformers.tokenization_utils import PreTrainedTokenizer


class IncrementalTokenizationError(ValueError):
    """Raised when incremental tokenization produces inconsistent token prefixes."""

    pass


def normalize_messages(messages: Any, default_role: str) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if isinstance(messages, str):
        return [{"role": default_role, "content": messages}]
    if isinstance(messages, dict):
        return [dict(messages)]
    if isinstance(messages, list):
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, str):
                normalized.append({"role": default_role, "content": message})
            elif isinstance(message, dict):
                normalized.append(dict(message))
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")
        return normalized
    raise TypeError(f"Unsupported messages container type: {type(messages)}")


def deserialize_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _deserialize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function", {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return {
            **tool_call,
            "function": {**function, "arguments": arguments},
        }

    deserialized_messages: list[dict[str, Any]] = []
    for message in messages:
        if "tool_calls" not in message:
            deserialized_messages.append(dict(message))
            continue

        tool_calls = message.get("tool_calls") or []
        deserialized_messages.append(
            {
                **message,
                "tool_calls": [_deserialize_tool_call(tc) for tc in tool_calls],
            }
        )

    return deserialized_messages


def strip_message_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _strip(message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if isinstance(content, str):
            return {**message, "content": content.strip()}
        return message

    return [_strip(message) for message in messages]


def resolve_sft_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    if "messages" in example:
        messages = normalize_messages(example["messages"], default_role="assistant")
    elif "prompt" in example and "completion" in example:
        messages = normalize_messages(example["prompt"], default_role="user") + normalize_messages(
            example["completion"], default_role="assistant"
        )
    else:
        raise ValueError("SFT rows need either 'messages' or both 'prompt' and 'completion'")
    return strip_message_content(deserialize_tool_calls(messages))


def resolve_sft_tools(example: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tools = example.get("tools", example.get("tool_defs"))
    if not raw_tools:
        return []
    if isinstance(raw_tools, str):
        raw_tools = json.loads(raw_tools)
    return [
        tool
        if isinstance(tool, dict) and tool.get("type") == "function" and "function" in tool
        else {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
                **({} if tool.get("strict") is None else {"strict": tool["strict"]}),
            },
        }
        for tool in raw_tools
    ]


def should_add_generation_prompt(messages: list[dict[str, Any]], idx: int) -> bool:
    role = messages[idx].get("role")
    if role not in ("user", "tool"):
        return False
    if idx + 1 >= len(messages):
        return False
    return messages[idx + 1].get("role") == "assistant"


def render_messages(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    add_generation_prompt: bool = False,
) -> list[int]:
    kwargs = dict(chat_template_kwargs or {})
    kwargs["add_generation_prompt"] = add_generation_prompt
    if tools is not None:
        kwargs["tools"] = tools
    kwargs["return_dict"] = False
    return list(tokenizer.apply_chat_template(messages, **kwargs))


def build_incremental_token_mask(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, Any]],
    *,
    role_to_mask: Callable[[dict[str, Any]], bool],
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    collapse_consecutive_tool_messages: bool = False,
) -> tuple[list[int], list[bool]]:
    token_mask: list[bool] = []
    prev_ids: list[int] = []
    prev_len = 0

    for idx, message in enumerate(messages):
        role = message.get("role")
        if collapse_consecutive_tool_messages and role == "tool" and idx + 1 < len(messages):
            if messages[idx + 1].get("role") == "tool":
                continue

        cur_ids = render_messages(
            tokenizer,
            messages[: idx + 1],
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=should_add_generation_prompt(messages, idx),
        )

        if prev_ids != cur_ids[:prev_len]:
            raise IncrementalTokenizationError(
                f"Mismatch in incremental tokenization with chat template at message {idx} (role={role}). "
                "This usually means the chat template is not stable under incremental application. "
                "The sample will be skipped."
            )

        token_mask.extend([role_to_mask(message)] * (len(cur_ids) - prev_len))
        prev_ids = cur_ids
        prev_len = len(cur_ids)

    return prev_ids, token_mask

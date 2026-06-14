import json
from collections.abc import Mapping
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages


def token_payload_length(tokens: Mapping[str, Any] | None, key: str) -> int | None:
    """Return a token-array length from either full or compacted raw tokens."""
    if not tokens:
        return None

    value = tokens.get(key)
    if value is not None:
        try:
            return len(value)
        except TypeError:
            pass

    compact_value = tokens.get(f"{key}_len")
    return compact_value if isinstance(compact_value, int) else None


def token_payload_ids(tokens: Mapping[str, Any] | None) -> list[int] | None:
    """Return prompt+completion token IDs when raw arrays are still present."""
    if not tokens:
        return None
    prompt_ids = tokens.get("prompt_ids")
    completion_ids = tokens.get("completion_ids")
    if not isinstance(prompt_ids, list) or not isinstance(completion_ids, list):
        return None
    return prompt_ids + completion_ids


def render_prompt_completion_text(tokenizer: PreTrainedTokenizer, prompt: Any, completion: Any) -> str:
    """Render prompt/completion messages after raw token IDs have been compacted."""
    messages = normalize_messages(prompt, default_role="user")
    messages.extend(normalize_messages(completion, default_role="assistant"))
    if not messages:
        return ""
    try:
        return tokenizer.apply_chat_template(deserialize_tool_calls(messages), tokenize=False)
    except Exception:
        return json.dumps(messages)

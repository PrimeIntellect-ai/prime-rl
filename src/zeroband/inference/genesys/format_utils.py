import json
import re


def _find_last_json_block(text: str) -> str | None:
    """Return the string content of the last JSON object in ``text``."""
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    matches = list(fence_pattern.finditer(text))
    if matches:
        return matches[-1].group(1).strip()

    end = text.rfind("}")
    if end == -1:
        return None

    depth = 0
    i = end
    while i >= 0:
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                start = i
                return text[start : end + 1].strip()
        i -= 1
    return None


def extract_last_json(completion: str) -> dict | None:
    """Extract and parse the last JSON object following ``</think>``."""
    split_response = completion.split("</think>", 1)
    if len(split_response) == 1:
        return None
    tail = split_response[1].strip()
    json_str = _find_last_json_block(tail)
    if json_str is None:
        return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

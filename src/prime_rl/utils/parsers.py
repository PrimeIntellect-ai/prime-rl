"""Regex-based auto-resolution for vLLM tool_call_parser and reasoning_parser.

Each list is an ordered sequence of (pattern, parser_name) tuples.
The first matching pattern wins, so more specific patterns must come first.
"""

import re

# (regex, parser_name) — first match wins.
TOOL_CALL_PARSER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^zai-org/GLM-4\.5"), "glm45"),
    (re.compile(r"^zai-org/GLM-4\.7"), "glm47"),
    (re.compile(r"^MiniMaxAI/MiniMax-M2"), "minimax_m2"),
    (re.compile(r"^PrimeIntellect/INTELLECT-3"), "hermes"),
    # Qwen3.5 uses qwen3_coder — must be before the Qwen3 catch-all
    (re.compile(r"^Qwen/Qwen3\.5-"), "qwen3_coder"),
    (re.compile(r"^Qwen/Qwen3-"), "hermes"),
]

REASONING_PARSER_PATTERNS: list[tuple[re.Pattern[str], str]] = []


def resolve_parser(model_name: str, patterns: list[tuple[re.Pattern[str], str]]) -> str | None:
    """Auto-detect parser from model name. Returns the first matching pattern's parser."""
    for pattern, parser_name in patterns:
        if pattern.search(model_name):
            return parser_name
    return None

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


def resolve_parser(
    model_name: str, parser_value: str | None, patterns: list[tuple[re.Pattern[str], str]]
) -> str | None:
    """Resolve a parser value. If "auto", walk patterns and return the first match."""
    if parser_value != "auto":
        return parser_value
    for pattern, parser_name in patterns:
        if pattern.search(model_name):
            return parser_name
    return None


def resolve_tool_call_parser(model_name: str, tool_call_parser: str | None) -> str | None:
    return resolve_parser(model_name, tool_call_parser, TOOL_CALL_PARSER_PATTERNS)


def resolve_reasoning_parser(model_name: str, reasoning_parser: str | None) -> str | None:
    return resolve_parser(model_name, reasoning_parser, REASONING_PARSER_PATTERNS)

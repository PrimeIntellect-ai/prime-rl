import re

# (regex, parser_name) — first match wins.
TOOL_CALL_PARSER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^deepseek-ai/DeepSeek-V3\.2"), "deepseek_v32"),
    (re.compile(r"^deepseek-ai/DeepSeek-V3\.1"), "deepseek_v31"),
    (re.compile(r"^zai-org/GLM-4\.5"), "glm45"),
    (re.compile(r"^zai-org/GLM-4\.7"), "glm47"),
    (re.compile(r"^MiniMaxAI/MiniMax-M2"), "minimax_m2"),
    (re.compile(r"^PrimeIntellect/INTELLECT-3"), "qwen3_coder"),
    (re.compile(r"^stepfun-ai/Step-3\.5"), "step3p5"),
    # Qwen3.5 uses qwen3_coder — must be before the Qwen3 catch-all
    (re.compile(r"^Qwen/Qwen3\.5-"), "qwen3_coder"),
    (re.compile(r"^Qwen/Qwen3-"), "hermes"),
]

REASONING_PARSER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^zai-org/GLM-4"), "glm45"),
    (re.compile(r"^MiniMaxAI/MiniMax-M2"), "minimax_m2"),
    (re.compile(r"^PrimeIntellect/INTELLECT-3"), "deepseek_r1"),
    (re.compile(r"^stepfun-ai/Step-3\.5"), "step3p5"),
    # Qwen3.5 and Qwen3 both use the "qwen3" reasoning parser
    (re.compile(r"^Qwen/Qwen3"), "qwen3"),
]


def resolve_parser(model_name: str, patterns: list[tuple[re.Pattern[str], str]]) -> str | None:
    """Auto-detect parser from model name. Returns the first matching pattern's parser."""
    for pattern, parser_name in patterns:
        if pattern.search(model_name):
            return parser_name
    return None

# Acknowledgements:
#   SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution
#   Xie, Chengxing et al., 2025
#
#   Agentless: Demystifying LLM-based Software Engineering Agents
#   Xia, Chunqiu Steven et al., 2024
#
#   SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
#   Yuxiang Wei et al., 2025

import re
from typing import Dict

import cydifflib

DIFF_BLOCK_REGEX = re.compile(r"```diff\s*(.*?)\s*```", re.DOTALL)
INDEX_LINE_REGEX = re.compile(r"^index [^\n]*\n")
FUNC_CONTEXT_REGEX = re.compile(r"(?m)^(@@[^@]*@@).*")


def parse_last_diff_codeblock(markdown_str: str) -> str:
    """Extract the last ```diff``` code block from markdown text."""
    matches = DIFF_BLOCK_REGEX.findall(markdown_str)
    if matches:
        return matches[-1].strip()
    else:
        return ""


def normalize_diff(diff_text: str) -> str:
    """
    Normalize diff text by removing lines starting with 'index ...' and stripping function context after @@.
    The function context/section header can differ between diffs, because language specific parsing might not be enabled.

    Example:
    ```diff
    diff --git a/file.py b/file.py
    index 1234567890..1234567890
    --- a/file.py
    +++ b/file.py
    @@ -15,1 +15,1 @@ def some_func():
    -    pass
    +    return
    ```

    becomes:
    ```diff
    diff --git a/file.py b/file.py
    --- a/file.py
    +++ b/file.py
    @@ -15,1 +15,1 @@
    -    pass
    +    return
    ```
    """
    diff_text = INDEX_LINE_REGEX.sub("", diff_text)
    diff_text = FUNC_CONTEXT_REGEX.sub(r"\1", diff_text)
    diff_text = diff_text.strip() + "\n"
    return diff_text


def compute_git_diff_reward(completion: str, verification_info: Dict) -> float:
    """
    Compute reward for git diff generation tasks using LCS (Longest Common Subsequence) ratio.

    Args:
        completion: Model's response string
        verification_info: Dict containing golden_diff

    Returns:
        Float score (0.0 to 1.0) representing diff similarity
    """
    # Extract the response after thinking (if present)
    if "</think>" in completion:
        response = completion.split("</think>")[1].strip()
    else:
        response = completion.strip()

    if not response:
        return 0.0

    # Get expected answer from verification_info
    golden_diff = verification_info.get("golden_diff", "")
    if not golden_diff:
        return 0.0

    try:
        # Extract diff from response
        response_diff = parse_last_diff_codeblock(response)
        response_diff = normalize_diff(response_diff)

        if not response_diff.strip():
            return 0.0

        # Calculate LCS ratio
        similarity = cydifflib.SequenceMatcher(None, response_diff, golden_diff, autojunk=False).ratio()

        return similarity

    except Exception:
        return 0.0

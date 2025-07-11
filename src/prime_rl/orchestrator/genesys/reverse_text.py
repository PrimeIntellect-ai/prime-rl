import re
from difflib import SequenceMatcher


def lcs_ratio(x: str, y: str) -> float:
    """
    Return the longest common subsequence ratio of x and y.
    """
    return SequenceMatcher(None, x, y).ratio()


def reverse_text(completion: str, verification_info: dict) -> int:
    """
    LCS ratio of the reversed prompt and the parsed completion.
    """

    # Extract solution between <answer> and </answer> tags
    answer_text = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if not answer_text:
        return 0

    ground_truth = verification_info.get("ground_truth")
    if not ground_truth:
        return 0

    return lcs_ratio(answer_text.group(1).strip(), ground_truth)

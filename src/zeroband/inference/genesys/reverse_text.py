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

    # Extract solution.
    if "</think>" in completion:
        model_solution = completion.split("</think>")[1]
    else:
        return 0

    return lcs_ratio(model_solution, verification_info["ground_truth"])

from typing import Dict
import re
import difflib


def compute_reward(completion: str, verification_info: Dict):
    # Extract answer
    if "<ascii_formatted>" not in completion:
        return 0
    answer_text = re.search(r"<ascii_formatted>(.*?)</ascii_formatted>", completion, re.DOTALL)
    if not answer_text:
        return 0

    # Get ground truth
    ground_truth = verification_info.get("ground_truth")
    if not ground_truth:
        return 0

    try:
        # Clean and split both into lines
        answer_lines = answer_text.group(1).strip().split("\n")
        truth_lines = ground_truth.strip().split("\n")

        # Use difflib to compare
        matcher = difflib.SequenceMatcher(None, answer_lines, truth_lines)

        # Get the longest matching block
        longest_block = max(matcher.get_matching_blocks(), key=lambda x: x.size, default=difflib.Match(0, 0, 0))

        # Calculate similarity ratio weighted by the longest continuous match
        similarity = matcher.ratio()
        continuous_ratio = longest_block.size / len(truth_lines)

        # Combine both metrics, heavily weighting continuous matches
        reward = (0.3 * similarity) + (0.7 * continuous_ratio)

        # Penalize if tree structure is invalid
        if not all(line.startswith(" ") or line.rstrip() == answer_lines[0] for line in answer_lines[1:]):
            reward *= 0.5

        # Penalize if +-- or `-- or | missing
        if not any("--" in line for line in answer_lines[1:]):
            reward *= 0.5

        return reward

    except Exception:
        return 0

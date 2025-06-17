import difflib
import re
from typing import Dict


def detect_format_signature(text: str) -> tuple:
    signature = []
    current_text = text

    # Check for markdown emphasis (changed to if statements for layering)
    if current_text.startswith("***") and current_text.endswith("***") and len(current_text) > 6:
        signature.append("triple_asterisk")
        current_text = current_text[3:-3]
    elif current_text.startswith("**") and current_text.endswith("**") and len(current_text) > 4:
        signature.append("bold")
        current_text = current_text[2:-2]
    elif current_text.startswith("*") and current_text.endswith("*") and len(current_text) > 2:
        signature.append("italic")
        current_text = current_text[1:-1]

    # Check for code blocks
    if current_text.startswith("```\n") and current_text.endswith("\n```"):
        signature.append("code_block")
        current_text = current_text[4:-4]

    # Check for wrapper characters (changed to if statements for layering)
    if current_text.startswith('"') and current_text.endswith('"') and len(current_text) > 2:
        signature.append("quotes")
        current_text = current_text[1:-1]
    elif current_text.startswith("[") and current_text.endswith("]") and len(current_text) > 2:
        signature.append("brackets")
        current_text = current_text[1:-1]
    elif current_text.startswith("(") and current_text.endswith(")") and len(current_text) > 2:
        signature.append("parentheses")
        current_text = current_text[1:-1]
    elif current_text.startswith("-") and current_text.endswith("-") and len(current_text) > 2:
        signature.append("dashes")
        current_text = current_text[1:-1]
    elif current_text.startswith("<") and current_text.endswith(">") and len(current_text) > 2:
        signature.append("angle_brackets")
        current_text = current_text[1:-1]

    # Check for uppercase (this can layer with other formats)
    if current_text.isupper() and len(current_text) > 0:
        signature.append("uppercase")

    return tuple(signature) if signature else ("none",)


def extract_and_score(text: str, tag_name: str, ground_truth: str) -> float:
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return 0

    extracted = match.group(1)

    # Check format compliance
    gt_sig = detect_format_signature(ground_truth)
    ext_sig = detect_format_signature(extracted)

    # Calculate similarity
    similarity = difflib.SequenceMatcher(None, extracted, ground_truth).ratio()

    # If format is wrong but content is extractable, give 0.1x reward
    if gt_sig != ("none",) and gt_sig != ext_sig:
        return similarity * 0.1

    return similarity


def compute_reward(completion: str, verification_info: Dict, tag_name: str = "extracted_formatted") -> float:
    try:
        # Skip thinking section
        text = completion
        think_end = completion.find("</think>")
        if think_end != -1:
            text = completion[think_end + len("</think>") :]

        # Check for dual case first
        if "ground_truth1" in verification_info and "ground_truth2" in verification_info:
            score1 = extract_and_score(text, "extracted_formatted1", verification_info["ground_truth1"])
            score2 = extract_and_score(text, "extracted_formatted2", verification_info["ground_truth2"])

            if score1 == 0 or score2 == 0:
                return 0

            return (score1 + score2) / 2.0

        # Single case
        ground_truth = verification_info.get("ground_truth")
        if not ground_truth:
            return 0

        return extract_and_score(text, tag_name, ground_truth)

    except Exception:
        return 0

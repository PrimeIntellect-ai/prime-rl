import difflib
import re
from typing import Dict


def detect_format_type(text: str) -> str:
    """
    Detect the formatting type used in the text.
    Returns the format type or 'none' if no specific format detected.
    """
    # Check for various formatting patterns
    if text.startswith("**") and text.endswith("**") and len(text) > 4:
        if text.startswith("***") and text.endswith("***"):
            return "triple_asterisk"
        return "bold"
    elif text.startswith("*") and text.endswith("*") and len(text) > 2:
        return "italic"
    elif text.isupper() and len(text) > 1:
        return "uppercase"
    elif text.startswith("```\n") and text.endswith("\n```"):
        return "code_block"
    elif text.startswith('"') and text.endswith('"') and len(text) > 2:
        return "quotes"
    elif text.startswith("[") and text.endswith("]") and len(text) > 2:
        return "brackets"
    elif text.startswith("(") and text.endswith(")") and len(text) > 2:
        return "parentheses"
    elif text.startswith("-") and text.endswith("-") and len(text) > 2:
        return "dashes"
    elif text.startswith("<") and text.endswith(">") and len(text) > 2:
        return "angle_brackets"

    return "none"


def compute_reward(completion: str, verification_info: Dict, tag_name: str = "extracted_formatted"):
    """
    Generic difflib-based reward computation for tasks expecting extracted content in XML tags.
    Uses normalized text comparison to avoid harsh penalties for formatting differences.
    Only looks for XML tags AFTER </think> if </think> exists.
    Returns 0 if format specification is not followed exactly.

    Args:
        completion: The model's completion text
        verification_info: Dictionary containing ground truth
        tag_name: XML tag name to extract content from

    Returns:
        Float reward between 0 and 1
    """
    # First, check if </think> exists and skip thinking section if it does
    search_text = completion
    think_end = completion.find("</think>")
    if think_end != -1:
        # If </think> found, only search AFTER the thinking section
        search_text = completion[think_end + len("</think>") :]

    # Extract answer from specified tag in the search text
    tag_pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    if f"<{tag_name}>" not in search_text:
        return 0

    answer_match = re.search(tag_pattern, search_text, re.DOTALL)
    if not answer_match:
        return 0

    # Check if there's exactly one XML tag with the right name (for bonus)
    xml_bonus = 0
    all_tags = re.findall(f"<{tag_name}>.*?</{tag_name}>", search_text, re.DOTALL)
    if len(all_tags) == 1:
        xml_bonus = 0.01

    # Get ground truth
    ground_truth = verification_info.get("ground_truth")
    if not ground_truth:
        return xml_bonus  # Return bonus if no ground truth to compare against

    try:
        # Extract content from both
        extracted_text = answer_match.group(1)
        ground_truth_text = ground_truth

        # CHECK FORMAT COMPLIANCE FIRST
        ground_truth_format = detect_format_type(ground_truth_text)
        extracted_format = detect_format_type(extracted_text)

        # If ground truth has a specific format, extracted text must match that format
        if ground_truth_format != "none" and ground_truth_format != extracted_format:
            return 0  # Format specification not followed - return 0

        # If format check passes, proceed with difflib comparison
        # Use difflib on raw text for sequence comparison
        matcher = difflib.SequenceMatcher(None, extracted_text, ground_truth_text)

        # Calculate similarity ratio
        reward = matcher.ratio()

        # Add the XML tag bonus
        reward += xml_bonus

        return min(reward, 1.0)  # Cap at 1.0 to prevent bonus from pushing over 1

    except Exception:
        return xml_bonus  # Return bonus even if comparison fails

import re
from zeroband.logger import get_logger


def extract_final_answer(text: str) -> str:
    """
    Extracts the final answer from the provided text.
    First, it searches for a marker "Answer:" (case-insensitive).
    If found, it returns the token right after it.
    Otherwise, it falls back to the last non-empty line.
    """
    match = re.search(r"(?i)Answer:\s*(\S+)", text)
    if match:
        return match.group(1).strip()
    # fallback: if no "answer: " use the last non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def verify_gsm_infinite(completion: str, verification_info: dict) -> int:
    """
    Verifies a GSM–Infinite puzzle by comparing the extracted final answer
    from the model's output with the ground truth solution.

    For example, given a problem like:

      ... (problem text) ...
      What is the average number of newborn children per adult wolf in Mayer Aquarium?
      Define average number of newborn children per adult wolf in South Zoo as I; so I = 2.
      Define average number of newborn children per adult wolf in Mayer Aquarium as p; so p = I = 2.
      Answer: 2.

    The verifier extracts "2" from the ground truth and from the model output (after a "</think>" separator if present)
    and returns 1 if they match, and -1 otherwise.
    """
    logger = get_logger()

    ground_truth_text = verification_info.get("solution", "").strip()
    if not ground_truth_text:
        logger.warning("No ground truth solution provided in verification_info")
        return -1

    gold_answer = extract_final_answer(ground_truth_text)

    if "</think>" in completion:
        text_to_parse = completion.split("</think>", 1)[1]
    else:
        text_to_parse = completion

    pred_answer = extract_final_answer(text_to_parse)

    if pred_answer == gold_answer:
        return 1
    else:
        logger.warning(f"Answer mismatch: predicted '{pred_answer}' vs gold '{gold_answer}'")
        return -1

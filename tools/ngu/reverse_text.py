"""Binary reverse-text reward for the NGU integration run."""

import re
from difflib import SequenceMatcher


async def binary_lcs(task, trace) -> float:
    match = re.search(r"<reversed_text>(.*?)</reversed_text>", trace.last_reply or "", re.DOTALL)
    response = match.group(1).strip() if match else ""
    return float(SequenceMatcher(None, response, task.answer).ratio() >= 0.8)

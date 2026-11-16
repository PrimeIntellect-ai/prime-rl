"""Built-in System One encoders: the prompts vLLM 0.33's ``/v1/systemone`` builds for the decision models it serves.

The prompt code is ported from ``vllm/entrypoints/openai/decisions/strategies.py`` (Apache-2.0, Copyright
contributors to the vLLM project). Most of what vLLM refuses is a malformed record, which ``parse_record`` raises on.
An encoder skips, with a warning, only a record it cannot encode as vLLM reads it: more questions or options than vLLM
reads, a non-string instructions, description or score level, which vLLM would render with ``str()``, or Clef-Flash
questions that alone do not fit.
"""

from prime_rl.trainer.sft.system_one.record import SystemOneRecord

MAX_QUESTIONS = 64


def find_refusal(record: SystemOneRecord) -> str | None:
    """Why the record cannot be encoded as vLLM reads it, or None."""
    if len(record.questions) > MAX_QUESTIONS:
        return f"it has {len(record.questions)} questions, more than the {MAX_QUESTIONS} vLLM reads per request"
    for question in record.questions:
        where = f"question {question.id!r}"
        if not isinstance(question.instructions, str):
            return f"{where}: instructions must be a string"
        if question.type == "score" and not all(isinstance(level, str) for level in question.criteria):
            return f"{where}: score levels must be strings"
        if any(text is not None and not isinstance(text, str) for text in question.descriptions):
            return f"{where}: descriptions must be strings or null"
    return None

"""Built-in System One strategies: the training half of the read strategies vLLM 0.33's ``/v1/systemone`` uses for the
decision models it serves, their prompts and their readouts.

The prompt code is ported from ``vllm/entrypoints/openai/decisions/strategies.py`` (Apache-2.0, Copyright
contributors to the vLLM project). Most of what vLLM refuses is a malformed record, which ``parse_record`` raises on.
A strategy skips, with a warning, only a record whose request it cannot render as vLLM reads it: more questions or
options than vLLM reads, a non-string instructions, description or score level, which vLLM would render with
``str()``, or Clef-Flash questions that alone do not fit.
"""

from prime_rl.trainer.sft.system_one.record import SystemOneRequest

MAX_QUESTIONS = 64


def find_refusal(request: SystemOneRequest) -> str | None:
    """Why the request cannot be rendered as vLLM reads it, or None."""
    if len(request.questions) > MAX_QUESTIONS:
        return f"it has {len(request.questions)} questions, more than the {MAX_QUESTIONS} vLLM reads per request"
    for question in request.questions:
        where = f"question {question.id!r}"
        if not isinstance(question.instructions, str):
            return f"{where}: instructions must be a string"
        if question.type == "score" and not all(isinstance(level, str) for level in question.criteria):
            return f"{where}: score levels must be strings"
        if any(text is not None and not isinstance(text, str) for text in question.descriptions):
            return f"{where}: descriptions must be strings or null"
    return None

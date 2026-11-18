"""A response stop that ends a rollout once the policy leaves the Qwen3.8 agent-turn format.

The Qwen3.8 renderer parses completions leniently: a tool call inside reasoning still runs, and a
malformed ``<tool_call>`` block is dropped or runs with partial arguments. This stop accepts only the
structure the chat template writes, after the ``<think>\\n`` the generation prompt opens:

    reasoning </think> content (<tool_call> <function=NAME> (<parameter=KEY> VALUE </parameter>)* </function> </tool_call>)* <|im_end|>

The violating turn stays on the trace, so it trains, and the harness gets no further model calls.
With ``skip_scoring``, the trace keeps no task rewards, so its reward is 0.

    [orchestrator.train.source.env.taskset.task.stops]
    format_violation = { fn = "prime_rl.orchestrator.plugins.qwen38_format:format_violation", skip_scoring = true }
"""

import logging
import re
from functools import cache

import verifiers.v1 as vf
from renderers.base import load_tokenizer

logger = logging.getLogger(__name__)

STRUCTURAL = (
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<tool_response>",
    "</tool_response>",
)
CALL = re.compile(r"\s*<function=([^>\n]+)>\s*((?:<parameter=[^>\n]+>.*?</parameter>\s*)*)</function>\s*", re.S)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>")


class _Tokens:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = {tag: tokenizer.convert_tokens_to_ids(tag) for tag in STRUCTURAL}
        self.tags = {token: tag for tag, token in self.ids.items()}


@cache
def _tokens(model: str) -> _Tokens:
    return _Tokens(load_tokenizer(model))


def qwen38_format_violation(tokens: _Tokens, token_ids: list[int]) -> str | None:
    """Where a completion first leaves the agent-turn format, or None if it keeps to it.

    A completion without a stop token was cut off, so only its tokens so far are checked.
    """
    ids = tokens.ids
    stop = next((k for k, token in enumerate(token_ids) if token in (ids["<|im_end|>"], ids["<|endoftext|>"])), None)
    if stop is not None and token_ids[stop] == ids["<|endoftext|>"]:
        return f"<|endoftext|> at token {stop}, where <|im_end|> belongs"
    end = len(token_ids) if stop is None else stop
    # Segments of plain text between structural tokens: [(tag or None, start, stop)].
    segments, start = [], 0
    for k in range(end):
        if token_ids[k] in tokens.tags:
            segments.append((None, start, k))
            segments.append((tokens.tags[token_ids[k]], k, k + 1))
            start = k + 1
    segments.append((None, start, end))
    tags = [(tag, k) for tag, k, _ in segments if tag is not None]
    texts = [tokens.tokenizer.decode(token_ids[a:b]) for tag, a, b in segments if tag is None]

    expected = "</think>"
    for index, (tag, position) in enumerate(tags):
        if tag != expected:
            return f"{tag} at token {position}, where {expected} belongs"
        text = texts[index + 1]
        if tag == "<tool_call>":
            if index + 1 == len(tags):
                return None if stop is None else f"unclosed <tool_call> at token {position}"
            match = CALL.fullmatch(text)
            if match is None:
                return f"malformed tool call at token {position}"
            keys = PARAMETER.findall(match.group(2))
            if len(keys) != len(set(keys)):
                return f"repeated argument name in the tool call at token {position}"
            expected = "</tool_call>"
        else:
            if tag == "</tool_call>" and text.strip():
                return f"text after </tool_call> at token {position}"
            expected = "<tool_call>"
    if expected == "</think>" and stop is not None:
        return "<|im_end|> before </think>"
    return None


def format_violation(response: vf.Response, trace: vf.Trace) -> bool:
    # A relayed call (the judge) returns no token ids, so there is nothing to check.
    if response.tokens is None:
        return False
    violation = qwen38_format_violation(_tokens(response.model), response.tokens.completion_ids)
    if violation is None:
        return False
    logger.warning("rollout %s: format violation: %s", trace.id, violation)
    trace.info["format_violation"] = violation
    return True

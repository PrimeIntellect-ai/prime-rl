"""A response stop that ends a rollout once Qwen3.6 leaves its agent-turn format (see format-stop.toml).

A completion must keep to what the Qwen3.6 chat template writes for an assistant turn: reasoning with no tags,
``</think>``, text, then only ``<tool_call>`` blocks of ``<function=NAME>`` with ``<parameter=KEY>VALUE</parameter>``
fields, then ``<|im_end|>``. A tool call before ``</think>`` is a violation; an empty ``<think></think>`` is not. The
renderer parses any completion leniently and the harness answers a malformed call with a tool error, so without this
stop the rollout goes on. A call must also name an offered tool and give its required arguments and no unknown ones;
the trace holds its tool list from the second turn on, so the first turn gets the format check only. A completion
without a stop token was cut off; only its tokens so far are checked. The violating turn stays on the trace and
trains; with ``skip_scoring`` its reward is 0. ``trace.info["format_violation"]`` records where the turn left the format.
"""

import logging
import re
from functools import cache

import verifiers.v1 as vf
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"
CALL = re.compile(r"\s*<function=([^>\n]*)>\s*(.*?)</function>\s*", re.S)
PARAM = re.compile(r"<parameter=([^>\n]*)>(.*?)</parameter>\s*", re.S)

logger = logging.getLogger(__name__)


@cache
def _tokenizer():
    # Stops run on the env server's event loop, so one tokenizer per process is enough.
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    return tokenizer, {i: t for t, i in tokenizer.get_added_vocab().items()}


def _call_violation(body: str, tools: dict[str, dict] | None) -> str | None:
    call = CALL.fullmatch(body)
    if call is None:
        return f"a tool call that is not <function=NAME> with <parameter=KEY> fields: {body[:80]!r}"
    name, fields = call.groups()
    keys, position = [], 0
    while position < len(fields):
        field = PARAM.match(fields, position)
        if field is None:
            return f"text that is not a <parameter=KEY> field in a {name!r} call: {fields[position : position + 80]!r}"
        keys.append(field.group(1))
        position = field.end()
    if len(set(keys)) < len(keys):
        return f"a repeated argument in a {name!r} call: {keys}"
    if tools is None:
        return None
    if name not in tools:
        return f"an unknown tool {name!r}"
    schema = tools[name]
    unknown = sorted(set(keys) - set(schema.get("properties", {}))) if "properties" in schema else []
    missing = sorted(set(schema.get("required", [])) - set(keys))
    if unknown or missing:
        return f"a {name!r} call with unknown arguments {unknown} and missing arguments {missing}"
    return None


def violation(ids: list[int], tools: dict[str, dict] | None = None) -> str | None:
    """Where a Qwen3.6 completion first leaves the agent-turn format, or None if it keeps to it. ``tools`` maps each
    offered tool's name to its JSON-schema parameters; None skips the name and argument checks."""
    tokenizer, tags = _tokenizer()
    # The completion as alternating text and tag items, the tag (an added token) or None for text.
    items, start = [], 0
    for k, token in enumerate(ids):
        if token in tags:
            if k > start:
                items.append((None, tokenizer.decode(ids[start:k])))
            items.append((tags[token], ""))
            start = k + 1
    if start < len(ids):
        items.append((None, tokenizer.decode(ids[start:])))
    position = 0

    def tag() -> str | None:
        return items[position][0] if position < len(items) else None

    def skip_text() -> str:
        nonlocal position
        if position < len(items) and items[position][0] is None:
            position += 1
            return items[position - 1][1]
        return ""

    if tag() == "<think>":
        position += 1
    skip_text()
    if position == len(items):
        return None
    if tag() != "</think>":
        return f"{tag()} in the reasoning, where </think> belongs"
    position += 1
    skip_text()
    while tag() == "<tool_call>":
        position += 1
        body = skip_text()
        if position == len(items):
            return None
        if tag() != "</tool_call>":
            return f"{tag()} inside a tool call, where </tool_call> belongs"
        position += 1
        if found := _call_violation(body, tools):
            return found
        if skip_text().strip():
            return "text after a tool call"
    if position == len(items):
        return None
    if tag() != "<|im_end|>":
        return f"{tag()} after the reasoning, where <tool_call> or <|im_end|> belongs"
    if position + 1 < len(items):
        return "tokens after <|im_end|>"
    return None


def format_violation(response: vf.Response, trace: vf.Trace) -> bool:
    # A relayed call returns no token ids, so there is nothing to check.
    if response.tokens is None:
        return False
    tools = {tool.name: tool.parameters for tool in trace.tools} or None
    found = violation([int(i) for i in response.tokens.completion_ids], tools)
    if found is None:
        return False
    logger.warning("rollout %s: format violation: %s", trace.id, found)
    trace.info["format_violation"] = found
    return True

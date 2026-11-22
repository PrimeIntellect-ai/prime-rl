"""Memory-turn helpers for the glyphbench verifiers integration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import verifiers as vf
from verifiers.types import SamplingArgs


@dataclass(frozen=True)
class MemoryExtraction:
    """Result of parsing the memory-turn assistant response."""
    memory: str
    parse_failed: bool


_MEMORY_RE = re.compile(
    r"<\s*memory\s*>(.*?)<\s*/\s*memory\s*>", re.DOTALL | re.IGNORECASE
)
_ACTION_RE = re.compile(
    r"<\s*action\s*>.*?<\s*/\s*action\s*>", re.DOTALL | re.IGNORECASE
)

ACTION_STOP = "</action>"
MEMORY_STOP = "</memory>"
_PROTOCOL_STOPS = {ACTION_STOP, MEMORY_STOP}


def extract_memory_update(text: str) -> MemoryExtraction:
    """Strict <memory>...</memory> extraction.

    Returns ``MemoryExtraction(memory=joined_tags, parse_failed=False)`` if at
    least one complete ``<memory>...</memory>`` tag is found (multiple tags
    are joined with a blank-line separator). Anything else — no tag, only an
    open tag, raw post-think text — yields ``MemoryExtraction(memory="",
    parse_failed=True)``.

    The caller (verifiers integration) keeps the previous memory on failure.
    """
    raw = text or ""
    matches = _MEMORY_RE.findall(raw)
    if not matches:
        if _ACTION_RE.search(raw):
            recovered = _strip_action_tags(raw).strip()
            if recovered:
                return MemoryExtraction(memory=recovered, parse_failed=False)
        return MemoryExtraction(memory="", parse_failed=True)
    return MemoryExtraction(memory="\n\n".join(m.strip() for m in matches), parse_failed=False)


def _strip_action_tags(text: str) -> str:
    return _ACTION_RE.sub("", text or "")


def build_memory_update_user(
    *,
    action_reasoning: str,
    action_chosen: str,
    parse_failed: bool,
    parse_failure_reason: str | None,
    forfeit_mode: str | None = None,
    forfeit_action_name: str | None = None,
    action_truncated: bool,
    reward: float,
    terminated: bool,
    truncated: bool,
    next_obs: str,
) -> vf.UserMessage:
    """Build the user message for the memory-update generation.

    Sections, in order:

      [Last Action]      — the model's ``<think>`` reasoning trace
                           (re-injected because the chat template strips
                           it from prior assistant turns) PLUS an Outcome
                           block reporting what the parser actually
                           applied (``Action applied`` / ``Parse status`` /
                           ``Output truncated``). The literal ``<action>``
                           tag is NOT repeated here — it is already in
                           the conversation prefix as
                           ``assistant_action_T.content`` and structured
                           as ``Action applied: NAME``.

      [Env Response]     — reward, terminated, truncated (episode-level).

      [Next Observation] — the new grid + legend + cleaned complementary
                           HUD + message produced by the env's response to
                           the action. This is the same observation content
                           the next action turn will receive. Surfacing it
                           here lets memory react to the action's actual
                           effect, which a sparse-reward signal alone
                           cannot convey.

      [Memory Update]    — write instruction with the synth/non-describe
                           guidance.
    """
    if parse_failed:
        if forfeit_mode == "noop":
            fallback = forfeit_action_name or "NOOP"
            parse_status = (
                f"FAILED — {parse_failure_reason or 'no <action> tag'} "
                f"(turn forfeited; env applied fallback action {fallback})"
            )
        else:
            parse_status = (
                f"FAILED — {parse_failure_reason or 'no <action> tag'} "
                "(turn forfeited; env state did not advance)"
            )
    else:
        parse_status = "ok"

    reasoning_clean = action_reasoning.strip() if action_reasoning else ""
    if reasoning_clean:
        reasoning_block = (
            "Reasoning that led to the action (re-injected because the "
            "chat template strips <think> from prior assistant turns):\n"
            "---\n"
            f"{reasoning_clean}\n"
            "---\n"
        )
    else:
        reasoning_block = "Reasoning: (none emitted)\n"

    last_action = (
        "[Last Action]\n"
        + reasoning_block
        + "Outcome:\n"
        + f"  Action applied: {action_chosen}\n"
        + f"  Parse status: {parse_status}\n"
        + f"  Output truncated: {str(bool(action_truncated)).lower()}"
    )

    env_response_block = (
        "[Env Response]\n"
        f"  Reward: {reward:+.3f}\n"
        f"  Terminated: {str(bool(terminated)).lower()}\n"
        f"  Truncated: {str(bool(truncated)).lower()}"
    )

    next_obs_clean = next_obs.rstrip() if next_obs else "(none)"
    next_obs_block = (
        "[Next Observation]\n"
        f"{next_obs_clean if next_obs_clean else '(none)'}"
    )

    instruction = (
        "[Memory Update]\n"
        "Update your memory based on the above: the action you took, the "
        "env's response, and the new observation. Reply ONLY with "
        "`<memory>...your concise updated memory...</memory>`. Anything "
        "outside the <memory> tag is discarded. Do not emit an <action> "
        "tag. Memory is for synthesis — causal facts you've established, "
        "plans, discoveries, retracted hypotheses, what the last action "
        "actually did vs. what you expected. Do NOT re-describe the grid: "
        "the next observation above is shown again at the next action "
        "turn, so duplicating its contents wastes the budget."
    )

    content = "\n\n".join(
        [last_action, env_response_block, next_obs_block, instruction]
    )
    return vf.UserMessage(content=content)


def action_reasoning_text(action_completion: list[Any]) -> str:
    """Extract just the action turn's ``<think>`` reasoning content.

    The ``<action>`` tag is intentionally excluded — it is already
    present in the conversation prefix as the assistant_action_T
    message's ``content`` (which the chat template does carry over to
    prior turns), and its parsed value is also reported as
    ``Action applied: NAME`` in the [Last Action] Outcome section.
    Repeating the literal tag in the [Last Action] re-injection would
    triple it in the memory call's input.

    For non-thinking instruct models, visible text before/around the final
    action tag is treated as reasoning after removing complete action tags.
    Returns ``""`` only if the model emitted no reasoning trace.
    """
    if not action_completion:
        return ""
    msg = action_completion[-1]
    reasoning = msg.get("reasoning_content")
    if reasoning:
        return reasoning.strip()
    content = msg.get("content", "") or ""
    if "</think>" in content:
        before = content.split("</think>", 1)[0].lstrip()
        if before.startswith("<think>"):
            before = before[len("<think>"):].lstrip("\n")
        return _strip_action_tags(before).strip()
    plain_reasoning = _strip_action_tags(content).strip()
    return plain_reasoning


def xml_stop_sampling_args(
    sampling_args: SamplingArgs | None,
    stop: str,
) -> SamplingArgs:
    """Return sampling args that stop immediately after a protocol XML tag.

    vLLM excludes stop strings by default, which would break the strict XML
    parser. ``include_stop_str_in_output`` preserves the closing tag in the
    response text and token stream while avoiding post-tag decode waste.
    """
    args: dict[str, Any] = dict(sampling_args or {})
    existing = args.get("stop")
    if existing is None:
        stops: list[str] = []
    elif isinstance(existing, str):
        stops = [existing]
    else:
        stops = [str(s) for s in existing]

    stops = [s for s in stops if s not in _PROTOCOL_STOPS]
    if stop not in stops:
        stops.append(stop)
    args["stop"] = stops

    extra_body = dict(args.get("extra_body") or {})
    extra_body["include_stop_str_in_output"] = True
    args["extra_body"] = extra_body
    return args


def memory_sampling_args(
    sampling_args: SamplingArgs | None,
    memory_update_max_tokens: int | None,
    *,
    stop_after_xml: bool = False,
) -> SamplingArgs | None:
    """Return sampling args for the memory-update generation.

    Always disables thinking-mode for the memory-update call when an
    `extra_body.chat_template_kwargs` slot is present (Qwen3/Qwen3.5).
    The action turn's RESPONSE FORMAT block conditions the model to emit
    ``<action>...</action>`` after ``</think>``; if we left thinking on for
    the memory update the model would default to that habit and our parser
    would store the action tag as the memory.
    """
    args: dict[str, Any] = dict(sampling_args or {})
    if memory_update_max_tokens is not None:
        if "max_completion_tokens" in args:
            args["max_completion_tokens"] = int(memory_update_max_tokens)
        elif "max_tokens" in args:
            args["max_tokens"] = int(memory_update_max_tokens)
        else:
            args["max_tokens"] = int(memory_update_max_tokens)
    extra_body = dict(args.get("extra_body") or {})
    chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_kwargs
    args["extra_body"] = extra_body
    if stop_after_xml:
        args = xml_stop_sampling_args(args, MEMORY_STOP)
    return args

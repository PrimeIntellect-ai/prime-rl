"""GlyphBench CLI: list/replay helpers.

Eval execution is delegated to `prime eval run` (prime-rl's standard
runner) — see eval/run_debug.sh / eval/run_full.sh for the canonical
invocations. The CLI here covers introspection + visualisation only.

Examples:
    glyphbench list-suites
    glyphbench list-envs --suite atari
    glyphbench replay path/to/runs --suite minigrid --pause
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import shlex
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

from glyphbench.envs import ALL_SUITES as ENV_SUITES
from glyphbench.verifiers_integration.memory import extract_memory_update
from glyphbench.verifiers_integration.parser import (
    GlyphbenchXMLParser,
    action_parse_region,
)

# Singleton parser instance used by the canonical eval pipeline; we route
# CLI replay action extraction through it so the displayed action matches
# what the verifiers eval actually scored.
_REPLAY_PARSER = GlyphbenchXMLParser()

ALL_SUITES = [suite for suite in ENV_SUITES if suite != "dummy"]


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def _discover_results_files(target: Path) -> list[Path]:
    """Return every results.jsonl under target. target is required."""
    if target.is_file() and target.name == "results.jsonl":
        return [target]
    if target.is_dir():
        return sorted(target.glob("**/results.jsonl"))
    return []


def _discover_pro_transcripts(target: Path) -> list[Path]:
    """Return every Pro-harness ``transcript_*.jsonl`` under target.

    Lets ``gb replay`` animate standalone pro_harness runs through the same
    renderer/--list/--pause path as verifiers ``results.jsonl`` rollouts.
    """
    if (
        target.is_file()
        and target.name.startswith("transcript_")
        and target.suffix == ".jsonl"
    ):
        return [target]
    if target.is_dir():
        return sorted(target.glob("**/transcript_*.jsonl"))
    return []


def _pro_run_meta(run_dir: Path) -> tuple[str, int | None]:
    """(env_id, base_seed) for a pro-harness run dir, from its config.json."""
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            d = json.loads(cfg.read_text())
            return str(d.get("task_id", "?")), d.get("seed")
        except (OSError, json.JSONDecodeError):
            pass
    name = run_dir.name
    slug = name.split("-azure-")[0].split("-openai-")[0]
    return (f"glyphbench/{slug}" if slug else "?"), None


def _pro_rollouts(paths: list[Path]) -> list[tuple[Path, dict]]:
    """Build verifiers-shaped rollout dicts from pro-harness transcripts so the
    existing replay renderer handles them unchanged (one action step per turn,
    matching the non-memory rollout shape)."""
    out: list[tuple[Path, dict]] = []
    for p in paths:
        rows: list[dict] = []
        try:
            with p.open() as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln:
                        rows.append(json.loads(ln))
        except (OSError, json.JSONDecodeError):
            continue
        if not rows:
            continue
        env_id, base_seed = _pro_run_meta(p.parent)
        try:
            ep_idx = int(p.stem.split("_")[-1])
        except ValueError:
            ep_idx = 0
        seed = (base_seed + ep_idx) if isinstance(base_seed, int) else None

        steps: list[dict] = []
        for row in rows:
            action = row.get("action", "") or ""
            forfeit = action == "FORFEIT"
            reasoning = (row.get("reasoning") or "").strip()
            output = (row.get("output") or "").strip()
            text = reasoning
            if output:
                text = (text + "\n\n" + output).strip() if text else output
            if action and not forfeit and "<action>" not in text.lower():
                text = (text + f"\n<action>{action}</action>").strip()
            obs = row.get("observation", "")
            mem_before = row.get("memory_before")
            # Surface the carried memory so the replay memory panel shows the
            # "previous memory" (read from a <memory> tag in the action user).
            action_user = (
                f"<memory>\n{mem_before}\n</memory>\n\n{obs}" if mem_before else obs
            )
            steps.append({
                "prompt": [{"role": "user", "content": action_user}],
                "completion": [{"role": "assistant", "content": text}],
                "reward": float(row.get("reward", 0.0) or 0.0),
                "is_truncated": False,
                "extras": {
                    "glyphbench_step_role": "action",
                    "action_chosen": "" if forfeit else action,
                    "parse_failed": forfeit,
                    "parse_failure_reason": "no_action" if forfeit else None,
                    "forfeit": forfeit,
                    "env_info": {
                        "raw_score": row.get("raw_score"),
                        "raw_score_label": row.get("raw_score_label"),
                        "num_achievements": row.get("achievements"),
                        "turn": row.get("turn"),
                    },
                },
            })
            # Synthesize a memory step so `gb replay` renders the scratchpad +
            # landmark memory panel (skipped on the terminal turn, which has no
            # memory update).
            mem_after = row.get("memory_after")
            if mem_after:
                mem_out = row.get("memory_output") or ""
                lookups = row.get("pending_lookups") or ""
                mem_user = (
                    "[Last Action]\n"
                    f"  Action applied: {action}\n\n"
                    "[Current Pro Memory]\n"
                    f"{mem_before or '(empty)'}\n\n"
                    "[Memory Update]\nUpdate the scratchpad and landmark map."
                )
                steps.append({
                    "prompt": [{"role": "user", "content": mem_user}],
                    "completion": [{"role": "assistant", "content": mem_out}],
                    "reward": float(row.get("reward", 0.0) or 0.0),
                    "is_truncated": False,
                    "extras": {
                        "glyphbench_step_role": "memory",
                        "memory_parse_failed": False,
                        "stored_memory": (
                            mem_after + (f"\n\n[Recalled next turn]\n{lookups}" if lookups else "")
                        ),
                    },
                })
        first_obs = rows[0].get("observation", "")
        rollout = {
            "info": {"env_id": env_id, "seed": seed},
            "reward": rows[-1].get("total_return"),
            "prompt": [
                {"role": "system", "content": ""},
                {"role": "user", "content": first_obs},
            ],
            "completion": [],
            "trajectory": steps,
        }
        out.append((p, rollout))
    return out


_USER_TURN_TRAILER_RE = re.compile(
    r"^(?:"
    r"Now emit your move as `<action>ACTION_NAME</action>`\.?"
    r"|Now reason briefly, then end with a final <action> tag containing "
    r"exactly one action name from (?:the \[Actions\] list|the system prompt's action list)\."
    r")$"
)
_CURRENT_OBSERVATION_RE = re.compile(r"^\[Current Observation\b")


def _extract_block(content: str, header: str) -> str | None:
    """Extract a `[Header]\n…\n[Next]` block from a rendered user turn.

    Stops when the block closes (next ``[Header]`` line) or when the
    user-turn trailer (``Now reason briefly…`` / legacy ``Now emit…``) is reached. The trailer
    sits outside any bracketed section, so without explicit handling
    it would bleed into the LAST extracted block (typically [Grid] or
    [Message] in turn-T action prompts).
    """
    lines = content.split("\n")
    try:
        i = lines.index(header)
    except ValueError:
        return None
    out: list[str] = []
    for ln in lines[i + 1:]:
        if ln.startswith("[") and ln.endswith("]"):
            break
        out.append(ln)
    # Trim trailing blanks AND a trailing user-turn trailer when the
    # block runs to end-of-content (no closing [Header] was hit).
    while out and (
        not out[-1].strip()
        or _USER_TURN_TRAILER_RE.match(out[-1].strip())
    ):
        out.pop()
    return "\n".join(out) if out else None


def _extract_block_last(content: str, header: str) -> str | None:
    """Extract the last occurrence of a bracketed block from ``content``."""
    lines = content.split("\n")
    indices = [i for i, ln in enumerate(lines) if ln == header]
    if not indices:
        return None
    i = indices[-1]
    out: list[str] = []
    for ln in lines[i + 1:]:
        if ln.startswith("[") and ln.endswith("]"):
            break
        out.append(ln)
    while out and (
        not out[-1].strip()
        or _USER_TURN_TRAILER_RE.match(out[-1].strip())
    ):
        out.pop()
    return "\n".join(out) if out else None


def _current_observation_region(content: str) -> str | None:
    """Return the current-observation slice of an action prompt.

    Memory-mode prompts can contain arbitrary model-written memory before the
    real observation. If that memory contains lines such as ``[HUD]`` or
    ``[Grid]``, first-match extraction shows stale user text instead of the
    turn being replayed. Current prompts carry a dedicated
    ``[Current Observation — turn T]`` marker, so prefer the last such section
    and fall back to legacy full-prompt extraction only when the marker is
    absent.
    """
    lines = content.split("\n")
    starts = [
        i for i, ln in enumerate(lines)
        if _CURRENT_OBSERVATION_RE.match(ln)
    ]
    if not starts:
        return None
    out: list[str] = []
    for ln in lines[starts[-1] + 1:]:
        if _USER_TURN_TRAILER_RE.match(ln.strip()):
            break
        out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) if out else ""


def _extract_observation_block(content: str, header: str) -> str | None:
    region = _current_observation_region(content)
    if region is not None:
        return _extract_block(region, header)
    return _extract_block_last(content, header)


def _extract_grid(content: str) -> str | None:
    return _extract_observation_block(content, "[Grid]")


def _extract_hud(content: str) -> str | None:
    return _extract_observation_block(content, "[HUD]")


def _extract_message(content: str) -> str | None:
    return _extract_observation_block(content, "[Message]")


def _extract_legend(content: str) -> str | None:
    region = _current_observation_region(content)
    if region is not None:
        prefix = content[: content.rfind("[Current Observation")]
        return _extract_block_last(prefix, "[Legend]")
    return _extract_block_last(content, "[Legend]")


_THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
_ACTION_RE_LOCAL = re.compile(
    r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", re.DOTALL | re.IGNORECASE
)
_ACTION_TAG_RE_LOCAL = re.compile(
    r"<\s*action\s*>.*?<\s*/\s*action\s*>", re.DOTALL | re.IGNORECASE
)
_ACTION_OPEN_TAG_RE_LOCAL = re.compile(r"<\s*action\s*>", re.IGNORECASE)


def _split_assistant(content: str) -> tuple[str, str]:
    """Split an assistant turn into (reasoning, action).

    Mirrors the strict eval parser so the replay panel shows exactly what
    the eval scored:

    * Reasoning is the segment ending at the LAST ``</think>``. Qwen3.5's
      chat template prefills ``<think>\\n``, so the stored content
      commonly starts mid-thinking with no opener; treat start-of-string
      as an implicit opener when only a closer is present.
    * Action is the LAST complete ``<action>…</action>`` tag content.
      Malformed content (unclosed tag, JSON, bare token) is no longer
      recovered — the eval would have forfeited those turns.
    """
    text = content or ""

    # ---- reasoning ----
    closes = list(_THINK_CLOSE_RE.finditer(text))
    if closes:
        last_close = closes[-1]
        opens_before = [
            m for m in _THINK_OPEN_RE.finditer(text)
            if m.end() <= last_close.start()
        ]
        start = opens_before[-1].end() if opens_before else 0
        think = text[start:last_close.start()].strip()
    else:
        think = ""

    # ---- action: use the strict XML regex directly — last complete
    # <action>NAME</action> wins; no malformed-content recovery since
    # the eval no longer accepts those formats either.
    xml_matches = _ACTION_RE_LOCAL.findall(text)
    action = xml_matches[-1].strip() if xml_matches else ""

    # Reasoning residual fallback — only fires when there was no </think>
    # at all (genuine leak, not chat-template prefill).
    if not think and not closes:
        residual = _ACTION_RE_LOCAL.sub("", text)
        residual = _THINK_OPEN_RE.sub("", residual).strip()
        if residual:
            think = residual

    return think, action


def _raw_action_output(content: str) -> str:
    """Return the literal text replay should show as the raw action output.

    For native-thinking responses, this is the text after the final
    ``</think>`` boundary. For visible-reasoning instruct responses there is
    no reliable structural boundary, so prefer the last action-shaped fragment
    and fall back to the final non-empty line.
    """
    text = content or ""
    closes = list(_THINK_CLOSE_RE.finditer(text))
    if closes:
        return text[closes[-1].end():].strip()

    candidates: list[tuple[int, str]] = [
        (match.start(), match.group(0).strip())
        for match in _ACTION_TAG_RE_LOCAL.finditer(text)
    ]
    complete_tag_starts = {start for start, _ in candidates}
    for match in _ACTION_OPEN_TAG_RE_LOCAL.finditer(text):
        if match.start() not in complete_tag_starts:
            candidates.append((match.start(), text[match.start():].strip()))
    if candidates:
        return max(candidates, key=lambda candidate: candidate[0])[1]

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return nonempty_lines[-1] if nonempty_lines else ""


def _reasoning_kind(content: str) -> str:
    """Return ``native``, ``visible``, or ``none`` for replay labelling.

    ``visible`` means plain assistant text before the action tag, which is
    expected for non-native-thinking instruct models.
    """
    think, _ = _split_assistant(content)
    if not think:
        return "none"
    if _THINK_CLOSE_RE.search(content or ""):
        return "native"
    return "visible"


@functools.cache
def _spec_for_env_id(env_id: str) -> tuple[object, str] | None:
    """Look up (ActionSpec, noop_action_name) for an env id.

    Cached; returns ``None`` when the env can't be instantiated (e.g.
    optional native dep missing on this machine). Used by
    ``_resolve_action`` to canonicalise the displayed action through
    the same path the eval used.
    """
    try:
        from glyphbench.core.registry import make_env
        env = make_env(env_id)
        return env.action_spec, env.noop_action_name
    except Exception:
        return None


def _resolve_action(text: str, env_id: str | None) -> tuple[str, bool]:
    """Return (display_name, parse_failed) for an assistant message.

    When the env can be loaded, this calls ``GlyphbenchXMLParser.parse_action``
    with the env's ActionSpec — exactly the entry point the eval used —
    so the displayed action is the canonicalised name (or the env's
    noop on parse failure). Otherwise falls back to ``_split_assistant``
    which is the same parser fallback chain without spec validation.
    """
    raw = text or ""
    if env_id:
        spec_info = _spec_for_env_id(env_id)
        if spec_info is not None:
            spec, noop = spec_info
            try:
                _idx, canonical, failed, _reason = _REPLAY_PARSER.parse_action(
                    raw, spec, noop=noop,
                )
                return canonical, failed
            except Exception:
                pass
    _, action = _split_assistant(raw)
    return action, not bool(action)


def _action_parse_details(text: str, env_id: str | None) -> dict[str, object]:
    raw = text or ""
    if env_id:
        spec_info = _spec_for_env_id(env_id)
        if spec_info is not None:
            spec, noop = spec_info
            try:
                _idx, canonical, failed, reason = _REPLAY_PARSER.parse_action(
                    raw, spec, noop=noop,
                )
                return {
                    "action": canonical,
                    "parse_failed": failed,
                    "parse_failure_reason": reason,
                    "valid_actions": spec.names,
                }
            except Exception as exc:
                return {
                    "action": "",
                    "parse_failed": True,
                    "parse_failure_reason": f"replay_parser_error: {exc}",
                    "valid_actions": (),
                }
    _, action = _split_assistant(raw)
    return {
        "action": action,
        "parse_failed": not bool(action),
        "parse_failure_reason": "no_action_tag" if not action else None,
        "valid_actions": (),
    }


def _last_action_match_details(text: str) -> dict[str, object]:
    matches = list(_ACTION_RE_LOCAL.finditer(text or ""))
    if not matches:
        return {"count": 0}
    match = matches[-1]
    return {
        "count": len(matches),
        "span": match.span(),
        "candidate_span": match.span(1),
        "whole": match.group(0),
        "candidate": match.group(1),
    }


def _clip_middle_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = f"\n[... {len(text) - max_chars} chars clipped ...]\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def _codepoint_summary(text: str, max_chars: int = 80) -> str:
    if not text:
        return "(empty)"
    sample = text
    if len(sample) > max_chars:
        head = max_chars // 2
        tail = max_chars - head
        sample = sample[:head] + sample[-tail:]
    return " ".join(f"U+{ord(ch):04X}" for ch in sample)


def _format_parse_details(prefix: str, details: dict[str, object]) -> list[str]:
    failed = bool(details.get("parse_failed"))
    action = str(details.get("action") or "(no action parsed)")
    reason = details.get("parse_failure_reason")
    lines = [
        f"{prefix} parse status: {'parse failed' if failed else 'ok'}",
        f"{prefix} parsed action: {action}",
    ]
    if reason:
        lines.append(f"{prefix} failure reason: {reason}")
    return lines


def _action_pager_text(
    content: str,
    env_id: str | None,
    action_extras: dict | None = None,
) -> str:
    """Return the full text shown when pause-mode expands the action panel."""
    current_details = _action_parse_details(content, env_id)
    raw_action = _raw_action_output(content).strip()
    raw_details = _action_parse_details(raw_action, env_id)
    parse_region = action_parse_region(content)
    match = _last_action_match_details(parse_region)
    ignored_full_match = (
        _last_action_match_details(content)
        if parse_region != (content or "")
        else {"count": 0}
    )
    raw_match = _last_action_match_details(raw_action)

    lines: list[str] = ["ACTION (current turn)", ""]
    lines.extend(_format_parse_details("current parser", current_details))
    lines.append("")
    lines.extend(_format_parse_details("raw-region parser", raw_details))

    valid_actions = current_details.get("valid_actions") or raw_details.get("valid_actions")
    if valid_actions:
        lines.extend(["", "valid actions: " + ", ".join(str(a) for a in valid_actions)])

    if action_extras:
        keys = ("parse_failed", "parse_failure_reason", "action_chosen", "forfeit")
        stored = [
            f"  {key}: {action_extras.get(key)!r}"
            for key in keys
            if key in action_extras
        ]
        if stored:
            lines.extend(["", "STORED EVAL EXTRAS", *stored])
            if (
                action_extras.get("parse_failed") is True
                and not bool(current_details.get("parse_failed"))
            ):
                lines.append(
                    "  diagnosis: stored result was scored by an older or "
                    "different parser than the current replay parser."
                )

    lines.extend(["", "=" * 60, "", "PARSER MATCH ON CURRENT ACTION REGION", ""])
    count = int(match.get("count") or 0)
    lines.append(f"match count: {count}")
    if count:
        candidate = str(match.get("candidate") or "")
        lines.extend([
            f"match span: {match.get('span')}",
            f"candidate span: {match.get('candidate_span')}",
            "parser candidate repr:",
            _clip_middle_chars(repr(candidate), 1600),
            "parser candidate codepoints:",
            _codepoint_summary(candidate),
        ])
        if "<" in candidate:
            lines.append(
                "diagnosis: parser candidate contains '<'; the current action "
                "region still contains a malformed nested action tag."
            )
    else:
        lines.append("diagnosis: no complete <action>...</action> tag matched.")

    ignored_count = int(ignored_full_match.get("count") or 0)
    if ignored_count:
        ignored_candidate = str(ignored_full_match.get("candidate") or "")
        lines.extend([
            "",
            "=" * 60,
            "",
            "IGNORED FULL-TEXT MATCH (PRE-THINK INCLUDED)",
            "",
            f"match count: {ignored_count}",
            f"match span: {ignored_full_match.get('span')}",
            f"candidate span: {ignored_full_match.get('candidate_span')}",
            "candidate repr:",
            _clip_middle_chars(repr(ignored_candidate), 1600),
        ])
        if "<" in ignored_candidate:
            lines.append(
                "diagnosis: a literal <action> before the final action would "
                "have consumed the final </action> under the old full-text "
                "parser; current parsing ignores it because it is before "
                "</think>."
            )

    raw_count = int(raw_match.get("count") or 0)
    lines.extend(["", "=" * 60, "", "PARSER MATCH ON RAW ACTION OUTPUT", ""])
    lines.append(f"match count: {raw_count}")
    if raw_count:
        raw_candidate = str(raw_match.get("candidate") or "")
        lines.extend([
            f"candidate repr: {repr(raw_candidate)}",
            "candidate codepoints:",
            _codepoint_summary(raw_candidate),
        ])

    if bool(current_details.get("parse_failed")) and not bool(raw_details.get("parse_failed")):
        lines.extend([
            "",
            "diagnosis: raw output parses cleanly, but the full assistant text "
            "does not. Inspect the current action-region candidate above.",
        ])

    lines.extend(["", "=" * 60, "", "RAW ACTION OUTPUT", "", raw_action or "(empty)"])
    return "\n".join(lines)


def _clip_to_lines(
    text: str,
    max_lines: int,
    *,
    mode: str = "tail",
    max_line_width: int = 0,
) -> str:
    """Clip a text block to at most ``max_lines`` logical lines.

    ``mode`` is one of:
      * ``"tail"``   — keep the last lines (best for chain-of-thought,
        where the conclusion lives at the end).
      * ``"head"``   — keep the first lines.
      * ``"middle"`` — keep first half + last half with a "lines clipped"
        marker between them.

    When clipping happens the first/last/middle line of the result is a
    ``[... N lines clipped ...]`` marker, so the caller always knows how
    much was hidden. ``max_line_width`` truncates each LOGICAL line to
    that many characters (a defence against single-line walls of text
    that would otherwise wrap into many visual rows inside a Rich panel
    and overflow the layout). ``0`` disables per-line truncation.
    """
    raw = text or ""
    if max_lines <= 0:
        return ""
    lines = raw.splitlines()
    if max_line_width and max_line_width > 4:
        cut = max_line_width - 1
        lines = [
            (ln[:cut] + "…") if len(ln) > max_line_width else ln
            for ln in lines
        ]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    dropped = len(lines) - (max_lines - 1)
    marker = f"[… {dropped} lines clipped …]"
    if mode == "head":
        return "\n".join(lines[: max_lines - 1] + [marker])
    if mode == "middle":
        keep = max_lines - 1
        head = keep // 2
        tail = keep - head
        return "\n".join(lines[:head] + [marker] + (lines[-tail:] if tail else []))
    return "\n".join([marker] + lines[-(max_lines - 1):])


def _model_from_jsonl_path(path: Path) -> str | None:
    """Recover the HF model id from `<root>/.../glyphbench--<owner>--<rest>/<hash>/results.jsonl`."""
    meta = path.parent / "metadata.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text()).get("model")
            if isinstance(m, str) and m:
                return m
        except (OSError, json.JSONDecodeError):
            pass
    # Pro-harness runs record the model in config.json next to the transcript.
    cfg = path.parent / "config.json"
    if cfg.exists():
        try:
            m = json.loads(cfg.read_text()).get("model")
            if isinstance(m, str) and m:
                return m
        except (OSError, json.JSONDecodeError):
            pass
    for part in path.parts:
        if part.startswith("glyphbench--"):
            bits = part.split("--")
            if len(bits) >= 3:
                return "/".join([bits[1], "-".join(bits[2:])])
    return None


def _parse_info(info_field: object) -> dict:
    if isinstance(info_field, dict):
        return info_field
    if isinstance(info_field, str):
        try:
            return json.loads(info_field)
        except json.JSONDecodeError:
            return {}
    return {}


def _format_score_value(value: object) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def _nethack_raw_score(
    env_id: str | None,
    action_extras: dict | None = None,
    *,
    hud: str | None = None,
    message: str | None = None,
) -> str | None:
    if not (env_id or "").startswith("glyphbench/nethack-"):
        return None

    env_info = (action_extras or {}).get("env_info")
    if isinstance(env_info, dict):
        for key in ("nethack_score", "score"):
            formatted = _format_score_value(env_info.get(key))
            if formatted is not None:
                return formatted

    if message and "Points" in message:
        match = re.search(r"(?m)^\s*\d+\s+(\d+)\s+\S+", message)
        if match:
            return match.group(1)

    if hud:
        match = re.search(r"\bScore:\s*(-?\d+)", hud)
        if match:
            return match.group(1)
    return None


def _suite_of(env_id: str) -> str:
    return env_id.split("/", 1)[-1].split("-", 1)[0]


def _iter_rollouts(files: list[Path]) -> list[tuple[Path, dict]]:
    """Yield (results_file, rollout_dict) for every parseable row."""
    out: list[tuple[Path, dict]] = []
    for f in files:
        try:
            with f.open() as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        row = json.loads(ln)
                        if isinstance(row, dict):
                            out.append((f, row))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return out


def _filter_rollouts(
    rollouts: list[tuple[Path, dict]],
    *,
    envs: list[str] | None,
    suites: list[str] | None,
    models: list[str] | None,
    seeds: list[int] | None,
) -> list[tuple[Path, dict, dict]]:
    """Return [(file, rollout, info)] passing every (AND-combined) filter."""
    out: list[tuple[Path, dict, dict]] = []
    env_set = set(envs) if envs else None
    suite_set = set(suites) if suites else None
    seed_set = set(seeds) if seeds else None
    model_set = set(models) if models else None
    for f, r in rollouts:
        info = _parse_info(r.get("info"))
        env_id = info.get("env_id", "")
        if env_set is not None and env_id not in env_set:
            continue
        if suite_set is not None and _suite_of(env_id) not in suite_set:
            continue
        if seed_set is not None:
            try:
                if int(info.get("seed")) not in seed_set:
                    continue
            except (TypeError, ValueError):
                continue
        if model_set is not None:
            model = _model_from_jsonl_path(f)
            if model not in model_set:
                continue
        out.append((f, r, info))
    return out


def _content_from_role(messages: list[dict], role: str) -> str:
    msg = next((m for m in messages if m.get("role") == role), None)
    return (msg or {}).get("content") or ""


def _last_message_from_role(messages: list[dict], role: str) -> dict | None:
    return next((m for m in reversed(messages) if m.get("role") == role), None)


def _last_content_from_role(messages: list[dict], role: str) -> str:
    msg = _last_message_from_role(messages, role)
    return (msg or {}).get("content") or ""


def _is_memory_update_user(msg: dict) -> bool:
    """Detect the memory-update user message inside a rollout's completion.

    Post-rework (commit fc024a3): memory_user starts with [Last Action]
    and the [Memory Update] block is the FOURTH section, not the first.
    Old trajectories saved before the rework still had [Memory Update]
    as the leading marker — keep that fallback so historical eval JSONLs
    replay correctly.
    """
    if msg.get("role") != "user":
        return False
    content = (msg.get("content") or "").lstrip()
    return content.startswith("[Last Action]") or content.startswith("[Memory Update]")


def _extract_prompt_memory(content: str) -> str:
    match = re.search(
        r"<\s*memory\s*>(.*?)<\s*/\s*memory\s*>",
        content or "",
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _attach_memory_update(turn: dict, memory_user: dict, memory_assistant: dict) -> None:
    memory_text = extract_memory_update(memory_assistant.get("content") or "").memory
    turn["memory"] = {
        "previous_memory": _extract_prompt_memory(turn.get("user") or ""),
        "stored_memory": memory_text,
        "memory_update_prompt": [memory_user],
        "memory_update_response": [memory_assistant],
    }


def _memory_user_from_prompt(prompt: list[dict]) -> dict | None:
    for msg in reversed(prompt):
        if _is_memory_update_user(msg):
            return msg
    return _last_message_from_role(prompt, "user")


def _build_role_trajectory_turns(rollout: dict) -> list[dict] | None:
    """Build replay turns from current trajectory action/memory step roles."""
    trajectory = rollout.get("trajectory") or []
    turns: list[dict] = []
    pending: dict | None = None
    saw_role = False

    for step in trajectory:
        extras = step.get("extras") or {}
        role = extras.get("glyphbench_step_role")
        if role == "action":
            saw_role = True
            if pending is not None:
                turns.append(pending)
            prompt = step.get("prompt") or []
            completion = step.get("completion") or []
            pending = {
                "user": _last_content_from_role(prompt, "user"),
                "assistant": _content_from_role(completion, "assistant"),
                "memory": None,
            }
        elif role == "memory":
            saw_role = True
            if pending is None:
                continue
            prompt = step.get("prompt") or []
            completion = step.get("completion") or []
            memory_user = _memory_user_from_prompt(prompt)
            memory_assistant = _last_message_from_role(completion, "assistant") or {}
            stored_memory = extras.get("stored_memory")
            if not isinstance(stored_memory, str):
                stored_memory = extract_memory_update(
                    memory_assistant.get("content") or ""
                ).memory
            pending["memory"] = {
                "previous_memory": _extract_prompt_memory(pending.get("user") or ""),
                "stored_memory": stored_memory,
                "memory_update_prompt": [memory_user] if memory_user else [],
                "memory_update_response": [memory_assistant] if memory_assistant else [],
            }
            turns.append(pending)
            pending = None

    if pending is not None:
        turns.append(pending)
    return turns if saw_role and turns else None


def _build_legacy_memory_turns(rollout: dict) -> list[dict] | None:
    trajectory = rollout.get("trajectory") or []
    turns: list[dict] = []
    saw_memory = False
    for step in trajectory:
        memory = (step.get("extras") or {}).get("glyphbench_memory")
        if not memory:
            continue
        saw_memory = True
        prompt = step.get("prompt") or []
        action_response = memory.get("action_response") or step.get("completion") or []
        turns.append(
            {
                "user": _content_from_role(prompt, "user"),
                "assistant": _content_from_role(action_response, "assistant"),
                "memory": memory,
            }
        )
    return turns if saw_memory else None


def _build_turns(rollout: dict) -> list[dict]:
    """Walk the rollout into per-environment-turn display records."""
    role_turns = _build_role_trajectory_turns(rollout)
    if role_turns is not None:
        return role_turns
    legacy_memory_turns = _build_legacy_memory_turns(rollout)
    if legacy_memory_turns is not None:
        return legacy_memory_turns

    prompt_msgs = rollout.get("prompt") or []
    completion = rollout.get("completion") or []
    initial_user = next((m for m in prompt_msgs if m.get("role") == "user"), None)
    turns: list[dict] = []
    i = 0
    if initial_user is not None and completion and completion[0].get("role") == "assistant":
        turns.append(
            {
                "user": initial_user.get("content") or "",
                "assistant": completion[0].get("content") or "",
                "memory": None,
            }
        )
        i = 1
    while i < len(completion):
        if completion[i].get("role") != "user":
            i += 1
            continue
        if _is_memory_update_user(completion[i]):
            if (
                turns
                and i + 1 < len(completion)
                and completion[i + 1].get("role") == "assistant"
            ):
                _attach_memory_update(turns[-1], completion[i], completion[i + 1])
                i += 2
            else:
                i += 1
            continue
        u = completion[i]
        a = completion[i + 1] if i + 1 < len(completion) and completion[i + 1].get("role") == "assistant" else None
        turns.append(
            {
                "user": u.get("content") or "",
                "assistant": (a or {}).get("content") or "",
                "memory": None,
            }
        )
        i += 2 if a is not None else 1
    return turns


def _scale_grid(grid: str, x: int, y: int) -> str:
    """Visually scale a glyph grid: each cell occupies `x` terminal columns
    (the glyph plus `x-1` trailing spaces) and `y` terminal rows (each line
    repeated `y` times). x=2,y=1 makes the cell roughly square in a typical
    2:1 height/width terminal font without doubling the glyph itself —
    `@` becomes `@ ` rather than `@@`."""
    if x <= 1 and y <= 1:
        return grid
    scaled_lines: list[str] = []
    pad = " " * max(0, x - 1)
    for line in grid.split("\n"):
        if x > 1:
            line = "".join(ch + pad for ch in line)
        for _ in range(max(1, y)):
            scaled_lines.append(line)
    return "\n".join(scaled_lines)


def _grid_display_for_replay(
    grid: str,
    *,
    console_width: int,
) -> tuple[str, int, int]:
    """Return replay grid text plus left/right layout ratios.

    Small grids keep the 2-column visual scale. Wider layouts, such as
    MiniHack corridor R5, stay unscaled and get more horizontal space so
    Rich does not wrap rows into visually broken maps.
    """
    raw_cols = max((len(line) for line in grid.splitlines()), default=0)

    def left_content_width(left_ratio: int, right_ratio: int) -> int:
        total = max(1, left_ratio + right_ratio)
        # Approximate panel border + padding overhead. The final Text is
        # also no-wrap/crop, so this only decides scaling and column ratios.
        return max(1, int(console_width * (left_ratio / total)) - 4)

    if raw_cols * 2 <= left_content_width(2, 3):
        return _scale_grid(grid, 2, 1), 2, 3
    if raw_cols <= left_content_width(2, 3):
        return grid, 2, 3
    if raw_cols <= left_content_width(3, 2):
        return grid, 3, 2
    return grid, 4, 1


def _render_turn_line(
    *,
    turn: int,
    grid: str,
    hud: str | None = None,
    message: str | None = None,
    action_chosen: str,
    raw_action_output: str | None = None,
    raw_score: str | None = None,
    reward: float,
    is_truncated: bool,
    memory_parse_failed: bool,
    step_role: str,
) -> str:
    """Compose a single-turn display line with failure-mode chips.

    Returns a short multi-line string of the form::

        (turn N) [chip1] [chip2]
        <grid>
        optional [HUD] / [Message]
        chose ACTION → reward R

    Chips:
      ``[forfeit]``       — action turn parse failed; env was not stepped.
      ``[trunc-action]``  — action completion was truncated (role="action").
      ``[trunc-memory]``  — memory completion was truncated (role="memory").
      ``[mem-parse-fail]``— memory turn emitted no ``<memory>`` tag.

    Used both by the plain-text fallback renderer and as a testable unit
    for the chip-presence assertions.
    """
    chips: list[str] = []
    if action_chosen == "FORFEIT":
        chips.append("[forfeit]")
    if is_truncated and step_role == "action":
        chips.append("[trunc-action]")
    if is_truncated and step_role == "memory":
        chips.append("[trunc-memory]")
    if memory_parse_failed:
        chips.append("[mem-parse-fail]")
    chip_str = " ".join(chips)
    suffix = f" {chip_str}" if chip_str else ""
    if action_chosen == "FORFEIT":
        action_part = "chose FORFEIT (parse failed) → reward 0"
    else:
        action_part = f"chose {action_chosen} → reward {reward:+.3f}"
    if raw_score is not None:
        action_part += f"    NetHack score {raw_score}"
    lines = [f"(turn {turn}){suffix}", grid]
    if hud:
        lines.extend(["[HUD]", hud])
    if message:
        lines.extend(["[Message]", message])
    lines.append(action_part)
    if raw_action_output is not None:
        raw = raw_action_output.strip() or "(empty)"
        lines.append(f"raw action output: {raw}")
    return "\n".join(lines)


def _render_rollout_rich(
    rollout: dict,
    info: dict,
    model_id: str,
    delay: float,
    reasoning_lines: int | None = None,
    pause: bool = False,
) -> None:
    """Render one rollout in a clean, in-place fullscreen layout: header
    bar at top, system prompt panel underneath (collapsible), then a per-turn
    composite (grid left, reasoning + action + env feedback right). Each turn
    fully replaces the previous frame — no scrollback churn."""
    import time

    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    turns = _build_turns(rollout)
    if not turns:
        console.print(Text("(no turns to render)", style="red"))
        return

    # Pre-index trajectory steps for per-turn chip data (action is_truncated,
    # memory is_truncated, memory_parse_failed).  Mirrors the logic in the
    # plain-text fallback path so both paths surface the same chips.
    # When an action step has no memory partner (non-memory mode or trailing
    # step) it is flushed into per_turn_extras with an empty memory dict.
    _rich_per_turn_extras: list[tuple[dict, dict]] = []
    _rich_a: dict = {}
    for _step in rollout.get("trajectory") or []:
        _se = (_step.get("extras") or {})
        _role = _se.get("glyphbench_step_role", "action")
        if _role == "action":
            if _rich_a:
                _rich_per_turn_extras.append((_rich_a, {}))
            _rich_a = {
                "is_truncated": bool(_step.get("is_truncated")),
                "action_chosen": _se.get("action_chosen", ""),
                "reward": _step.get("reward"),
                "env_info": _se.get("env_info") or {},
            }
        elif _role == "memory":
            _rich_per_turn_extras.append((
                _rich_a,
                {
                    "memory_parse_failed": bool(_se.get("memory_parse_failed")),
                    "is_truncated": bool(_step.get("is_truncated")),
                },
            ))
            _rich_a = {}
    if _rich_a:
        _rich_per_turn_extras.append((_rich_a, {}))

    console_h = console.size.height or 40
    if reasoning_lines is None:
        reasoning_cap = max(16, console_h // 2)
    elif reasoning_lines <= 0:
        reasoning_cap = 1_000_000
    else:
        reasoning_cap = reasoning_lines
    memory_cap = max(5, console_h // 6)
    sys_cap = max(4, console_h // 8)
    line_width_cap = 200

    # rollout-level flags we'll surface in the header on every frame
    overall_truncated = bool(
        rollout.get("is_truncated")
        or rollout.get("truncated")
        or rollout.get("episode_truncated_max_turns_rate")
        or rollout.get("truncated_flag")  # legacy key from pre-P3 rollout files
    )
    overall_completed = bool(
        rollout.get("is_completed")
        or rollout.get("terminated")
        or rollout.get("episode_terminated_rate")
        or rollout.get("terminated_flag")  # legacy key from pre-P3 rollout files
    )
    overall_pf_rate = rollout.get("forfeit_rate") or rollout.get("parse_failure_rate")
    stop_cond = rollout.get("stop_condition")
    reward = rollout.get("reward")
    metrics = rollout.get("metrics") or {}
    xml_ok = rollout.get("xml_format_reward", metrics.get("xml_format_reward"))

    sys_text = ""
    for m in rollout.get("prompt") or []:
        if m.get("role") == "system":
            sys_text = (m.get("content") or "").rstrip()
            break

    def render_frame(t_idx: int) -> Layout:
        turn = turns[t_idx - 1]
        u_content = turn["user"]
        a_content = turn["assistant"]
        memory = turn.get("memory")
        raw_grid = _extract_grid(u_content) or "(no [Grid] block in this turn)"
        grid, left_ratio, right_ratio = _grid_display_for_replay(
            raw_grid,
            console_width=console.size.width,
        )
        hud = _extract_hud(u_content) or ""
        legend = _extract_legend(u_content) or ""
        message = _extract_message(u_content) or ""
        reward_block = _extract_block(u_content, "[Reward]") or ""
        status = _extract_block(u_content, "[Status]") or ""
        think, _ = _split_assistant(a_content)
        reasoning_kind = _reasoning_kind(a_content)
        # Action is resolved through the canonical eval parser so the
        # displayed name matches what the eval scored (canonicalised
        # via ActionSpec when the env can be loaded; on parse failure
        # the env's noop name is shown, exactly as in eval rollouts).
        action, action_parse_failed = _resolve_action(a_content, info.get("env_id"))
        raw_action = _raw_action_output(a_content)

        # per-turn flags
        a_text = a_content or ""
        strict_action = bool(_ACTION_RE_LOCAL.search(a_text))
        max_tokens_hit = a_text and not strict_action and len(a_text) > 1500

        # Trajectory-level extras for this turn (chips that require the
        # stored TrajectoryStep data rather than just re-parsing the text).
        _t_action_extras, _t_mem_extras = (
            _rich_per_turn_extras[t_idx - 1]
            if t_idx - 1 < len(_rich_per_turn_extras)
            else ({}, {})
        )
        trunc_action = bool(_t_action_extras.get("is_truncated"))
        trunc_memory = bool(_t_mem_extras.get("is_truncated"))
        mem_parse_fail = bool(_t_mem_extras.get("memory_parse_failed"))
        turn_reward = _t_action_extras.get("reward")
        raw_score = _nethack_raw_score(
            info.get("env_id"),
            _t_action_extras,
            hud=hud,
            message=message,
        )

        flags: list[Text] = []
        # [forfeit] replaces the legacy "PARSE FAIL" chip so users see one
        # chip per failure mode rather than two overlapping labels.
        if action_parse_failed:
            flags.append(Text(" [forfeit] ", style="bold white on red"))
        elif not strict_action:
            flags.append(Text(" parse-fallback ", style="bold black on yellow"))
        if reasoning_kind == "visible":
            flags.append(Text(" visible reasoning ", style="bold grey70 on grey23"))
        if trunc_action:
            flags.append(Text(" [trunc-action] ", style="bold black on bright_yellow"))
        elif max_tokens_hit:
            flags.append(Text(" likely token-truncated ", style="bold black on bright_yellow"))
        if trunc_memory:
            flags.append(Text(" [trunc-memory] ", style="bold black on bright_yellow"))
        if mem_parse_fail:
            flags.append(Text(" [mem-parse-fail] ", style="bold black on bright_magenta"))

        # ---- header bar ----
        hdr = Table.grid(expand=True, padding=(0, 1))
        hdr.add_column(justify="left", ratio=2)
        hdr.add_column(justify="right", ratio=1)
        left_pieces = [
            Text.assemble((model_id, "bold cyan"), " · ",
                          (info.get("env_id", "?"), "bold white")),
        ]
        if flags:
            for f in flags:
                left_pieces.append(Text(" ", end=""))
                left_pieces.append(f)
        right_bits: list[object] = [
            ("turn ", "dim"), (f"{t_idx}/{len(turns)}", "bold magenta"),
            "   ", ("seed=", "dim"), (str(info.get("seed", "?")), "yellow"),
        ]
        if raw_score is not None:
            right_bits.extend([
                "   ", ("NetHack score=", "dim"), (raw_score, "bold yellow"),
            ])
        right_bits.extend([
            "   ", ("episodic reward=", "dim"), (f"{reward}", "bold green"),
            "   ", ("xml_ok=", "dim"), (f"{xml_ok:.2f}" if isinstance(xml_ok, (int, float)) else "?", "cyan"),
            "   ", ("pf_rate=", "dim"), (f"{overall_pf_rate:.2f}" if isinstance(overall_pf_rate, (int, float)) else "?", "yellow"),
        ])
        right_pieces = Text.assemble(*right_bits)
        hdr.add_row(Group(*left_pieces), right_pieces)

        rollout_state = []
        if overall_completed:
            rollout_state.append(Text(" terminated ", style="bold black on green"))
        if overall_truncated:
            rollout_state.append(Text(" truncated ", style="bold black on red"))
        if stop_cond:
            rollout_state.append(Text(f" stop={stop_cond} ", style="bold black on cyan"))
        if rollout_state:
            sub = Table.grid(expand=False, padding=(0, 1))
            for _s in rollout_state:
                sub.add_column()
            sub.add_row(*rollout_state)
            header_block = Group(hdr, sub)
        else:
            header_block = hdr

        # Per-turn errors that should be highlighted in red on the
        # affected panel. ``action_parse_failed`` is authoritative — it
        # comes from the same parser the eval used, with spec validation
        # when available.
        action_failed = action_parse_failed
        action_fallback = bool(action) and not strict_action and not action_parse_failed
        think_absent = not think and not strict_action  # nothing parseable at all
        turn_has_error = action_failed or think_absent or max_tokens_hit

        # ---- left: grid (red border if this turn has a generation error) ----
        grid_border = "red" if turn_has_error else "bright_white"
        grid_panel = Panel(
            Text(grid, style="bright_white", overflow="crop", no_wrap=True),
            title="grid" + ("  (turn error)" if turn_has_error else ""),
            title_align="left",
            border_style=grid_border,
            padding=(0, 1),
        )

        # ---- right: HUD + reasoning + action + env feedback (sized so
        # nothing in the column overflows the terminal). Each entry is
        # (panel, fixed-row-budget). The text inside is pre-clipped to
        # those budgets via _clip_to_lines, with mode="tail" for
        # reasoning + memory (the conclusion / latest update is what the
        # viewer cares about).
        right_entries: list[tuple[object, int]] = []
        if turn_has_error:
            err_msgs: list[str] = []
            if action_failed:
                err_msgs.append("no <action> parsed")
            elif action_fallback:
                err_msgs.append("action parsed via bare-name fallback")
            if think_absent:
                err_msgs.append("no reasoning + no action")
            if max_tokens_hit:
                err_msgs.append("response likely truncated by max_tokens")
            right_entries.append((
                Panel(
                    Text("\n".join(err_msgs), style="bold white on red",
                         overflow="fold"),
                    title="turn errors", title_align="left",
                    border_style="red", padding=(0, 1),
                ),
                len(err_msgs) + 2,
            ))
        if memory and memory.get("previous_memory"):
            prev_txt = _clip_to_lines(
                str(memory.get("previous_memory", "")).strip(),
                memory_cap, mode="tail", max_line_width=line_width_cap,
            )
            right_entries.append((
                Panel(Text(prev_txt, style="cyan", overflow="fold"),
                      title="previous memory", title_align="left",
                      border_style="cyan", padding=(0, 1)),
                memory_cap + 2,
            ))
        # Split the HUD into a dedicated `step` panel (the `Step: T / N`
        # progress indicator that every env now emits) and a residual
        # HUD panel (HP / food / score / etc.). Keeping them separate
        # lets the viewer track turn budget at a glance without having
        # to scan the rest of the HUD line.
        step_text = ""
        hud_residual = hud.strip() if hud else ""
        if hud_residual:
            m_step = re.search(r"Step:\s*\d+\s*/\s*\d+", hud_residual)
            if m_step:
                step_text = m_step.group(0)
                hud_residual = (
                    hud_residual[: m_step.start()]
                    + hud_residual[m_step.end():]
                )
                hud_residual = re.sub(r"\s{2,}", "    ", hud_residual)
                hud_residual = hud_residual.strip(" \t\n,;|")
        if step_text:
            right_entries.append((
                Panel(Text(step_text, style="bold yellow"),
                      title="step", title_align="left",
                      border_style="yellow", padding=(0, 1)),
                3,
            ))
        # HUD + legend share a row (HUD on the left, legend on the right)
        # so neither consumes a dedicated band of vertical space. The
        # legend gets the wider half because per-glyph descriptions are
        # denser than the HUD's space-separated key:value tokens, which
        # tolerate wrapping.
        hud_panel: Panel | None = None
        hud_rows = 0
        if hud_residual:
            hud_lines = max(2, min(6, hud_residual.count("\n") + 1))
            hud_txt = _clip_to_lines(hud_residual, hud_lines, mode="tail",
                                     max_line_width=line_width_cap)
            hud_panel = Panel(
                Text(hud_txt, style="cyan", overflow="fold"),
                title="HUD", title_align="left",
                border_style="cyan", padding=(0, 1),
            )
            hud_rows = hud_lines + 2
        legend_panel: Panel | None = None
        legend_rows = 0
        if legend:
            legend_cap = max(4, min(memory_cap, console_h // 5))
            legend_txt = _clip_to_lines(
                legend.strip(), legend_cap, mode="head",
                max_line_width=line_width_cap,
            )
            legend_panel = Panel(
                Text(legend_txt, style="bright_cyan", overflow="fold"),
                title="legend", title_align="left",
                border_style="bright_cyan", padding=(0, 1),
            )
            legend_rows = legend_cap + 2
        if hud_panel is not None and legend_panel is not None:
            row_size = max(hud_rows, legend_rows)
            row_layout = Layout(name="hud_legend")
            row_layout.split_row(
                Layout(hud_panel, name="hud", ratio=1),
                Layout(legend_panel, name="legend", ratio=2),
            )
            right_entries.append((row_layout, row_size))
        elif hud_panel is not None:
            right_entries.append((hud_panel, hud_rows))
        elif legend_panel is not None:
            right_entries.append((legend_panel, legend_rows))
        # Reasoning panel is rendered separately on the LEFT column,
        # below the grid box, so that the unused vertical space the
        # grid would otherwise leave empty gets reclaimed for chain
        # of thought. See body layout below.
        reasoning_panel: Panel | None = None
        if think:
            reasoning_border = "grey50"
            reasoning_title = (
                "reasoning (visible)"
                if reasoning_kind == "visible"
                else "reasoning"
            )
            think_txt = _clip_to_lines(
                think.strip(), reasoning_cap, mode="tail",
                max_line_width=line_width_cap,
            )
            reasoning_panel = Panel(
                Text(think_txt, style="italic grey78", overflow="fold"),
                title=reasoning_title, title_align="left",
                border_style=reasoning_border, padding=(0, 1),
            )
        action_color = "green" if (action and strict_action) else ("yellow" if action else "red")
        action_title = "action" + (
            " (PARSE FAIL)" if action_failed
            else " (fallback)" if action_fallback else ""
        )
        raw_action_txt = _clip_to_lines(
            raw_action, 4, mode="tail", max_line_width=line_width_cap,
        )
        action_body = [
            Text.assemble(
                ("parsed: ", "dim"),
                (action or "(no action parsed)", f"bold {action_color}"),
            ),
            Text.assemble(
                ("raw: ", "dim"),
                (raw_action_txt or "(empty)",
                 "bright_white" if raw_action_txt else "red"),
            ),
        ]
        action_rows = (
            2
            + 1
            + max(1, (raw_action_txt or "(empty)").count("\n") + 1)
        )
        right_entries.append((
            Panel(
                Group(*action_body),
                title=action_title, title_align="left",
                border_style=action_color, padding=(0, 1),
            ),
            action_rows,
        ))
        if turn_reward is not None or raw_score is not None or reward_block or status or message:
            footer_lines: list = []
            if turn_reward is not None:
                try:
                    reward_text = f"{float(turn_reward):+.3f}"
                except (TypeError, ValueError):
                    reward_text = str(turn_reward)
                footer_lines.append(Text.assemble(("turn reward: ", "bold"),
                                                  (reward_text, "yellow")))
            if raw_score is not None:
                footer_lines.append(Text.assemble(("NetHack score: ", "bold"),
                                                  (raw_score, "yellow")))
            if reward_block:
                footer_lines.append(Text.assemble(("reward: ", "bold"),
                                                  (reward_block.strip(), "yellow")))
            if status:
                footer_lines.append(Text.assemble(("status: ", "bold"),
                                                  (status.strip(), "magenta")))
            if message:
                footer_lines.append(Text.assemble(("message: ", "bold"),
                                                  (message.strip(), "bright_white")))
            right_entries.append((
                Panel(Group(*footer_lines), title="env feedback",
                      title_align="left", border_style="yellow", padding=(0, 1)),
                len(footer_lines) + 2,
            ))
        if memory is not None:
            stored_memory = str(memory.get("stored_memory") or "").strip()
            if stored_memory:
                stored_txt = _clip_to_lines(
                    stored_memory, memory_cap, mode="tail",
                    max_line_width=line_width_cap,
                )
                right_entries.append((
                    Panel(Text(stored_txt, style="bright_cyan", overflow="fold"),
                          title="updated memory", title_align="left",
                          border_style="bright_cyan", padding=(0, 1)),
                    memory_cap + 2,
                ))

        # ---- compose layout ----
        # Left column: sized grid panel on top, reasoning fills the
        # rest of the column (so the empty space the grid would have
        # left at the bottom now hosts the chain-of-thought instead of
        # going to waste). Right column stacks the remaining panels at
        # explicit row sizes.
        grid_panel_height = grid.count("\n") + 1 + 2  # +2 for borders
        body = Layout(name="body")
        body.split_row(
            Layout(name="left", ratio=left_ratio),
            Layout(name="right", ratio=right_ratio),
        )
        if reasoning_panel is not None:
            body["left"].split_column(
                Layout(grid_panel, name="grid", size=grid_panel_height),
                Layout(reasoning_panel, name="reasoning"),
            )
        else:
            # No reasoning this turn — give the grid the whole left
            # column so it doesn't sit pinned to the top with empty
            # space below.
            body["left"].update(grid_panel)
        def _as_sized_layout(item: object, size: int) -> Layout:
            if isinstance(item, Layout):
                item.size = size
                return item
            return Layout(item, size=size)

        body["right"].split_column(
            *[_as_sized_layout(panel, size) for panel, size in right_entries]
        )

        root = Layout(name="root")
        sub_layouts = [Layout(Panel(header_block, border_style="cyan", padding=(0, 1)),
                              name="hdr", size=4 + (1 if rollout_state else 0))]
        if sys_text:
            clipped_sys = _clip_to_lines(
                sys_text, sys_cap, mode="head",
                max_line_width=line_width_cap,
            )
            sys_panel = Panel(
                Text(clipped_sys, style="dim", overflow="fold"),
                title="system prompt", title_align="left",
                border_style="grey39", padding=(0, 1),
            )
            sub_layouts.append(Layout(sys_panel, name="sys", size=sys_cap + 2))
        sub_layouts.append(body)
        root.split_column(*sub_layouts)
        return root

    # Live: clean in-place screen (alternate buffer if TTY); each frame
    # fully replaces the previous one. auto_refresh=False — we drive
    # refreshes ourselves via live.update(..., refresh=True), so the
    # background ticker can't race with the raw-mode stdin read in
    # pause-mode (which would otherwise cause flickering / partial
    # frames after each keypress).
    is_tty = console.is_terminal
    with Live(render_frame(1), console=console, auto_refresh=False,
              screen=is_tty, transient=False, redirect_stdout=False,
              redirect_stderr=False) as live:
        if pause and is_tty:
            t_idx = 1
            n_turns = len(turns)
            while 1 <= t_idx <= n_turns:
                live.update(render_frame(t_idx), refresh=True)
                ch = _read_one_key()
                if ch == "q":
                    break
                # Pager hotkeys: stop Live, dump full content through
                # $PAGER, restart Live, redraw current frame.
                pager_text: str | None = None
                if ch == "s":
                    pager_text = sys_text or "(no system prompt)"
                elif ch == "r":
                    full_think, _ = _split_assistant(turns[t_idx - 1]["assistant"])
                    pager_text = full_think or "(no reasoning)"
                elif ch == "a":
                    action_extras = (
                        _rich_per_turn_extras[t_idx - 1][0]
                        if t_idx - 1 < len(_rich_per_turn_extras)
                        else {}
                    )
                    pager_text = _action_pager_text(
                        turns[t_idx - 1]["assistant"],
                        info.get("env_id"),
                        action_extras=action_extras,
                    )
                elif ch == "l":
                    full_legend = (_extract_legend(turns[t_idx - 1]["user"]) or "").strip()
                    pager_text = full_legend or "(no legend in this turn)"
                elif ch == "m":
                    mem = turns[t_idx - 1].get("memory") or {}
                    prev_m = (mem.get("previous_memory") or "").strip()
                    cur_m = (mem.get("stored_memory") or "").strip()
                    pager_text = (
                        f"PREVIOUS MEMORY (carried into turn {t_idx}):\n\n"
                        + (prev_m or "(empty)")
                        + "\n\n"
                        + "=" * 60
                        + f"\n\nUPDATED MEMORY (written at end of turn {t_idx}):\n\n"
                        + (cur_m or "(empty)")
                    )
                if pager_text is not None:
                    live.stop()
                    _show_in_pager(pager_text)
                    live.start()
                    live.update(render_frame(t_idx), refresh=True)
                    continue
                # Navigation: left arrow → previous, right arrow / any
                # other key → next.
                if ch == "\x1b[D":
                    t_idx = max(1, t_idx - 1)
                elif ch == "\x1b[C":
                    t_idx = min(n_turns, t_idx + 1)
                else:
                    t_idx += 1
        else:
            for t_idx in range(1, len(turns) + 1):
                live.update(render_frame(t_idx), refresh=True)
                if delay > 0:
                    time.sleep(delay)
            if is_tty:
                # Hold the final frame so the viewer sees the outcome.
                time.sleep(min(2.0, max(0.5, delay * 8)))


def _restore_terminal_sane() -> None:
    """Reset the controlling terminal to a sane cooked mode.

    Called both around each pager subprocess and on every exit path of
    the rich replay command. The rich Live + raw-mode keypress reads +
    a pager subprocess can leave the terminal with echo and line
    buffering disabled if anything goes sideways; this is the
    nuclear option that always recovers shell usability.
    """
    if not sys.stdout.isatty():
        return
    try:
        subprocess.run(["stty", "sane"], check=False)
    except (FileNotFoundError, OSError):
        # No stty available (Windows, stripped image): try a termios
        # restore from the saved state we cached at module import.
        if _SAVED_TTY_STATE is not None:
            try:
                import termios
                fd = sys.stdin.fileno()
                if sys.stdin.isatty():
                    termios.tcsetattr(fd, termios.TCSADRAIN, _SAVED_TTY_STATE)
            except Exception:
                pass


# Capture the inbound terminal state at import time. Used by
# _restore_terminal_sane as a fallback when ``stty`` isn't available.
_SAVED_TTY_STATE: object | None = None
try:
    if sys.stdin.isatty():
        import termios as _termios
        _SAVED_TTY_STATE = _termios.tcgetattr(sys.stdin.fileno())
except Exception:
    _SAVED_TTY_STATE = None


def _show_in_pager(text: str) -> None:
    """Open ``text`` in the user's $PAGER (less -R by default).

    Used by replay's pause mode so the viewer can scroll over content
    that's been clipped in the live frame (system prompt, full
    reasoning chain, full memory state). Falls back to a plain stdout
    dump if no pager is available.

    The text is written to a temp file and the pager is launched with
    that file as a positional argument — NOT via subprocess input
    piping — so the pager keeps stdin connected to the TTY and
    keyboard commands (``q`` to quit, arrow keys, ``/`` to search)
    actually reach it. Terminal state is restored before AND after the
    subprocess so a misbehaving pager can't leave the user's shell in
    a non-cooked mode.
    """
    import os
    import tempfile

    pager_env = os.environ.get("PAGER", "less -R")
    cmd = shlex.split(pager_env) if pager_env else []
    if not cmd:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        return
    fd, path = tempfile.mkstemp(prefix="gb-replay-", suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        _restore_terminal_sane()
        try:
            subprocess.run(cmd + [path], check=False)
        except (FileNotFoundError, OSError):
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
        finally:
            _restore_terminal_sane()
    finally:
        with suppress(OSError):
            os.unlink(path)


def _read_one_key() -> str:
    """Read a single keypress (or short escape sequence) from stdin
    without echo. Returns ``''`` on EOF.

    Recognised escape sequences are returned verbatim with the leading
    ``\\x1b``: arrow keys come back as ``\\x1b[A`` (up) / ``\\x1b[B``
    (down) / ``\\x1b[C`` (right) / ``\\x1b[D`` (left). Plain ASCII
    keys are returned lowercased.
    """
    import sys
    try:
        import select
        import termios
        import tty
    except ImportError:
        # Non-POSIX (Windows): fall back to line input.
        try:
            return (sys.stdin.readline() or "").strip().lower()[:1]
        except Exception:
            return ""
    fd = sys.stdin.fileno()
    if not sys.stdin.isatty():
        return (sys.stdin.readline() or "").strip().lower()[:1]
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Possible CSI / arrow sequence — drain up to 2 more bytes,
            # non-blocking, so a lone ESC doesn't hang and a 3-byte arrow
            # sequence ("\x1b[X") comes back whole.
            for _ in range(2):
                ready, _, _ = select.select([fd], [], [], 0.02)
                if not ready:
                    break
                ch += sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if not ch:
        return ""
    if ch.startswith("\x1b"):
        return ch  # raw escape sequence (preserve case in CSI byte)
    return ch.lower()


def _cmd_replay(args: argparse.Namespace) -> int:
    import time

    files = _discover_results_files(args.runs_dir)
    pro_paths = _discover_pro_transcripts(args.runs_dir)
    if not files and not pro_paths:
        print(
            f"no results.jsonl or transcript_*.jsonl found under {args.runs_dir}",
            file=sys.stderr,
        )
        return 2

    rollouts = _iter_rollouts(files) + _pro_rollouts(pro_paths)
    if not rollouts:
        print(
            f"no rollouts in {len(files)} results.jsonl + "
            f"{len(pro_paths)} transcript file(s)",
            file=sys.stderr,
        )
        return 2

    filtered = _filter_rollouts(
        rollouts,
        envs=args.env,
        suites=args.suite,
        models=args.model,
        seeds=args.seed,
    )

    if args.list:
        rows: dict[tuple[str, str, object], int] = {}
        for f, _, info in filtered:
            key = (_model_from_jsonl_path(f) or "?", info.get("env_id", "?"), info.get("seed"))
            rows[key] = rows.get(key, 0) + 1
        if not rows:
            print("(no rollouts match the filter)", file=sys.stderr)
            return 1
        width_m = max(len(k[0]) for k in rows) + 2
        width_e = max(len(k[1]) for k in rows) + 2
        print(f"{'model':<{width_m}}{'env_id':<{width_e}}{'seed':>6}  rollouts")
        print("-" * (width_m + width_e + 16))
        for (m, e, s), n in sorted(rows.items()):
            print(f"{m:<{width_m}}{e:<{width_e}}{str(s):>6}  {n}")
        source_count = len(files) + len(pro_paths)
        print(f"\n{len(filtered)} rollouts matched across {source_count} source file(s).")
        return 0

    if not filtered:
        print("(no rollouts match the filter; try `glyphbench replay <runs-dir> --list`)",
              file=sys.stderr)
        return 1

    if args.episode is not None:
        if args.episode < 0 or args.episode >= len(filtered):
            print(f"--episode {args.episode} out of range (0..{len(filtered) - 1})",
                  file=sys.stderr)
            return 2
        filtered = [filtered[args.episode]]

    # Wrap the actual playback in try/finally so that no exit path —
    # normal end, KeyboardInterrupt, an exception inside the rich Live
    # context, or a misbehaving pager subprocess — leaves the user's
    # shell in a non-cooked terminal mode (which would manifest as
    # missing keystroke echo and a dead readline history).
    use_rich = sys.stdout.isatty()
    try:
        if not use_rich:
            clear = "\x1b[2J\x1b[H"
            for f, rollout, info in filtered:
                model = _model_from_jsonl_path(f) or "?"
                header = (f"=== model={model}  env={info.get('env_id', '?')}  "
                          f"seed={info.get('seed', '?')}  reward={rollout.get('reward')} ===")
                sys.stdout.write(clear + header + "\n\n")
                sys.stdout.flush()
                time.sleep(min(args.delay * 4, 1.0))
                # Pre-index trajectory steps so we can look up per-step extras.
                # Trajectory is a sequence of steps: in memory mode each env-turn
                # produces an action step immediately followed by a memory step; in
                # non-memory mode every step is an action step.  We group them into
                # (action_extras, memory_extras) pairs — one pair per env-turn.
                # When an action step has no memory step (non-memory mode or the
                # last step of a memory-mode rollout) the memory_extras dict is {}.
                traj_steps = rollout.get("trajectory") or []
                per_turn_extras: list[tuple[dict, dict]] = []
                _a: dict = {}
                for _step in traj_steps:
                    _step_extras = (_step.get("extras") or {})
                    _role = _step_extras.get("glyphbench_step_role", "action")
                    if _role == "action":
                        # Flush any pending action step that had no memory partner.
                        if _a:
                            per_turn_extras.append((_a, {}))
                        _a = {
                            "action_chosen": _step_extras.get("action_chosen", ""),
                            "is_truncated": bool(_step.get("is_truncated")),
                            "reward": _step.get("reward"),
                            "env_info": _step_extras.get("env_info") or {},
                        }
                    elif _role == "memory":
                        per_turn_extras.append((
                            _a,
                            {
                                "memory_parse_failed": bool(
                                    _step_extras.get("memory_parse_failed")
                                ),
                                "is_truncated": bool(_step.get("is_truncated")),
                            },
                        ))
                        _a = {}
                # Flush the final action step if it has no memory partner.
                if _a:
                    per_turn_extras.append((_a, {}))
                for t_idx, turn in enumerate(_build_turns(rollout)):
                    grid = _extract_grid(turn["user"])
                    if grid is None:
                        continue
                    hud = _extract_hud(turn["user"])
                    message = _extract_message(turn["user"])
                    # Resolve per-step chip inputs from trajectory extras when
                    # available; fall back to safe defaults for legacy/partial files.
                    action_extras, mem_extras = (
                        per_turn_extras[t_idx] if t_idx < len(per_turn_extras)
                        else ({}, {})
                    )
                    # action_chosen: prefer trajectory extras (authoritative); fall
                    # back to parsing the assistant text (replay only needs the name).
                    action_chosen = action_extras.get("action_chosen") or ""
                    if not action_chosen:
                        action_chosen, _ = _resolve_action(
                            turn.get("assistant") or "", info.get("env_id")
                        )
                    reward_value = action_extras.get("reward")
                    if reward_value is None:
                        reward_value = rollout.get("reward")
                    try:
                        reward = float(reward_value or 0.0)
                    except (TypeError, ValueError):
                        reward = 0.0
                    raw_score = _nethack_raw_score(
                        info.get("env_id"),
                        action_extras,
                        hud=hud,
                        message=message,
                    )
                    line = _render_turn_line(
                        turn=t_idx + 1,
                        grid=grid,
                        hud=hud,
                        message=message,
                        action_chosen=action_chosen,
                        raw_action_output=_clip_to_lines(
                            _raw_action_output(turn.get("assistant") or ""),
                            4, mode="tail", max_line_width=200,
                        ),
                        raw_score=raw_score,
                        reward=reward,
                        is_truncated=bool(action_extras.get("is_truncated")),
                        memory_parse_failed=bool(mem_extras.get("memory_parse_failed")),
                        step_role="action",
                    )
                    memory = turn.get("memory")
                    suffix = ""
                    if memory and memory.get("stored_memory"):
                        suffix = "\n\n[Memory]\n" + _clip_to_lines(
                            str(memory["stored_memory"]).strip(),
                            8, mode="tail", max_line_width=200,
                        )
                    # If memory step itself was truncated, surface a chip line.
                    if mem_extras.get("is_truncated"):
                        mem_trunc_line = _render_turn_line(
                            turn=t_idx + 1,
                            grid="",
                            action_chosen=action_chosen,
                            raw_score=raw_score,
                            reward=reward,
                            is_truncated=True,
                            memory_parse_failed=bool(
                                mem_extras.get("memory_parse_failed")
                            ),
                            step_role="memory",
                        )
                        suffix += "\n" + mem_trunc_line
                    sys.stdout.write(clear + line + suffix + "\n")
                    sys.stdout.flush()
                    time.sleep(args.delay)
            return 0

        for f, rollout, info in filtered:
            model = _model_from_jsonl_path(f) or "?"
            _render_rollout_rich(
                rollout, info, model_id=model,
                delay=args.delay,
                reasoning_lines=args.reasoning_lines,
                pause=args.pause,
            )
        return 0
    finally:
        _restore_terminal_sane()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _cmd_list_suites(args: argparse.Namespace) -> int:
    for s in ALL_SUITES:
        print(s)
    return 0


def _cmd_list_envs(args: argparse.Namespace) -> int:
    import glyphbench  # noqa: F401
    from glyphbench.core import all_glyphbench_env_ids

    envs = [e for e in all_glyphbench_env_ids() if "dummy" not in e]
    if args.suite:
        envs = [e for e in envs if f"/{args.suite}-" in e]
    for e in envs:
        print(e)
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("glyphbench", description="GlyphBench CLI")
    subs = p.add_subparsers(dest="cmd", required=True)

    pr = subs.add_parser(
        "replay",
        help="Animate saved rollouts (rich panels in a TTY, plain text "
             "otherwise). Memory mode is rendered automatically when the "
             "rollout was produced with use_memory=True.",
    )
    pr.add_argument("runs_dir", type=Path,
                    help="Directory tree containing results.jsonl or "
                         "transcript_*.jsonl files (or one such file).")
    pr.add_argument("--env", action="append",
                    help="Filter by env_id. Repeatable. AND-combined with "
                         "--suite/--model/--seed.")
    pr.add_argument("--suite", action="append",
                    help="Filter by suite name. Repeatable.")
    pr.add_argument("--model", action="append",
                    help="Filter by HF model id. Repeatable.")
    pr.add_argument("--seed", action="append", type=int,
                    help="Filter by seed integer. Repeatable.")
    pr.add_argument("--episode", type=int, default=None,
                    help="0-indexed pick from the filtered set; plays only that one.")
    pr.add_argument("--list", action="store_true",
                    help="Don't play anything; print the (model, env_id, seed) "
                         "index of what matches the filters and exit.")
    pr.add_argument("--pause", action="store_true",
                    help="Step turn-by-turn. Keys: → / any → next frame; "
                         "← → previous frame; q → next rollout; "
                         "s → open full system prompt in $PAGER; "
                         "r → open full reasoning chain in $PAGER; "
                         "a → open full action parse/raw output in $PAGER; "
                         "l → open full legend (glyph table) in $PAGER; "
                         "m → open full previous + updated memory in "
                         "$PAGER. Overrides --delay.")
    pr.add_argument("--delay", type=float, default=0.15,
                    help="Seconds between turns. Default: 0.15")
    pr.add_argument("--reasoning-lines", type=int, default=None,
                    help="Max logical reasoning lines shown in the replay "
                         "panel. Default: about half the terminal height; "
                         "0 disables clipping. In --pause, press r to page "
                         "the full reasoning for the current turn.")
    pr.set_defaults(func=_cmd_replay)

    pls = subs.add_parser("list-suites", help="Print all suite names")
    pls.set_defaults(func=_cmd_list_suites)

    ple = subs.add_parser("list-envs", help="Print all env IDs")
    ple.add_argument("--suite", choices=ALL_SUITES,
                     help="Filter by a single suite")
    ple.set_defaults(func=_cmd_list_envs)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

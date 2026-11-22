"""Prompt construction for the Pro harness (two-call action/memory loop).

The system prompt = the env's own (mechanically-accurate, glyphbench-native)
tutorial + a PRO HARNESS PROTOCOL block describing the two turn types, the
persistent scratchpad, and the spatial-memory commands. Crucially it does
**not** threaten the model with forfeiting for "using too many tokens" — the
harness never truncates reasoning, so that self-defeating pressure is gone.

Per turn there are two user messages:

  * action turn — recent actions, the current-area focus, any pending
    landmark lookups, the persistent memory, and the current observation;
    the model reasons and commits exactly one ``<action>NAME</action>``.
  * memory turn — what the action did (re-injected reasoning + outcome), the
    env response, the next observation, and the current memory; the model
    replies with one ``<memory>...</memory>`` block, doing all of its
    scratchpad / plan / landmark bookkeeping there.
"""

from __future__ import annotations

from collections.abc import Iterable

from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.pro_harness.adapters import EnvAdapter

_PRO_PROTOCOL = """\
PRO HARNESS PROTOCOL

This run uses the GlyphBench Pro harness for long-horizon play. You have a
persistent external memory that carries between turns, and each environment
step has an action turn, with grounded memory turns inserted when useful:

1. ACTION TURN. The user message contains [Recent Actions], the current-area
   [Focus], your [Memory], and the [Observation]. Think as much as you need —
   there is no penalty for long reasoning and your reasoning is never cut off —
   then end your reply with exactly one action tag on its own final line:
       <action>ACTION_NAME</action>
   ACTION_NAME must be one of the names in the [Actions] list above. Discuss
   candidate actions as plain text while reasoning; only the final
   <action>...</action> tag is executed. Do NOT update memory on this turn.

2. MEMORY TURN (when requested). After an environment step, the user message contains
   [Last Action], [Env Response], [Next Observation], and your [Current
   Memory]. This turn is NOT an action turn — do not emit an <action> tag.
   Reply with exactly one block:
       <memory>
       ...updates...
       </memory>
   Do all of your strategic bookkeeping here: update the plan, record lessons,
   note what you explored, and maintain the landmark map.

PERSISTENT SCRATCHPAD (inside <memory>, merge-on-update — omit a section to
keep its previous value; include a header to overwrite it):
  STRATEGY        — high-level goal. Update rarely.
  PLAN            — numbered checklist; tick items and add next steps.
  LESSONS         — durable mechanics you have confirmed by observation.
  TACTICAL        — your immediate situation and next intent. Update often.
  FLOOR_NOTES     — exploration log for the current area: cleared vs.
                    unexplored regions, dead ends, hazards, resource/feature
                    locations, routes. Prevents backtracking and loops.
  FLOOR_CHECKLIST — what must be done before leaving this area.

LANDMARK MAP (inside <memory>) — a persistent, area-keyed location database:
  MAP_ADD: area=<floor/level>, type=<thing>, row=<r>, col=<c>, note=<short>
  MAP_REMOVE: area=<floor/level>, type=<thing>, row=<r>, col=<c>
  MAP_GET: area=<floor/level>            (recall this area's landmarks)
  MAP_GET: area=<floor/level>, type=<thing>
  MAP_GET_ALL                            (recall everything)
MAP_GET results are delivered at the top of your NEXT action turn under
[Recalled Landmarks]. {landmark_vocab}

GROUNDING: the [Observation] is always ground truth. Read it before trusting
memory; memory may be stale. Use [Recent Actions] to detect loops — if an
action keeps failing or you are oscillating, change approach.

Example memory turn:
<memory>
TACTICAL: Standing by a lake to the east; refill drink, then head to the
crafting table I placed before exploring north.
PLAN:
1. [x] Place crafting table
2. [ ] Mine 3 stone for a furnace
MAP_ADD: area=0, type=water, row=20, col=30, note=large lake
MAP_ADD: area=0, type=crafting_table, row=24, col=24, note=mine
</memory>"""


def build_system_prompt(
    game: BaseGlyphEnv,
    adapter: EnvAdapter,
    *,
    observation_mode: str = "text",
) -> str:
    """Compose the static system prompt: env tutorial + pro protocol.

    The adapter may override the env's own tutorial (Craftax substitutes the
    verbatim hand-tuned fork guide); otherwise the env's system_prompt() is
    used."""
    base = (adapter.base_system_prompt(game) or game.system_prompt()).rstrip()
    if observation_mode == "pixels":
        base += (
            "\n\n## Pixel observation modality\n"
            "Each action and memory turn includes the native Craftax pixel-rendered "
            "screen as an attached image. Treat that image as ground truth. The "
            "strategic guide may use textual glyph hints to name equivalent objects, "
            "but no ASCII/Unicode map or textual HUD is supplied as the primary "
            "observation in this run. Read visible map, status, inventory, equipment, "
            "lighting, and facing information from the image itself."
        )
    elif observation_mode == "native_text":
        base += (
            "\n\n## Native text observation modality\n"
            "Each turn supplies Craftax's native coordinate-by-coordinate text "
            "rendering. Coordinates are relative to the player: (0, 0) is your "
            "current cell. Treat this native text and its inventory/status fields "
            "as ground truth; no ASCII/Unicode glyph grid is supplied."
        )
    protocol = _PRO_PROTOCOL.format(landmark_vocab=adapter.landmark_vocabulary())
    return base + "\n\n---\n" + protocol


def render_recent_actions(recent: Iterable[tuple[int, str, float]]) -> str:
    items = list(recent)
    if not items:
        return "[Recent Actions]\n(none yet — this is the first turn)"
    lines = ["[Recent Actions] (most recent last)"]
    for turn, name, reward in items:
        if name == "FORFEIT":
            lines.append(f"  turn {turn}: FORFEIT (unparsed) -> reward {reward:+.3f}")
        else:
            lines.append(f"  turn {turn}: {name} -> reward {reward:+.3f}")
    return "\n".join(lines)


def render_memory_block(scratchpad_text: str, landmarks_text: str) -> str:
    return (
        "[Memory] (carried from previous turns; observation wins on conflict)\n"
        "Scratchpad:\n"
        f"{scratchpad_text}\n"
        "Landmarks:\n"
        f"{landmarks_text}"
    )


def render_action_user(
    *,
    adapter: EnvAdapter,
    game: BaseGlyphEnv,
    obs_text: str,
    turn: int,
    recent_actions: Iterable[tuple[int, str, float]],
    scratchpad_text: str,
    landmarks_text: str,
    pending_lookups: str = "",
    feedback: str = "",
    exploration_text: str = "",
    show_focus: bool = True,
    include_text_observation: bool = True,
) -> str:
    parts: list[str] = []
    if feedback:
        parts.append(f"[System Feedback]\n{feedback}")
    parts.append(f"[Turn {turn}] Choose the next action.")
    if pending_lookups:
        parts.append(f"[Recalled Landmarks]\n{pending_lookups}")
    parts.append(render_recent_actions(recent_actions))
    if show_focus:
        focus = adapter.focus(game, obs_text).strip()
        if focus:
            parts.append(f"[Focus]\n{focus}")
    if exploration_text:
        parts.append(f"[Exploration Aid]\n{exploration_text}")
    parts.append(render_memory_block(scratchpad_text, landmarks_text))
    if include_text_observation:
        parts.append(f"[Observation]\n{obs_text.rstrip()}")
    else:
        parts.append(
            "[Pixel Observation]\n"
            "The attached image is the current native Craftax screen."
        )
    parts.append(
        "Reason about the observation and your plan, then end with exactly one "
        "<action>NAME</action> tag on its own final line. NAME must be one of "
        "the action names listed in the system prompt."
    )
    return "\n\n".join(parts)


def render_memory_user(
    *,
    action_reasoning: str,
    action_name: str,
    parse_failed: bool,
    parse_reason: str | None,
    forfeit_mode: str,
    action_truncated: bool,
    reward: float,
    terminated: bool,
    truncated: bool,
    new_achievements: list[str],
    next_obs: str,
    scratchpad_text: str,
    landmarks_text: str,
    exploration_text: str = "",
    include_text_observation: bool = True,
) -> str:
    if parse_failed:
        if forfeit_mode == "noop":
            status = (
                f"FAILED ({parse_reason or 'no action tag'}) — turn forfeited; "
                f"the env applied its NOOP fallback, so dynamics advanced."
            )
        else:
            status = (
                f"FAILED ({parse_reason or 'no action tag'}) — turn forfeited; "
                "env state did not advance."
            )
    else:
        status = "ok"

    reasoning = (action_reasoning or "").strip()
    if reasoning:
        reasoning_block = (
            "Reasoning that led to the action (re-injected; the chat template "
            "drops prior-turn reasoning):\n---\n" + reasoning + "\n---"
        )
    else:
        reasoning_block = "Reasoning: (none emitted)"

    last_action = (
        "[Last Action]\n"
        + reasoning_block
        + "\nOutcome:\n"
        + f"  Action applied: {action_name}\n"
        + f"  Parse status: {status}\n"
        + f"  Output truncated: {str(bool(action_truncated)).lower()}"
    )
    ach = (
        ("  New achievements: " + ", ".join(new_achievements))
        if new_achievements
        else "  New achievements: none"
    )
    env_response = (
        "[Env Response]\n"
        f"  Reward: {reward:+.3f}\n"
        f"  Terminated: {str(bool(terminated)).lower()}\n"
        f"  Truncated: {str(bool(truncated)).lower()}\n"
        + ach
    )
    if include_text_observation:
        next_obs_block = "[Next Observation]\n" + (next_obs.rstrip() or "(none)")
    else:
        next_obs_block = (
            "[Next Pixel Observation]\n"
            "The attached image is the native Craftax screen after the action."
        )
    current_memory = (
        "[Current Memory]\nScratchpad:\n"
        f"{scratchpad_text}\nLandmarks:\n{landmarks_text}"
    )
    exploration = (
        "[Exploration Aid]\n" + exploration_text
        if exploration_text
        else ""
    )
    instruction = (
        "[Memory Update]\n"
        "This is a MEMORY TURN — do NOT emit an <action> tag and do not choose "
        "a game action. Reply with exactly one <memory>...</memory> block. "
        "Inside it, update only the scratchpad sections that changed (omitted "
        "sections are preserved) and issue any MAP_ADD/MAP_REMOVE/MAP_GET "
        "commands. Update TACTICAL for the new situation, tick/extend PLAN, "
        "record durable LESSONS and FLOOR_NOTES, and maintain the landmark map. "
        "Do not re-describe the grid; it is already shown above."
    )
    return "\n\n".join(
        part for part in
        [last_action, env_response, next_obs_block, exploration, current_memory, instruction]
        if part
    )

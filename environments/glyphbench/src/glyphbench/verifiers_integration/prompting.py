"""Prompt construction for the glyphbench verifiers integration.

Two entry points:
    build_system_prompt(game, max_output_tokens) → str
        Composed from game.system_prompt() + a standardised response-format
        block. Called ONCE per rollout (static across turns).

    render_user_turn(game, frames, current_obs, turn) → str
        The per-turn user message. Layout:

            [Legend]
              <union of glyphs across frames + current, rendered once>

            [History — last N turns]                 (omitted entirely if N=0)
              (turn T-K) <grid-only view>
                chose ACTION → reward R

              (turn T-K+1) ...

            [Current Observation — turn T]
              <full grid + HUD + optional message>

            Now reason briefly, then end with a final action tag.

Design goals:
    * One legend per user message, deduped across history + current.
    * Stable section order → KV-cache prefix overlap across turns.
    * History frames stripped to grid + action/reward only, saving tokens.
    * Current observation kept intact (full grid + HUD + message), minus legend.
    * Budget reminded every turn, deters runaway thinking.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable

from glyphbench.core.base_env import BaseGlyphEnv

AGENTICK_MARKOV_REASONER_HARNESS = "agentick_markov_reasoner"

_LEGEND_RE = re.compile(r"\[Legend\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)
_GRID_RE = re.compile(r"\[Grid\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)
_HUD_RE = re.compile(r"\[HUD\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)
_MESSAGE_RE = re.compile(r"\[Message\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)


REASONING_RESPONSE_FORMAT_BLOCK_TMPL = (
    "RESPONSE FORMAT\n"
    "Reason before acting. {reasoning_budget_guidance} Inspect the observation, "
    "track relevant state, compare plausible moves, and choose a plan before "
    "committing to an action. Always preserve room for the closing action tag. "
    "For models without a native thinking "
    "mode, write the reasoning in plain assistant text before the action. For models with "
        "a native thinking mode, use that mode normally. "
        "{reasoning_retention_guidance}\n"
    "\n"
    "After reasoning, end your response with exactly one final action tag on "
    "its own line. Put the exact chosen action name between the tags; for "
    "example, if MOVE_FORWARD is listed and you choose it, write:\n"
    "  <action>MOVE_FORWARD</action>\n"
    "\n"
    "The action name must be one of the names in the [Actions] list. Do not emit "
    "an <action> tag until the final line; if you discuss candidate actions "
    "during reasoning, write their names as plain text or in backticks instead "
    "of XML tags. Read the grid directly — when the player glyph is "
    "directional, it shows your orientation. The evaluator parses the final "
    "complete <action> tag as your committed move and stores the preceding "
    "reasoning in the conversation transcript.\n"
    "\n"
    "Failure modes: {truncation_guidance}If the final <action> tag is missing or "
    "contains an unknown name, the turn is also forfeited the same way."
)

ACTION_ONLY_RESPONSE_FORMAT_BLOCK_TMPL = (
    "RESPONSE FORMAT\n"
    "Valid action names for this task are: {valid_actions}.\n"
    "Reply with exactly one XML action tag and no other text:\n"
    "  <action>{example_action}</action>\n"
    "\n"
    "Replace {example_action} with the exact action name you choose. "
    "Do not include reasoning, explanations, markdown, code fences, extra XML, "
    "or text before or after the action tag. The evaluator parses the final "
    "complete <action> tag as your committed move.\n"
    "\n"
    "{action_budget_guidance}If the final <action> tag is missing "
    "or contains an unknown name, the turn is also forfeited the same way."
)

OBSERVATION_CONVENTIONS_BLOCK = (
    "OBSERVATION CONVENTIONS\n"
    "Each turn you receive a [Grid] (Unicode text) and a [Legend] mapping each "
    "glyph to its meaning for this turn. You may also receive a [HUD] "
    "block with complementary state that cannot be read from the grid "
    "(for example turn budget, HP, inventory, score, velocity, cooldowns, "
    "or facing only when the glyph itself is not directional), plus "
    "optionally a [Message] block with one-shot env feedback (last action "
    "result, etc.). Treat the "
    "[Grid] as authoritative for spatial state and the [Legend] as "
    "authoritative for glyph meaning; do not expect the HUD to repeat "
    "positions, visible enemy locations, or facing already encoded by a "
    "directional glyph. The full action list is in this system prompt "
    "and does not repeat per turn."
)

MEMORY_BLOCK_TMPL = (
    "MEMORY MODE (active for this run)\n"
    "Each environment turn is followed by a memory-update turn. After "
    "you emit the action above, you'll be shown a packet with four "
    "sections:\n"
    "  [Last Action]      your previous response (reasoning + action) "
    "plus parser/truncation flags so you know exactly what was applied;\n"
    "  [Env Response]     reward and the terminated/truncated flags;\n"
    "  [Next Observation] the grid + legend + HUD + message produced by "
    "your action — the same view your next action turn will receive;\n"
    "  [Memory Update]    write instructions.\n"
    "Reply ONLY with `<memory>...your concise updated memory...</memory>` "
    "— do not emit an <action> tag in the memory turn. Memory is for "
    "synthesis: causal facts you've established, plans, discoveries, "
    "retracted hypotheses. Do NOT re-describe the grid; the next "
    "observation is already in the prompt and will be shown again at the "
    "next action turn. The text inside <memory> is carried into the next "
    "turn's observation as a [Memory] block (authoritative for state you "
    "want to track across turns; the current observation still wins on "
    "conflicts). {memory_budget_guidance}"
)


def build_system_prompt(
    game: BaseGlyphEnv,
    max_output_tokens: int | None,
    *,
    use_memory: bool = False,
    memory_update_max_tokens: int | None = None,
    forfeit_mode: str = "freeze",
    harness: str = "default",
    enable_reasoning: bool | None = None,
) -> str:
    """Compose the system prompt: game rules + standard response-format
    block + observation conventions (+ memory block when memory mode is
    on).

    The output is stable across turns — verifiers reuses the cached
    tokenisation as long as this content doesn't change.
    """
    header = game.system_prompt().rstrip()
    if enable_reasoning is None:
        # Preserve the existing evaluation behavior unless a caller makes the
        # reasoning policy explicit. Native v1 RL does so independently of
        # memory: thinking remains enabled and trainable while memory is off.
        enable_reasoning = harness == AGENTICK_MARKOV_REASONER_HARNESS or use_memory
    if enable_reasoning:
        response_format = REASONING_RESPONSE_FORMAT_BLOCK_TMPL
    else:
        response_format = ACTION_ONLY_RESPONSE_FORMAT_BLOCK_TMPL
    if max_output_tokens is None:
        reasoning_budget_guidance = (
            "No explicit output-token cap is imposed: use the model's native "
            "reasoning budget and take as much reasoning as is useful."
        )
        action_budget_guidance = (
            "No explicit output-token cap is imposed, but a valid response is "
            "only the short action tag. "
        )
        truncation_guidance = ""
    else:
        reasoning_budget_guidance = (
            f"You may use up to {max_output_tokens} output tokens total for "
            "reasoning plus the final action."
        )
        action_budget_guidance = (
            f"The output budget is {max_output_tokens} tokens, but a valid "
            "response should be only the short action tag. If the full response "
            f"exceeds {max_output_tokens} tokens before the closing </action> "
            "tag, the action is discarded and the turn is forfeited "
            f"({_forfeit_consequence(forfeit_mode)}). "
        )
        truncation_guidance = (
            f"If your full response exceeds {max_output_tokens} tokens before "
            "the closing </action> tag, the action is discarded and the turn "
            f"is forfeited ({_forfeit_consequence(forfeit_mode)}). "
        )
    fmt = response_format.format(
        reasoning_budget_guidance=reasoning_budget_guidance,
        action_budget_guidance=action_budget_guidance,
        truncation_guidance=truncation_guidance,
        forfeit_consequence=_forfeit_consequence(forfeit_mode),
        valid_actions=", ".join(game.action_spec.names),
        example_action=game.action_spec.names[0],
        reasoning_retention_guidance=(
            "Any exposed reasoning will be stored, replayed, and passed to the "
            "memory update."
            if use_memory
            else "Your reasoning will be stored and replayed as part of the "
            "conversation on later turns."
        ),
    )
    blocks: list[str] = [header, OBSERVATION_CONVENTIONS_BLOCK]
    # Most env classes already append `action_spec.render_for_prompt()` in
    # their own `system_prompt()` (see e.g. minigrid/base.py, procgen/base.py,
    # most atari/*.py). Only inject an extra Actions block here when the
    # header doesn't already contain it.
    actions_block = game.action_spec.render_for_prompt().strip()
    if actions_block and actions_block not in header:
        blocks.append(actions_block)
    blocks.append(fmt)
    if use_memory:
        memory_budget_guidance = (
            "No explicit output-token cap is imposed on memory updates; keep "
            "them concise and useful."
            if memory_update_max_tokens is None
            else f"Cap the memory at {memory_update_max_tokens} tokens; output "
            "beyond that is discarded by the parser."
        )
        blocks.append(MEMORY_BLOCK_TMPL.format(
            memory_budget_guidance=memory_budget_guidance,
        ))
    return "\n\n---\n".join(blocks)


def _forfeit_consequence(forfeit_mode: str) -> str:
    if forfeit_mode == "freeze":
        return "env state unchanged, turn counter still advances, reward 0"
    if forfeit_mode == "noop":
        return (
            "the env applies its NOOP/fallback action, dynamics advance "
            "normally, and reward comes from that fallback step"
        )
    raise ValueError("forfeit_mode must be 'freeze' or 'noop'")


def render_user_turn(
    game: BaseGlyphEnv,
    frames: deque[tuple[str, str, float]] | Iterable[tuple[str, str, float]],
    current_obs: str,
    turn: int,
    max_output_tokens: int | None,
    *,
    memory: str | None = None,
    enable_reasoning: bool | None = None,
) -> str:
    """Render the user-turn message for the current timestep.

    Args:
        game: the live game instance (used for the action list).
        frames: iterable of (obs_text_before_action, action_name, reward) tuples
                in temporal order — oldest first, newest last.
        current_obs: the full ``GridObservation.render()`` text for this turn.
        turn: the absolute turn number (``game.turn`` after the last step).
        max_output_tokens: per-turn LLM budget, echoed in the footer reminder.
        memory: optional carried state to show before the current observation.
        enable_reasoning: match the system prompt's response format; when omitted,
            preserve the memory-based evaluation default.

    Returns:
        The user-turn string.
    """
    frames_list = list(frames)

    # 1. Build merged legend across history + current.
    legend_lines: dict[str, None] = {}  # ordered-set via dict
    for obs, _, _ in frames_list:
        for line in _extract_legend_lines(obs):
            legend_lines.setdefault(line, None)
    for line in _extract_legend_lines(current_obs):
        legend_lines.setdefault(line, None)

    parts: list[str] = []
    if memory is not None:
        memory_text = memory.strip() if memory.strip() else "(empty)"
        parts.append(
            "[Memory]\n"
            "Use this as carried state from previous turns. The current "
            "observation is authoritative if it conflicts.\n\n"
            f"{memory_text}"
        )
    if legend_lines:
        parts.append("[Legend]\n" + "\n".join(legend_lines))

    # 2. History block (only if non-empty).
    if frames_list:
        parts.append(_render_history(frames_list, turn))

    # 3. Current observation — strip legend, keep grid + HUD + message.
    parts.append(_render_current_block(current_obs, turn))

    # 4. Per-turn nudge — the full action list with descriptions is in
    # the cached system prompt, so we don't repeat it here. This line
    # only re-asserts the response format the agent must use right now.
    if enable_reasoning is None:
        enable_reasoning = memory is not None
    if not enable_reasoning:
        parts.append(
            "Reply now with exactly one <action>...</action> tag containing "
            "one action name from the system prompt's action list, and no "
            "other text."
        )
    else:
        parts.append(
            "Now reason as much as useful, then end with a final <action> tag containing "
            "exactly one action name from the system prompt's action list."
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extract_legend_lines(obs: str) -> list[str]:
    m = _LEGEND_RE.search(obs)
    if not m:
        return []
    return [ln for ln in m.group(1).splitlines() if ln.strip()]


def _extract_grid(obs: str) -> str:
    m = _GRID_RE.search(obs)
    return m.group(1).rstrip() if m else ""


def _extract_hud(obs: str) -> str:
    m = _HUD_RE.search(obs)
    return m.group(1).strip() if m else ""


def _extract_message(obs: str) -> str:
    m = _MESSAGE_RE.search(obs)
    return m.group(1).strip() if m else ""


def _render_history(frames_list: list[tuple[str, str, float]], current_turn: int) -> str:
    n = len(frames_list)
    lines = [f"[History — last {n} turn{'s' if n != 1 else ''}]"]
    # Number each historical turn from T-N .. T-1 (T is the current turn).
    # History stays grid-only to keep token use bounded; the current
    # observation carries the cleaned complementary HUD.
    for i, (obs, action, reward) in enumerate(frames_list):
        past_turn = current_turn - (n - i)
        grid = _extract_grid(obs)
        if action == "FORFEIT":
            reward_text = "0" if abs(float(reward)) < 1e-12 else f"{reward:+.3f}"
            action_line = f"chose FORFEIT (parse failed) → reward {reward_text}"
        else:
            action_line = f"chose {action} → reward {reward:+.3f}"
        lines.append(
            f"(turn {past_turn})\n"
            f"{grid}\n"
            f"{action_line}".replace("  \n", "").replace("\n\n", "\n")
        )
    return "\n".join(lines)


def _render_current_block(current_obs: str, turn: int) -> str:
    # Drop legend (rendered globally above), but preserve the cleaned HUD.
    grid = _extract_grid(current_obs)
    hud = _extract_hud(current_obs)
    msg = _extract_message(current_obs)
    parts = [f"[Current Observation — turn {turn}]"]
    if grid:
        parts.append(f"[Grid]\n{grid}")
    if hud:
        parts.append(f"[HUD]\n{hud}")
    if msg:
        parts.append(f"[Message]\n{msg}")
    return "\n".join(parts)

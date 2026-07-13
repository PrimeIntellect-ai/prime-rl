"""Group-level Q&A meta-extraction (TTT): one model call per finished GRPO group distills
contrastive lessons across attempts (rewards known), shipped as ce-routed samples exactly
like recycled pairs. Enrichment only: the train sink logs and skips failures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from verifiers.v1.ttt import QAConfig, dedup_items, parse_qa_items

from prime_rl.transport import TrainingSample
from prime_rl.utils.qa_render import qa_pairs_to_ce_samples

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from prime_rl.orchestrator.types import Rollout


async def extract_meta_lessons(group: list["Rollout"], qa: QAConfig, openai: "AsyncOpenAI", model: str) -> list[dict]:
    """One meta-extraction call over the group; return parsed, deduplicated items.

    Provider failures intentionally propagate to the train sink, which contains them
    and increments ``ttt/meta_groups_dropped``. Returning an empty list is reserved for
    valid no-op outcomes such as a group with fewer than two pair-bearing rollouts.
    """
    with_pairs = [r for r in group if any(u.get("qa_pairs") for u in r.info.get("ttt", {}).get("updates", []))]
    if len(with_pairs) < 2:
        return []  # nothing to contrast
    # The {attempts} block: each rollout's reward + its extracted Q&A items.
    blocks: list[str] = []
    for i, rollout in enumerate(with_pairs, 1):
        items = [
            pair for update in rollout.info.get("ttt", {}).get("updates", []) for pair in update.get("qa_pairs") or []
        ]
        lines = [f"Attempt {i} (reward: {rollout.reward:.3f}):"]
        for item in items:
            lines.append(f"- Q: {item['question']}")
            lines.append(f"  A: {item['answer']}")
        if not items:
            lines.append("- (no study-set items recorded)")
        blocks.append("\n".join(lines))
    prompt = qa.meta_prompt.format(num_attempts=len(with_pairs), attempts="\n\n".join(blocks))
    completion = await openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=qa.meta_max_tokens,
    )
    text = completion.choices[0].message.content or ""
    items = parse_qa_items(text)
    if qa.dedup_threshold is not None:
        items = dedup_items(items, qa.dedup_threshold)
    return items


def meta_lesson_samples(items: list[dict], group: list["Rollout"], tokenizer, env_name: str) -> list[TrainingSample]:
    """Render the meta lessons as ce-routed training samples, conditioned on the group's
    recorded system prompt + tools — the same frame `qa_recycle_samples` uses."""
    ttt_info = next(
        (info for r in group if (info := r.info.get("ttt", {})).get("system_prompt") or info.get("tools")),
        {},
    )
    return qa_pairs_to_ce_samples(items, ttt_info.get("system_prompt"), ttt_info.get("tools"), tokenizer, env_name)

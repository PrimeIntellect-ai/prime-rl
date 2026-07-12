"""Group-level Q&A meta-extraction (TTT): distill cross-rollout lessons from a finished
GRPO group into the policy's main weights.

Per-rollout Q&A (see `verifiers.v1.ttt`) is myopic by construction — each rollout only
sees its own experience. A finished group adds the signal no single rollout has: several
independent attempts at the *same* task, each with a reward. When `ttt.qa.meta_lessons`
is set, one model call per group sees every rollout's extracted Q&A items alongside its
reward and produces contrastive, general lessons ("the high-reward attempts did X; the
low-reward ones hit pitfall Y"), which ship as ce-routed training samples exactly like
recycled pairs (`qa_recycle_samples`) — riding the same training batch, no rl credit.

Runs in the train sink's group phase (the same scope as `Algorithm.score_group`: the
cohort is complete, rewards are known, filtering hasn't happened). The call goes to the
live policy through the env's sampler pool. The train sink logs and skips failures —
meta lessons are enrichment, not correctness; a failed extraction must not fail the
group.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from verifiers.v1.ttt import QAConfig, dedup_items, parse_qa_items

from prime_rl.transport import TrainingSample
from prime_rl.utils.qa_render import render_qa_pair

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from prime_rl.orchestrator.types import Rollout


def format_attempts(group: list["Rollout"]) -> str:
    """The `{attempts}` block: each rollout's reward + its extracted Q&A items."""
    blocks: list[str] = []
    for i, rollout in enumerate(group, 1):
        items = [
            pair for update in rollout.info.get("ttt", {}).get("updates", []) for pair in update.get("qa_pairs") or []
        ]
        lines = [f"Attempt {i} (reward: {rollout.reward:.3f}):"]
        if items:
            for item in items:
                lines.append(f"- Q: {item['question']}")
                lines.append(f"  A: {item['answer']}")
        else:
            lines.append("- (no study-set items recorded)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def extract_meta_lessons(group: list["Rollout"], qa: QAConfig, openai: "AsyncOpenAI", model: str) -> list[dict]:
    """One meta-extraction call over the group; return parsed, deduplicated items.

    Provider failures intentionally propagate to the train sink, which contains them
    and increments ``ttt/meta_groups_dropped``. Returning an empty list is reserved for
    valid no-op outcomes such as a group with fewer than two pair-bearing rollouts.
    """
    with_pairs = [r for r in group if any(u.get("qa_pairs") for u in r.info.get("ttt", {}).get("updates", []))]
    if len(with_pairs) < 2:
        return []  # nothing to contrast
    prompt = qa.meta_prompt.format(num_attempts=len(with_pairs), attempts=format_attempts(with_pairs))
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
    system prompt + tools (read from any rollout's recorded QA conditioning) — the same
    frame `qa_recycle_samples` uses."""
    ttt_info = next(
        (info for r in group if (info := r.info.get("ttt", {})).get("system_prompt") or info.get("tools")),
        {},
    )
    system_prompt = ttt_info.get("system_prompt")
    tools = ttt_info.get("tools")
    head = [{"role": "system", "content": system_prompt}] if system_prompt else []
    template_kwargs: dict = {"tools": tools} if tools else {}
    samples: list[TrainingSample] = []
    for item in items:
        answer = str(item.get("answer", ""))
        if not answer.strip():
            continue
        conversation = [
            *head,
            {"role": "user", "content": str(item.get("question", ""))},
            {"role": "assistant", "content": answer},
        ]
        rendered = render_qa_pair(tokenizer, conversation, template_kwargs)
        if rendered is None:
            continue  # non-prefix-stable render: skip rather than train on the full render
        full, prompt_len = rendered
        answer_len = len(full) - prompt_len
        if answer_len < 1:
            continue
        mask = [False] * prompt_len + [True] * answer_len
        samples.append(
            TrainingSample(
                token_ids=full,
                mask=mask,
                logprobs=[0.0] * len(full),
                temperatures=[1.0] * len(full),  # ce NLL is temperature-free MLE — never rescale logits
                env_name=env_name,
                rl_weights=[0.0] * len(full),
                ce_weights=[1.0 if m else 0.0 for m in mask],
            )
        )
    return samples


def build_meta_client(client_config) -> "AsyncOpenAI":
    """A plain chat client for the meta call, from a verifiers `ClientConfig` (the policy
    pool's eval client)."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=client_config.base_url,
        api_key=os.environ.get(client_config.api_key_var) or "EMPTY",
        default_headers=client_config.headers or None,
    )

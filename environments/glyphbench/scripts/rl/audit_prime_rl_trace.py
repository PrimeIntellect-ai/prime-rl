#!/usr/bin/env python3
"""Audit real PrimeRL/Verifiers traces against GlyphBench's RL contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from prime_rl.orchestrator.trajectories import trace_to_samples
from transformers import AutoTokenizer
from verifiers.v1.types import AssistantMessage, SystemMessage, UserMessage

TURN_RE = re.compile(r"\[Current Observation — turn (\d+)\]")
ACTION_RE = re.compile(
    r"<\s*action\s*>\s*([A-Za-z0-9_]+)\s*<\s*/\s*action\s*>",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate real saved traces and their upstream trainer samples."
    )
    parser.add_argument("trace_files", nargs="+", type=Path)
    parser.add_argument("--max-traces", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=32_768)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    return parser.parse_args()


def _sampled_text(tokenizer, node: vf.MessageNode) -> str:
    sampled_ids = [
        token_id for token_id, trainable in zip(node.token_ids, node.mask, strict=True) if trainable
    ]
    return tokenizer.decode(sampled_ids, skip_special_tokens=False)


def audit_trace(
    trace: vf.Trace,
    tokenizer,
    max_seq_len: int,
    *,
    env_name: str,
) -> dict[str, Any]:
    assert not trace.has_error, f"trace {trace.id} has errors: {trace.errors}"
    assert len(trace.branches) == 1, (
        f"trace {trace.id}: {trace.num_turns} turns became {len(trace.branches)} branches"
    )
    branch = trace.branches[0]
    nodes = branch.nodes
    messages = branch.messages

    assert isinstance(messages[0], SystemMessage), f"trace {trace.id}: system is not first"
    assert sum(isinstance(message, SystemMessage) for message in messages) == 1, (
        f"trace {trace.id}: system prompt was repeated"
    )
    assert len(nodes) == len(messages)

    expected_turn = 0
    assistant_count = 0
    reasoning_count = 0
    observation_turns: list[int] = []
    actions: list[str] = []
    assistant_trainable_tokens: list[int] = []
    assistant_reasoning_characters: list[int] = []
    for index, (node, message) in enumerate(zip(nodes, messages, strict=True)):
        assert len(node.token_ids) == len(node.mask), (
            f"trace {trace.id} node {index}: token/mask length mismatch"
        )
        if node.is_content:
            assert len(node.is_content) == len(node.token_ids), (
                f"trace {trace.id} node {index}: content-mask length mismatch"
            )

        if isinstance(message, (SystemMessage, UserMessage)):
            assert not any(node.mask), (
                f"trace {trace.id} node {index}: context token marked trainable"
            )
        if isinstance(message, UserMessage):
            match = TURN_RE.search(message.content or "")
            assert match, f"trace {trace.id} node {index}: missing absolute turn label"
            actual_turn = int(match.group(1))
            assert actual_turn == expected_turn, (
                f"trace {trace.id}: expected observation turn {expected_turn}, got {actual_turn}"
            )
            expected_turn += 1
            observation_turns.append(actual_turn)
        if isinstance(message, AssistantMessage):
            assistant_count += 1
            assert node.sampled, f"trace {trace.id} node {index}: assistant not sampled"
            assert any(node.mask), f"trace {trace.id} node {index}: assistant has no loss tokens"
            first_trainable = node.mask.index(True)
            assert all(node.mask[first_trainable:]), (
                f"trace {trace.id} node {index}: sampled completion contains masked holes"
            )
            sampled_text = _sampled_text(tokenizer, node)
            reasoning = (message.reasoning_content or "").strip()
            visible = (message.content or "").strip()
            assert reasoning or visible, f"trace {trace.id} node {index}: assistant reply is empty"
            action_tags = ACTION_RE.findall(sampled_text)
            reasoning_text = ACTION_RE.sub("", sampled_text)
            reasoning_text = reasoning_text.replace("<|im_end|>", "").strip()
            if reasoning_text:
                reasoning_count += 1
            actions.append(action_tags[-1] if action_tags else "FORFEIT")
            assistant_trainable_tokens.append(sum(node.mask))
            assistant_reasoning_characters.append(len(reasoning))

    assert assistant_count == trace.num_turns, (
        f"trace {trace.id}: assistant turns={assistant_count}, trace.num_turns={trace.num_turns}"
    )
    assert expected_turn == assistant_count, (
        f"trace {trace.id}: observations={expected_turn}, assistants={assistant_count}"
    )

    samples = trace_to_samples(trace, env_name=env_name)
    assert len(samples) == 1, f"trace {trace.id}: produced {len(samples)} trainer samples"
    sample = samples[0]
    assert len(sample.token_ids) <= max_seq_len, (
        f"trace {trace.id}: trainer sample has {len(sample.token_ids)} tokens, "
        f"limit is {max_seq_len}"
    )
    assert len(sample.token_ids) == len(sample.mask) == len(sample.logprobs), (
        f"trace {trace.id}: trainer token/mask/logprob arrays are misaligned"
    )
    assert sample.token_ids == branch.token_ids
    assert sample.mask == branch.sampled_mask
    assert sum(sample.mask) == trace.num_output_tokens

    return {
        "trace": trace.id,
        "stop_condition": trace.stop_condition,
        "is_truncated": trace.is_truncated,
        "turns": trace.num_turns,
        "branches": len(trace.branches),
        "samples": len(samples),
        "roles": [message.role for message in messages],
        "observation_turns": observation_turns,
        "actions": actions,
        "assistant_trainable_tokens": assistant_trainable_tokens,
        "assistant_reasoning_characters": assistant_reasoning_characters,
        "tokens": len(sample.token_ids),
        "trainable_tokens": sum(sample.mask),
        "reasoning_turns": reasoning_count,
    }


def main() -> None:
    args = parse_args()
    if args.max_traces < 1:
        raise ValueError("--max-traces must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    checked = 0
    for path in args.trace_files:
        with path.open() as handle:
            for line in handle:
                episode = vf.Episode.model_validate(json.loads(line))
                for trace in episode.traces:
                    print(
                        json.dumps(
                            audit_trace(
                                trace,
                                tokenizer,
                                args.max_seq_len,
                                env_name=episode.env.name,
                            )
                        )
                    )
                    checked += 1
                    if checked >= args.max_traces:
                        print(f"audited {checked} trace(s): PASS")
                        return
    if checked == 0:
        raise RuntimeError("no traces found")
    print(f"audited {checked} trace(s): PASS")


if __name__ == "__main__":
    main()

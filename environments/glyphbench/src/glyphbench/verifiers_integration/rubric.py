"""Rubric: sums per-step rewards across the rollout + tracks observability metrics.

Primary reward: ``episodic_return`` (weight 1.0). All other functions are
``weight=0`` metrics for observability — kept distinct so future debugging
never confuses different failure modes.

Metric glossary (full reference in docs/llm-agent-failure-modes.md):
  - ``episode_length`` — number of env-action turns this rollout took.
  - ``episodic_return`` — sum of per-turn rewards (the trained signal).
  - ``episode_terminated_rate`` — 1.0 if the env reached a terminal state.
  - ``episode_truncated_max_turns_rate`` — 1.0 if the env hit ``max_turns``.
  - ``context_limit_hit`` — 1.0 if the rollout framework refused the next
        generation because its context capacity was exhausted.
  - ``forfeit_rate`` — fraction of action turns that forfeited (parse failed).
  - ``action_completion_truncation_rate`` — fraction of action turns whose
        completion ran into ``max_tokens`` (4096 today).
  - ``memory_completion_truncation_rate`` — same for memory turns
        (memory mode only; 4096 today).
  - ``memory_parse_failure_rate`` — fraction of memory turns that emitted
        unparseable output (no ``<memory>...</memory>``); previous memory is
        retained on those turns.
  - ``xml_format_reward`` — verifiers' built-in XML-format compliance.
"""

from __future__ import annotations

from typing import Any

import verifiers as vf

from glyphbench.termination import is_context_limit_stop


class EpisodicReturnRubric(vf.Rubric):
    def __init__(self, parser: vf.Parser | None = None, **kwargs: Any) -> None:
        super().__init__(parser=parser, **kwargs)
        self.add_reward_func(self.episodic_return, weight=1.0)
        self.add_metric(self.episode_length)
        self.add_metric(self.episode_terminated_rate)
        self.add_metric(self.episode_truncated_max_turns_rate)
        self.add_metric(self.context_limit_hit)
        self.add_metric(self.forfeit_rate)
        self.add_metric(self.parse_recovery_rate)
        self.add_metric(self.action_completion_truncation_rate)
        self.add_metric(self.memory_completion_truncation_rate)
        self.add_metric(self.memory_parse_failure_rate)
        if parser is not None:
            try:
                fmt = parser.get_format_reward_func()
                fmt.__name__ = "xml_format_reward"
                self.add_metric(fmt)
            except AttributeError:
                pass

    async def episodic_return(self, state: dict[str, Any]) -> float:
        return float(state.get("episode_return", 0.0))

    async def episode_length(self, state: dict[str, Any]) -> float:
        n = state.get("num_turns")
        if n is None:
            n = len(state.get("trajectory", []))
        return float(n)

    async def episode_terminated_rate(self, state: dict[str, Any]) -> float:
        return 1.0 if state.get("terminated") else 0.0

    async def episode_truncated_max_turns_rate(self, state: dict[str, Any]) -> float:
        return 1.0 if state.get("truncated") else 0.0

    async def context_limit_hit(self, state: dict[str, Any]) -> float:
        return 1.0 if is_context_limit_stop(state.get("stop_condition")) else 0.0

    async def forfeit_rate(self, state: dict[str, Any]) -> float:
        n = state.get("num_action_turns", 0)
        if not n:
            return 0.0
        return float(state.get("forfeit_count", 0)) / float(n)

    async def parse_recovery_rate(self, state: dict[str, Any]) -> float:
        n = state.get("num_action_turns", 0)
        if not n:
            return 0.0
        return float(state.get("parse_recoveries", 0)) / float(n)

    async def action_completion_truncation_rate(self, state: dict[str, Any]) -> float:
        n = state.get("num_action_turns", 0)
        if not n:
            return 0.0
        return float(state.get("action_completion_truncations", 0)) / float(n)

    async def memory_completion_truncation_rate(self, state: dict[str, Any]) -> float:
        n = state.get("num_memory_turns", 0)
        if not n:
            return 0.0
        return float(state.get("memory_completion_truncations", 0)) / float(n)

    async def memory_parse_failure_rate(self, state: dict[str, Any]) -> float:
        n = state.get("num_memory_turns", 0)
        if not n:
            return 0.0
        return float(state.get("memory_parse_failures", 0)) / float(n)

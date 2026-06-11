"""Eval-batch MARScore panel: per-member aggregation, winner distribution,
and inert-scalar suppression of avg@k / pass@k."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

from verifiers.protocols.debate.prompts import DebatePrompts
from verifiers.protocols.debate.rubric import DebateRubric

from prime_rl.orchestrator.envs import episode_scalar_is_inert
from prime_rl.orchestrator.eval_sink import EvalSink, aggregate_mar_panel
from prime_rl.orchestrator.types import EvalRollout


def _mar(winner: str | None, *, reward_a: float = 1.0, reward_b: float = -1.0) -> dict[str, Any]:
    return {
        "members": [
            {"member_id": "debater_a", "reward": reward_a, "metrics": {"accuracy": 1.0}},
            {"member_id": "debater_b", "reward": reward_b, "parse_error_count": 1},
        ],
        "episode_scalar": 0.0,
        "episode_metrics": {"judge_confidence": 0.8},
        "episode_categorical": {"winner": winner},
    }


def _raw(*, reward: float = 0.0, mar_score: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "reward": reward,
        "error": None,
        "is_truncated": False,
        "completion": [{"role": "assistant", "content": "x"}],
        "trajectory": [],
        "token_usage": {"final_input_tokens": 10, "final_output_tokens": 5},
    }
    if mar_score is not None:
        out["mar_score"] = mar_score
    return out


def _rollout(raw: dict[str, Any], *, env_name: str = "debate", example_id: int = 0) -> EvalRollout:
    return EvalRollout(
        raw=raw,  # type: ignore[arg-type]
        env_name=env_name,
        example_id=example_id,
        group_id=uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
        eval_step=0,
    )


class _StubEnvs:
    def __init__(self, *, group_size: int, n_examples: int, inert: bool = False):
        self._env = SimpleNamespace(
            config=SimpleNamespace(group_size=group_size),
            examples=list(range(n_examples)),
            has_inert_episode_scalar=inert,
        )

    def get(self, env_name: str) -> SimpleNamespace:
        return self._env


def _process(rollouts: list[EvalRollout], *, env_name: str = "debate", group_size: int = 2, inert: bool = False):
    sink = EvalSink(eval_envs=_StubEnvs(group_size=group_size, n_examples=1, inert=inert))  # type: ignore[arg-type]
    sink.pending_batches[(env_name, 0)] = rollouts
    return sink.process_batch((env_name, 0))


def test_aggregate_mar_panel_means_and_winner_distribution():
    rollouts = [
        _raw(mar_score=_mar("debater_a")),
        _raw(mar_score=_mar("debater_b", reward_a=-1.0, reward_b=1.0)),
        _raw(mar_score=_mar("tie", reward_a=0.0, reward_b=0.0)),
        _raw(mar_score=_mar(None)),
    ]
    mar_metrics, winner_counts = aggregate_mar_panel(rollouts)
    assert mar_metrics["reward/debater_a"] == (1.0 - 1.0 + 0.0 + 1.0) / 4
    assert mar_metrics["reward/debater_b"] == (-1.0 + 1.0 + 0.0 - 1.0) / 4
    assert mar_metrics["accuracy/debater_a"] == 1.0
    assert mar_metrics["parse_errors/debater_b"] == 1.0
    assert mar_metrics["judge_confidence"] == 0.8
    assert winner_counts == {"debater_a": 1, "debater_b": 1, "tie": 1, "none": 1}


def test_aggregate_mar_panel_empty_for_single_agent_rollouts():
    assert aggregate_mar_panel([_raw(reward=1.0), _raw(reward=0.0)]) == ({}, {})


def test_structurally_inert_env_omits_avg_and_pass_at_k():
    # truth_member=None debate env: scalar ≡ 0.0 by construction.
    batch = _process(
        [_rollout(_raw(mar_score=_mar("debater_a"))), _rollout(_raw(mar_score=_mar("tie")))],
        inert=True,
    )
    assert batch.metrics.inert_scalar
    assert batch.metrics.winner_counts == {"debater_a": 1, "tie": 1}
    payload = batch.metrics.to_wandb_dict(env_name="debate", step=4)
    assert "eval/debate/avg@2" not in payload
    assert not any("pass@" in key for key in payload)
    assert payload["eval/debate/mar/reward/debater_a"] == 1.0
    assert payload["eval/debate/winner_count/tie"] == 1.0
    assert payload["eval/debate/winner_share/debater_a"] == 0.5
    # non-degenerate panels survive the suppression
    assert payload["eval/debate/num_turns/mean"] == 0.0


def test_truth_member_env_all_zero_batch_keeps_avg_at_k():
    # truth_member set ⇒ live binary scalar. A batch where the truth side
    # lost every episode is a TRUE zero — it must stay visible, not flap
    # into "no data" because the runtime observation happened to be all-zero.
    batch = _process(
        [
            _rollout(_raw(reward=0.0, mar_score=_mar("debater_b", reward_a=-1.0, reward_b=1.0))),
            _rollout(_raw(reward=0.0, mar_score=_mar("tie", reward_a=0.0, reward_b=0.0))),
        ],
        inert=False,
    )
    assert not batch.metrics.inert_scalar
    payload = batch.metrics.to_wandb_dict(env_name="debate", step=4)
    assert payload["eval/debate/avg@2"] == 0.0
    assert any("pass@" in key for key in payload)
    assert payload["eval/debate/mar/reward/debater_a"] == -0.5


def test_truth_member_env_with_live_scalar_keeps_avg_at_k():
    batch = _process([_rollout(_raw(reward=1.0, mar_score=_mar("debater_a")))])
    assert not batch.metrics.inert_scalar
    payload = batch.metrics.to_wandb_dict(env_name="debate", step=4)
    assert payload["eval/debate/avg@2"] == 1.0
    assert payload["eval/debate/mar/reward/debater_a"] == 1.0


def test_scalar_env_keeps_avg_and_pass_at_k_even_when_all_zero():
    batch = _process([_rollout(_raw(reward=0.0)), _rollout(_raw(reward=0.0))], env_name="math")
    assert not batch.metrics.inert_scalar
    assert batch.metrics.mar_metrics == {}
    payload = batch.metrics.to_wandb_dict(env_name="math", step=4)
    assert payload["eval/math/avg@2"] == 0.0
    assert any("pass@" in key for key in payload)
    assert not any("/mar/" in key or "winner" in key for key in payload)


def _debate_rubric(truth_member: str | None) -> DebateRubric:
    prompts = DebatePrompts(
        system={},
        user={},
        question={},
        fields={},
        think_visibility={},
        opponent_wrap=None,
        judges={},
        source_ref="test",
    )
    return DebateRubric(members=["debater_a", "debater_b", "judge"], prompts=prompts, truth_member=truth_member)


def test_episode_scalar_is_inert_symmetric_debate_rubric():
    env = SimpleNamespace(rubric=_debate_rubric(None))
    assert episode_scalar_is_inert(env)  # type: ignore[arg-type]


def test_episode_scalar_is_inert_false_for_truth_member_pack():
    env = SimpleNamespace(rubric=_debate_rubric("debater_a"))
    assert not episode_scalar_is_inert(env)  # type: ignore[arg-type]


def test_episode_scalar_is_inert_false_when_undeterminable():
    # Non-debate rubrics carry no structural declaration; fail toward NOT
    # suppressing — a flat-zero line is honest, a missing panel is not.
    assert not episode_scalar_is_inert(SimpleNamespace(rubric=object()))  # type: ignore[arg-type]
    assert not episode_scalar_is_inert(SimpleNamespace())  # type: ignore[arg-type]

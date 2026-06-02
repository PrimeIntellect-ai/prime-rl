"""Normalized rollout schema — the contract every component reads and writes.

One source format (verifiers ``RolloutOutput`` jsonl, one episode per line) is
normalized into ``Episode`` → ordered ``Step`` s. The same model expresses all
three cases without forking:

  - single-turn      : one Step, ``member_id`` None
  - single-agent     : N Steps, one ``member_id`` (or None), prompt grows per turn
    multi-turn
  - multi-agent      : N Steps with interleaved ``member_id`` s, ``mar`` present

The viewer branches on detected ``kind`` (see ``detect_kind``), never on a
hard-coded "debate" assumption. Debate is a *view* over the multi-agent case.

Per-seat visibility is first-class: ``Step.prompt`` is *exactly* the messages
that seat saw at that turn. Diffing ``prompt`` across consecutive same-member
steps yields the injected-this-turn delta (what the visibility-verify engine
checks).

Field mapping from the raw verifiers RolloutOutput jsonl is documented inline on
``Episode.from_rollout_output`` and ``Step.from_trajectory_step``. The token-id
arrays (``tokens.prompt_ids`` etc., ~92% of raw bytes) are intentionally dropped
here: they are reduced to ``StepDiagnostics`` upstream (the watertight strip),
never carried into the viewer artifact.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EpisodeKind = Literal["single_turn", "single_agent_multiturn", "multi_agent"]
DiagnosticsStatus = Literal["present", "masked_out", "absent"]


class Message(BaseModel):
    """One chat turn as a model saw or produced it."""

    role: str
    # ``content`` is str for text rollouts, list for structured/multimodal.
    content: str | list[Any] | None = None
    # Passthrough for tool_calls / name / any provider-specific keys, so the
    # viewer can render them without this schema enumerating every variant.
    extra: dict[str, Any] = Field(default_factory=dict)
    # Exact token count of ``extra.reasoning_content``, computed at sync with the
    # run's tokenizer (reasoning is folded into ``completion_ids`` with no array of
    # its own). ``None`` when the message carries no reasoning OR the tokenizer was
    # unavailable — the viewer then shows a char-share estimate, marked ``≈``.
    reasoning_tokens: int | None = None


class ValueSummary(BaseModel):
    """A per-(episode, member, step) reduction of a per-token quantity.

    Carries ``sum`` + ``count`` so the viewer re-aggregates as ``sum/count``
    (never mean-of-means). Quantiles/extrema are optional: the producer (W0)
    may emit the minimal ``{sum, count}`` or the rich form. ``mean`` is
    materialized for convenience and MUST equal ``sum / count``.
    """

    sum: float
    count: int
    mean: float
    p50: float | None = None
    p90: float | None = None
    p99: float | None = None
    min: float | None = None
    max: float | None = None


class StepDiagnostics(BaseModel):
    """Watertight per-step training diagnostics, joined from the W0 sidecar.

    These quantities require the trainer's forward-pass logprobs and therefore
    CANNOT be reconstructed from the rollout dump alone — they are captured at
    the source (see ``contracts.md`` → diagnostics sidecar) and joined on
    ``(trajectory_id, member_id, step_index)``.

    ``status`` is load-bearing and never lied about:
      - "present"    : real diagnostics computed over ``n_tokens`` tokens
      - "masked_out" : the sequence was fully clipped by ``mask_ratio_*`` — it
                       contributed no gradient, so there is no IS/KL to report.
                       This is a real state, NOT zero.
      - "absent"     : no sidecar row matched (e.g. diagnostics emission was
                       disabled for the run). The join MUST raise instead of
                       fabricating this when a match was expected; "absent" is
                       only set when the whole run lacks a sidecar.

    A missing summary is ``None``, never ``0.0``.
    """

    status: DiagnosticsStatus
    n_tokens: int | None = None
    importance_ratio: ValueSummary | None = None  # exp(trainer_lp - inference_lp)
    mismatch_kl: ValueSummary | None = None  # k3 estimator: exp(d) - d - 1
    entropy: ValueSummary | None = None
    masked_low_frac: float | None = None  # fraction below mask_ratio_low
    masked_high_frac: float | None = None  # fraction above mask_ratio_high


class Step(BaseModel):
    """One generation in an episode's trajectory."""

    index: int  # position within the episode trajectory (0-based)
    member_id: str | None  # debate seat / agent id; None for a lone anonymous agent
    phase: str | None = None  # extras.phase (e.g. debate phase), if present
    prompt: list[Message]  # EXACTLY what this seat saw at this turn (visibility)
    completion: list[Message]  # what this seat produced
    # Lengths of the raw ``tokens.{prompt,completion}_ids`` arrays, captured BEFORE
    # those (~92% of raw bytes) are dropped. ``None`` only when the raw step carried
    # no ``tokens`` key at all; an external-API seat with ``tokens: null`` (no local
    # id capture) reads ``0`` — a real zero, not a missing value.
    n_prompt_tokens: int | None = None
    n_completion_tokens: int | None = None
    reward: float | None = None
    advantage: float | None = None
    is_truncated: bool = False
    diagnostics: StepDiagnostics | None = None  # joined from W0 sidecar; None pre-join
    extras: dict[str, Any] = Field(default_factory=dict)  # generation idx, fields, ...

    @classmethod
    def from_trajectory_step(cls, raw: dict[str, Any], index: int) -> "Step":
        """Map a raw verifiers ``TrajectoryStep`` to a normalized ``Step``.

        Source keys (verifiers ``TrajectoryStep``):
          prompt, completion, response, tokens, reward, advantage,
          is_truncated, trajectory_id, extras{member_id, phase, generation, fields}

        ``tokens`` (the id/mask/logprob arrays) is deliberately NOT carried —
        it is reduced to ``StepDiagnostics`` upstream.
        """
        extras = dict(raw.get("extras") or {})
        n_prompt, n_completion = _token_counts(raw)
        return cls(
            index=index,
            member_id=extras.get("member_id"),
            phase=extras.get("phase"),
            prompt=_messages(raw.get("prompt")),
            completion=_messages(raw.get("completion")),
            n_prompt_tokens=n_prompt,
            n_completion_tokens=n_completion,
            reward=raw.get("reward"),
            advantage=raw.get("advantage"),
            is_truncated=bool(raw.get("is_truncated", False)),
            extras=extras,
        )


class MemberOutcome(BaseModel):
    """Per-member episode outcome (from ``mar_score.members``)."""

    member_id: str
    reward: float
    parse_error_count: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)


class MultiAgentScore(BaseModel):
    """Episode-level multi-agent score (from ``mar_score``)."""

    members: list[MemberOutcome] = Field(default_factory=list)
    episode_scalar: float | None = None
    episode_metrics: dict[str, float] = Field(default_factory=dict)
    # e.g. {"winner": "debater_a", "final_answer/debater_a": "C", ...}
    categorical: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_mar_score(cls, raw: dict[str, Any]) -> "MultiAgentScore":
        return cls(
            members=[MemberOutcome(**m) for m in raw.get("members", [])],
            episode_scalar=raw.get("episode_scalar"),
            episode_metrics=dict(raw.get("episode_metrics") or {}),
            categorical={k: str(v) for k, v in (raw.get("episode_categorical") or {}).items()},
        )


class Episode(BaseModel):
    """A single rollout episode, normalized for the viewer."""

    # identity
    example_id: str
    rollout_id: str | None = None
    trajectory_id: str | None = None
    run_id: str  # which experiment run (cross-run axis)
    step: int  # training step (which rollouts/step_N produced it)

    # structure
    steps: list[Step] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)  # distinct ids, first-seen order
    kind: EpisodeKind

    # episode-level outcome
    reward: float | None = None
    advantage: float | None = None
    mar: MultiAgentScore | None = None
    metrics: dict[str, float] = Field(default_factory=dict)  # flat per-episode metrics
    is_filtered: bool = False
    error: str | None = None

    # task
    task: dict[str, Any] = Field(default_factory=dict)  # question / answer / info

    @classmethod
    def from_rollout_output(cls, raw: dict[str, Any], *, run_id: str, step: int) -> "Episode":
        """Normalize one raw verifiers ``RolloutOutput`` jsonl record.

        Source keys used (measured from a real prime-rl debate dump):
          example_id, rollout_id, trajectory_id, task, prompt, completion,
          reward, advantage, is_filtered, error, metrics, mar_score, trajectory,
          and flat ``<metric>/<member>`` keys (reward/debater_a, flipped/..., ...).

        The diagnostics join (attach ``StepDiagnostics`` to each Step) is a
        SEPARATE step owned by the parser worker (W2), because it depends on the
        W0 sidecar which is produced in parallel. This method returns Steps with
        ``diagnostics=None``.
        """
        traj = raw.get("trajectory") or []
        steps = [Step.from_trajectory_step(s, i) for i, s in enumerate(traj)]
        members: list[str] = []
        for s in steps:
            if s.member_id is not None and s.member_id not in members:
                members.append(s.member_id)
        mar = MultiAgentScore.from_mar_score(raw["mar_score"]) if raw.get("mar_score") is not None else None
        return cls(
            example_id=str(raw["example_id"]),
            rollout_id=_opt_str(raw.get("rollout_id")),
            trajectory_id=_opt_str(raw.get("trajectory_id")),
            run_id=run_id,
            step=step,
            steps=steps,
            members=members,
            kind=detect_kind(steps, mar),
            reward=raw.get("reward"),
            advantage=raw.get("advantage"),
            mar=mar,
            metrics=_flat_metrics(raw),
            is_filtered=bool(raw.get("is_filtered", False)),
            error=_opt_str(raw.get("error")),
            task=dict(raw.get("task") or {}),
        )


def detect_kind(steps: list[Step], mar: MultiAgentScore | None) -> EpisodeKind:
    """Classify episode structure. Multi-agent iff ≥2 seats or an MAR score."""
    distinct = {s.member_id for s in steps if s.member_id is not None}
    if len(distinct) >= 2 or mar is not None:
        return "multi_agent"
    if len(steps) > 1:
        return "single_agent_multiturn"
    return "single_turn"


# --- raw-record helpers -------------------------------------------------------

# Top-level RolloutOutput keys that are NOT flat metrics (mirrors verifiers'
# RESERVED_ROLLOUT_OUTPUT_KEYS plus viewer-internal fields).
_RESERVED: frozenset[str] = frozenset(
    {
        "example_id",
        "rollout_id",
        "trajectory_id",
        "task",
        "prompt",
        "completion",
        "answer",
        "info",
        "reward",
        "advantage",
        "timing",
        "is_completed",
        "is_truncated",
        "stop_condition",
        "metrics",
        "error",
        "trajectory",
        "tool_defs",
        "token_usage",
        "mar_score",
        "sampling_args",
        "env_name",
        "filters",
        "is_filtered",
        "multi_agent_dispatch_id",
        "agreement",
    }
)


def _flat_metrics(raw: dict[str, Any]) -> dict[str, float]:
    """Collect the flat ``<metric>/<member>`` scalar keys + ``metrics`` dict."""
    out: dict[str, float] = {}
    for k, v in (raw.get("metrics") or {}).items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    for k, v in raw.items():
        if k in _RESERVED:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    if isinstance(raw.get("agreement"), (int, float)):
        out["agreement"] = float(raw["agreement"])
    return out


def _token_counts(raw: dict[str, Any]) -> tuple[int | None, int | None]:
    """Lengths of ``tokens.{prompt,completion}_ids``, captured before the strip.

    Three cases, no silent fallbacks:
      - ``tokens`` key absent              → ``(None, None)`` (genuinely no token data).
      - ``tokens`` is ``None``             → ``(0, 0)`` — an external-API seat (e.g. the
                                             judge) with no LOCAL id capture; a real zero.
      - ``tokens`` is a dict               → ``(len(prompt_ids), len(completion_ids))``;
                                             a present-but-non-list id field raises.
    """
    if "tokens" not in raw:
        return None, None
    tokens = raw["tokens"]
    if tokens is None:
        return 0, 0
    if not isinstance(tokens, dict):
        raise TypeError(f"tokens must be a mapping or null, got {type(tokens).__name__}: {tokens!r}")
    return _id_len(tokens, "prompt_ids"), _id_len(tokens, "completion_ids")


def _id_len(tokens: dict[str, Any], key: str) -> int:
    """Length of a token-id array in a ``tokens`` dict; raise if malformed.

    A real token dict carries the id arrays as lists. A missing/None/non-list field
    is a malformed dump — raise rather than guessing a count or silently returning 0.
    """
    ids = tokens.get(key)
    if not isinstance(ids, list):
        raise TypeError(
            f"tokens.{key} must be a list of ids, got {type(ids).__name__} "
            f"({ids!r}) — refusing to fabricate a token count from a malformed dump"
        )
    return len(ids)


def _messages(raw: Any) -> list[Message]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError(f"expected a list of messages, got {type(raw).__name__}")
    msgs: list[Message] = []
    for m in raw:
        if not isinstance(m, dict):
            raise TypeError(f"message must be a mapping, got {type(m).__name__}")
        known = {"role", "content"}
        msgs.append(
            Message(
                role=m.get("role", ""),
                content=m.get("content"),
                extra={k: v for k, v in m.items() if k not in known},
            )
        )
    return msgs


def _opt_str(v: Any) -> str | None:
    return None if v is None else str(v)

"""Parser (W2) — raw rollout jsonl ⊕ diagnostics sidecar → joined ``Episode`` s.

``load_episodes`` reads ``train_rollouts.jsonl`` (one verifiers ``RolloutOutput``
per line) via ``Episode.from_rollout_output``, then — iff a
``train_diagnostics.jsonl`` sidecar sits next to it — joins ``StepDiagnostics``
onto every ``Step`` on the exact key ``(trajectory_id, member_id, step_index)``.

The join is *watertight* (see ``contracts.md`` → "The join"):
  - exact key, never align-by-position, never default-fill;
  - a sidecar row that should match but doesn't → raise (a mismatch is a bug,
    not a missing value), UNLESS the episode is errored/filtered (the trainer
    legitimately emits no diagnostics for those);
  - no sidecar file at all → every Step gets ``status="absent"`` — the *only*
    path that sets "absent", explicit, never inferred per-step;
  - ``status="masked_out"`` rows attach verbatim (real state, not zeroed).

The schema (``detect_kind``) already collapses single-turn / single-agent /
multi-agent into one model, so this parser is debate-agnostic: it joins by key
and never assumes a seat layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rollout_viewer.schema import Episode, StepDiagnostics

ROLLOUTS_FILENAME = "train_rollouts.jsonl"
DIAGNOSTICS_FILENAME = "train_diagnostics.jsonl"

# The exact diagnostics-join key: (trajectory_id, member_id, step_index).
DiagKey = tuple[str | None, str | None, int]


def load_episodes(step_dir: str | Path, run_id: str, step: int) -> list[Episode]:
    """Load + normalize one training step's rollouts, with diagnostics joined.

    Reads ``<step_dir>/train_rollouts.jsonl`` into ``Episode`` s, then joins
    ``<step_dir>/train_diagnostics.jsonl`` if it exists. If the sidecar is absent
    every Step's ``diagnostics`` is set to ``StepDiagnostics(status="absent")``.
    """
    step_dir = Path(step_dir)
    rollouts_path = step_dir / ROLLOUTS_FILENAME
    if not rollouts_path.exists():
        raise FileNotFoundError(f"no rollouts file at {rollouts_path}")

    episodes: list[Episode] = []
    with rollouts_path.open() as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{rollouts_path}:{lineno} is not valid JSON: {e}") from e
            episodes.append(Episode.from_rollout_output(raw, run_id=run_id, step=step))

    join_diagnostics(episodes, step_dir / DIAGNOSTICS_FILENAME)
    return episodes


def join_diagnostics(episodes: list[Episode], diag_path: str | Path) -> None:
    """Attach ``StepDiagnostics`` to every Step in ``episodes``, in place.

    Watertight contract:
      - sidecar absent  → every Step ← ``StepDiagnostics(status="absent")``;
      - sidecar present → every Step is keyed on
        ``(episode.trajectory_id, step.member_id, step.index)`` against the
        sidecar; a step with no matching row raises *unless* its episode is
        errored or filtered. A sidecar row matched by no step also raises
        (the sidecar carries an entry the rollouts can't account for).
    """
    diag_path = Path(diag_path)

    if not diag_path.exists():
        # The ONLY path that sets "absent": the whole run lacks a sidecar.
        # A FRESH instance per step — a shared one would alias every Step to one
        # object, so mutating any Step's diagnostics would flip them all.
        for ep in episodes:
            for s in ep.steps:
                s.diagnostics = StepDiagnostics(status="absent")
        return

    by_key, dup = _load_sidecar(diag_path)
    if dup:
        raise ValueError(
            f"{diag_path}: duplicate diagnostics key(s) {sorted(dup)} — the join key must be unique per row"
        )

    consumed: set[DiagKey] = set()
    for ep in episodes:
        skip_missing = ep.error is not None or ep.is_filtered
        for s in ep.steps:
            key: DiagKey = (ep.trajectory_id, s.member_id, s.index)
            row = by_key.get(key)
            if row is None:
                if skip_missing:
                    # Trainer legitimately emits no diagnostics for an episode
                    # that produced no gradient (errored / filtered out).
                    s.diagnostics = StepDiagnostics(status="absent")
                    continue
                raise KeyError(
                    f"{diag_path}: no diagnostics row for key {key} "
                    f"(episode example_id={ep.example_id!r}, run_id={ep.run_id!r}, "
                    f"step={ep.step}). A sidecar exists and this episode is "
                    f"neither errored nor filtered, so a row was expected. "
                    f"Refusing to align-by-position or fill defaults."
                )
            s.diagnostics = StepDiagnostics.model_validate(row)
            consumed.add(key)

    unmatched = set(by_key) - consumed
    if unmatched:
        raise KeyError(
            f"{diag_path}: {len(unmatched)} diagnostics row(s) matched no "
            f"rollout step: {sorted(unmatched)}. The sidecar carries entries "
            f"the rollouts can't account for — the join is not watertight."
        )


def _load_sidecar(
    diag_path: Path,
) -> tuple[dict[DiagKey, dict[str, Any]], set[DiagKey]]:
    """Read the sidecar jsonl into ``{(traj, member, step_index): row}``.

    Returns the map and the set of keys seen more than once (duplicates are a
    fatal sidecar defect — the join key must be unique). Each row must carry the
    three key fields and a ``status``; a malformed line raises immediately.
    """
    by_key: dict[DiagKey, dict[str, Any]] = {}
    dup: set[DiagKey] = set()
    with diag_path.open() as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{diag_path}:{lineno} is not valid JSON: {e}") from e
            for required in ("trajectory_id", "member_id", "step_index", "status"):
                if required not in row:
                    raise KeyError(f"{diag_path}:{lineno} diagnostics row missing required key {required!r}: {row!r}")
            step_index = row["step_index"]
            # The join key is exact: a float step_index (2.7 -> 2) would silently
            # truncate and misjoin. Require a real int (bool is not a step index).
            if not isinstance(step_index, int) or isinstance(step_index, bool):
                raise TypeError(
                    f"{diag_path}:{lineno} step_index must be an int, got "
                    f"{type(step_index).__name__} ({step_index!r}): {row!r}. "
                    f"Refusing to truncate-and-misjoin."
                )
            key: DiagKey = (
                row["trajectory_id"],
                row["member_id"],
                step_index,
            )
            if key in by_key:
                dup.add(key)
            by_key[key] = row
    return by_key, dup

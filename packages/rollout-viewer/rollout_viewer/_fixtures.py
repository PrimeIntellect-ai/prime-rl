"""Synthetic diagnostics producer — stand-in for the W0 trainer sidecar.

W0 (the trainer emitting ``train_diagnostics.jsonl``) is built in parallel, so
the join (W2) needs a way to be exercised before it lands. ``make_synthetic_diagnostics``
walks a list of parsed ``Episode`` s and emits one sidecar row per Step on the
exact join key — with a *rich* ``ValueSummary`` (sum/count/mean + quantiles +
extrema) so the present-path is tested against the full shape, not the minimal
``{sum, count, mean}``.

The values are deterministic per key (seeded by the key tuple), not random, so
fixtures are reproducible and the prove-it tests can assert exact counts/means.
This is test scaffolding — it lives under ``_fixtures`` and is never imported by
the runtime path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rollout_viewer.schema import Episode

# Per-token quantities the sidecar reduces (contracts.md → diagnostics sidecar).
_SUMMARY_FIELDS = ("importance_ratio", "mismatch_kl", "entropy")


def make_synthetic_diagnostics(
    episodes: list[Episode],
    diag_path: str | Path,
    *,
    n_tokens_per_step: int = 9458,
) -> Path:
    """Write a ``train_diagnostics.jsonl`` covering every Step in ``episodes``.

    Skips episodes the trainer wouldn't emit diagnostics for (errored / filtered),
    mirroring W0's behaviour — those steps legitimately have no sidecar row, and
    the join treats their absence as expected. One rich-``ValueSummary`` row per
    remaining ``(trajectory_id, member_id, step_index)``.
    """
    diag_path = Path(diag_path)
    rows: list[dict] = []
    for ep in episodes:
        if ep.error is not None or ep.is_filtered:
            continue
        for s in ep.steps:
            rows.append(
                _row(
                    trajectory_id=ep.trajectory_id,
                    member_id=s.member_id,
                    step_index=s.index,
                    n_tokens=n_tokens_per_step,
                )
            )
    with diag_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return diag_path


def _row(
    *,
    trajectory_id: str | None,
    member_id: str | None,
    step_index: int,
    n_tokens: int,
) -> dict:
    row: dict = {
        "trajectory_id": trajectory_id,
        "member_id": member_id,
        "step_index": step_index,
        "n_tokens": n_tokens,
        "status": "present",
        "masked_low_frac": 0.0,
        "masked_high_frac": 0.0,
    }
    for field in _SUMMARY_FIELDS:
        row[field] = _summary(
            seed=f"{trajectory_id}|{member_id}|{step_index}|{field}",
            count=n_tokens,
        )
    return row


def _summary(*, seed: str, count: int) -> dict:
    """A rich ``ValueSummary`` dict whose ``mean`` is exactly ``sum / count``.

    The per-token mean is derived deterministically from ``seed`` so a given key
    always yields the same numbers; ``sum = mean * count`` keeps the invariant
    ``mean == sum / count`` that the viewer relies on.
    """
    h = hashlib.sha256(seed.encode()).digest()
    # Deterministic per-token mean in (0.5, 1.5), away from the trivial 1.0.
    mean = 0.5 + (int.from_bytes(h[:8], "big") / 2**64)
    total = mean * count
    spread = mean * 0.1
    return {
        "sum": total,
        "count": count,
        "mean": mean,
        "p50": mean,
        "p90": mean + spread,
        "p99": mean + 2 * spread,
        "min": mean - spread,
        "max": mean + 2 * spread,
    }

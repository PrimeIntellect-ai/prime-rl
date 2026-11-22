"""Notebook-facing utilities for loading historical per-rollout records.

prime-rl writes per-rollout JSONL records to:

    <output_dir>/run_default/rollouts/step_<N>/train_rollouts.jsonl
    <output_dir>/run_default/rollouts/step_<N>/eval_rollouts.jsonl

This module flattens those into a pandas DataFrame for analysis. Bulky
per-trajectory fields (prompt, completion, trajectory) are dropped;
``env_id`` is pulled out of ``info`` since it's the primary grouping key.

pandas is lazy-imported so importing this module from environments without
pandas (e.g. minimal CI) does not crash.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fields we drop from the row dict by default — too bulky or already pulled out.
_DROP_FIELDS = frozenset(
    {"prompt", "completion", "trajectory", "tool_defs", "info"}
)

_STEP_RE = re.compile(r"^step_(\d+)$")

# Glyphbench env ids look like "glyphbench/<suite>-<game>-<variant>-v<n>". The
# "suite" is the first hyphen-delimited token after the slash. Examples:
#   glyphbench/atari-skiing-v0           → atari
#   glyphbench/minigrid-doorkey-16x16-v0 → minigrid
#   glyphbench/classics-mirrorlaser-v0   → classics
#   glyphbench/craftax-find-water-v0     → craftax
#   glyphbench/minihack-keyroom-dark-s15-v0 → minihack
#   glyphbench/procgen-maze-v0           → procgen
_SUITE_RE = re.compile(r"^glyphbench/([^/-]+)-")


def _suite_from_env_id(env_id: Any) -> Any:
    if not isinstance(env_id, str):
        return None
    m = _SUITE_RE.match(env_id)
    return m.group(1) if m else None


def _import_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "load_rollouts requires pandas; install with `pip install pandas`"
        ) from e
    return pd


def _coerce_info(info: Any) -> dict[str, Any]:
    """Return info as a dict; tolerate JSON-encoded strings and None."""
    if info is None:
        return {}
    if isinstance(info, dict):
        return info
    if isinstance(info, str):
        try:
            parsed = json.loads(info)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _iter_rollout_records(
    rollouts_dir: Path, phases: Iterable[str]
) -> Iterable[dict[str, Any]]:
    phases = tuple(phases)
    if not rollouts_dir.exists():
        logger.warning("rollouts directory does not exist: %s", rollouts_dir)
        return
    for step_dir in sorted(rollouts_dir.iterdir()):
        if not step_dir.is_dir():
            continue
        m = _STEP_RE.match(step_dir.name)
        if m is None:
            continue
        step = int(m.group(1))
        for phase in phases:
            f = step_dir / f"{phase}_rollouts.jsonl"
            if not f.exists():
                continue
            with f.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "skipping unparseable record %s:%d: %s", f, lineno, e
                        )
                        continue
                    if not isinstance(rec, dict):
                        logger.warning("skipping non-object record %s:%d", f, lineno)
                        continue
                    info = _coerce_info(rec.get("info"))
                    out: dict[str, Any] = {
                        k: v for k, v in rec.items() if k not in _DROP_FIELDS
                    }
                    out["step"] = step
                    out["phase"] = phase
                    out["env_id"] = info.get("env_id")
                    out["seed"] = info.get("seed")
                    out["suite"] = _suite_from_env_id(out["env_id"])
                    yield out


def load_rollouts(
    output_dir: Path | str,
    phases: Iterable[str] = ("train", "eval"),
) -> Any:
    """Read all rollout JSONL files under ``output_dir/run_default/rollouts/``
    and return one row per rollout as a pandas DataFrame.

    Columns include: step, phase, env_id, task, example_id, reward,
    episodic_return, episode_length, num_turns, forfeit_rate,
    xml_format_reward, is_completed, is_truncated, advantage. Bulky fields
    (prompt, completion, trajectory, info) are dropped — but
    env_id is pulled out of info.

    Legacy rollout files (pre-P3) may use ``parse_failure_rate`` instead of
    ``forfeit_rate``; both are coalesced into ``forfeit_rate`` automatically.
    """
    pd = _import_pandas()
    output_dir = Path(output_dir)
    rollouts_dir = output_dir / "run_default" / "rollouts"
    rows = list(_iter_rollout_records(rollouts_dir, phases))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Coalesce legacy parse_failure_rate → forfeit_rate for old JSONL files.
    if "forfeit_rate" not in df.columns and "parse_failure_rate" in df.columns:
        df["forfeit_rate"] = df["parse_failure_rate"]
    sort_cols = [c for c in ("step", "phase", "example_id") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df


def _summary_by(df: Any, group_col: str) -> Any:
    pd = _import_pandas()
    columns = [
        group_col,
        "count",
        "mean_reward",
        "std_reward",
        "mean_episodic_return",
        "mean_episode_length",
        "mean_num_turns",
        "mean_forfeit_rate",
        "mean_xml_format_reward",
        "completion_rate",
        "truncation_rate",
    ]
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    if "is_completed" in work.columns:
        work["_is_completed_f"] = work["is_completed"].astype(float)
    if "is_truncated" in work.columns:
        work["_is_truncated_f"] = work["is_truncated"].astype(float)

    agg_spec: dict[str, tuple[str, str]] = {"count": (group_col, "size")}
    for out_name, src_col, op in (
        ("mean_reward", "reward", "mean"),
        ("std_reward", "reward", "std"),
        ("mean_episodic_return", "episodic_return", "mean"),
        ("mean_episode_length", "episode_length", "mean"),
        ("mean_num_turns", "num_turns", "mean"),
        ("mean_forfeit_rate", "forfeit_rate", "mean"),
        ("mean_xml_format_reward", "xml_format_reward", "mean"),
        ("completion_rate", "_is_completed_f", "mean"),
        ("truncation_rate", "_is_truncated_f", "mean"),
    ):
        if src_col in work.columns:
            agg_spec[out_name] = (src_col, op)

    grouped = work.groupby(group_col, dropna=False)
    out = grouped.agg(**agg_spec).reset_index()
    out = out.sort_values("count", ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    return out


def summary_by_env(df: Any) -> Any:
    """Aggregate a DataFrame from :func:`load_rollouts` by ``env_id``.

    One row per env with: count, mean_reward, std_reward,
    mean_episodic_return, mean_episode_length, mean_num_turns,
    mean_forfeit_rate, mean_xml_format_reward, completion_rate,
    truncation_rate. Sorted by count descending.
    """
    return _summary_by(df, "env_id")


def summary_by_suite(df: Any) -> Any:
    """Aggregate by ``suite`` (parsed from env_id, e.g. ``atari`` or
    ``minigrid``). Same column set as :func:`summary_by_env`.
    """
    return _summary_by(df, "suite")

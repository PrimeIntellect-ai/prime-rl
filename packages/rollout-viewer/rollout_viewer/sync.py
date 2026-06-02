"""``rollout-sync`` — strip → join → gzip/parquet → store, one training step.

Reads a ``rollouts/`` tree of ``step_N/train_rollouts.jsonl`` (the verifiers
``RolloutOutput`` dump), normalizes each line into an ``Episode`` via W2's
``parser.load_episodes`` (which joins the W0 diagnostics sidecar, or stamps every
Step ``status="absent"`` when there is no sidecar), strips token arrays
(structural — ``Episode`` never carries them), applies retention ``ρ``, and
writes index + transcript shards through a ``StorageBackend``.

Retention semantics (``contracts.md`` → Retention): metrics + diagnostics for
EVERY episode are always written to the index. ``ρ`` only gates which episodes'
full TRANSCRIPTS are shipped — a ρ-fraction (deterministic, by source order). The
index still points at the shard; non-retained episodes simply have no line there,
which the index marks with ``transcript_line = -1``.

Fail-loud ceiling: the sync WARNS once the store crosses ``SOFT_LIMIT_BYTES`` and
the backend itself REFUSES to write past ``HARD_LIMIT_BYTES``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path

from rollout_viewer.index import (
    update_runs_registry,
    write_index,
    write_setup,
    write_transcript_shard,
)
from rollout_viewer.parser import load_episodes
from rollout_viewer.schema import Episode
from rollout_viewer.setup import load_run_setup
from rollout_viewer.storage import (
    SOFT_LIMIT_BYTES,
    StorageBackend,
    index_path,
)
from rollout_viewer.storage_backends import HuggingFaceBackend, LocalFSBackend

logger = logging.getLogger("rollout_viewer.sync")

# Source-tree convention (mirrors prime_rl's get_step_path: path / f"step_{step}").
STEP_DIR_GLOB = "step_*"
ROLLOUTS_FILENAME = "train_rollouts.jsonl"
NOT_RETAINED = -1  # transcript_line sentinel for episodes whose transcript we dropped


def _step_of_dir(d: Path) -> int:
    name = d.name
    if not name.startswith("step_"):
        raise ValueError(f"not a step dir: {d}")
    return int(name[len("step_") :])


ReasoningEncoder = Callable[[str], int]


def _build_reasoning_encoder(setup: dict | None) -> ReasoningEncoder | None:
    """Best-effort ``fn(reasoning_text) -> token_count`` from the run's tokenizer.

    Reasoning is folded into ``completion_ids`` with no array of its own, so the
    only exact count comes from re-encoding the text with the SAME tokenizer the
    model used — a decode→encode round-trip is identity for in-vocab text (verified:
    matches the ``</think>`` boundary in ``completion_ids`` to ±1). Returns ``None``
    — the viewer then shows a client-side char-share estimate, marked ``≈`` — when
    there is no config, no model name, or the tokenizer can't load. A load failure is
    logged loud, never silently swallowed.
    """
    if not setup:
        return None
    model_name = (setup.get("model") or {}).get("name")
    if not model_name:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        logger.warning(
            "`tokenizers` not installed — reasoning token counts will be estimated "
            "client-side (sync with `--with tokenizers`, or add it to the env)."
        )
        return None
    try:
        tok = Tokenizer.from_pretrained(model_name)
    except Exception as e:  # noqa: BLE001 — any load failure degrades to the estimate
        logger.warning(
            "could not load tokenizer %r (%s: %s) — reasoning token counts will be estimated client-side",
            model_name,
            type(e).__name__,
            e,
        )
        return None
    return lambda s: len(tok.encode(s, add_special_tokens=False).ids)


def _annotate_reasoning_tokens(episodes: list[Episode], encode: ReasoningEncoder) -> None:
    """Set ``Message.reasoning_tokens`` on every message carrying reasoning text."""
    for ep in episodes:
        for st in ep.steps:
            for m in (*st.prompt, *st.completion):
                rc = m.extra.get("reasoning_content")
                if isinstance(rc, str) and rc:
                    m.reasoning_tokens = encode(rc)


def _load_step_episodes(
    step_dir: Path,
    *,
    run_id: str,
    step: int,
    encode: ReasoningEncoder | None = None,
) -> list[Episode]:
    """Load + normalize one step's episodes via W2's watertight diagnostics join.

    Always routes through ``parser.load_episodes``, which performs the join and —
    for a no-sidecar run — sets every Step's ``diagnostics`` to the contract
    status ``"absent"``. There is no raw fallback: ``Episode.from_rollout_output``
    leaves ``diagnostics=None``, which the rollup would mislabel ``"none"`` and
    silently strip the watertight-absent guarantee.

    When ``encode`` is given, annotates ``Message.reasoning_tokens`` before return.
    """
    rollouts = step_dir / ROLLOUTS_FILENAME
    if not rollouts.exists():
        raise FileNotFoundError(f"no {ROLLOUTS_FILENAME} in {step_dir}")

    episodes = load_episodes(step_dir, run_id=run_id, step=step)
    if not episodes:
        raise ValueError(f"parser.load_episodes returned no episodes for {step_dir}")
    episodes = list(episodes)
    if encode is not None:
        _annotate_reasoning_tokens(episodes, encode)
    return episodes


def _retained_mask(n: int, rho: float) -> list[bool]:
    """Deterministic retention mask: first ``ceil(rho*n)`` episodes keep transcripts.

    By source order so a re-sync is byte-stable and the index/shard agree. ρ only
    gates transcript retention; metrics + diagnostics are always indexed.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    keep = math.ceil(rho * n)
    return [i < keep for i in range(n)]


def _check_soft_limit(backend: StorageBackend) -> None:
    total = backend.stat().total_bytes
    if total >= SOFT_LIMIT_BYTES:
        logger.warning(
            "store at %d bytes — past SOFT_LIMIT_BYTES=%d. Approaching the HF-free "
            "ceiling; plan the R2 migration (see contracts.md).",
            total,
            SOFT_LIMIT_BYTES,
        )


def sync_step(
    step_dir: Path,
    backend: StorageBackend,
    *,
    run_id: str,
    rho: float = 1.0,
    encode: ReasoningEncoder | None = None,
) -> dict:
    """Sync ONE step dir: write its transcript shard + registry entry.

    Returns a summary dict. Does NOT write the index — the index is a whole-run
    parquet rewritten once all steps are loaded (see ``main``); callers that sync a
    single step in isolation should follow with ``write_index`` over the run's
    episodes.
    """
    step = _step_of_dir(step_dir)
    episodes = _load_step_episodes(step_dir, run_id=run_id, step=step, encode=encode)

    mask = _retained_mask(len(episodes), rho)
    retained = [ep for ep, keep in zip(episodes, mask) if keep]

    # Empty retained list (ρ rounds to 0) → no transcript to ship. The index
    # already marks every row in this step transcript_line=-1 (NOT_RETAINED), so
    # writing a zero-line shard would be a dead, misleading object.
    if retained:
        write_transcript_shard(retained, backend, run_id, step)
    update_runs_registry(backend, run_id, step)
    _check_soft_limit(backend)

    return {
        "step": step,
        "n_episodes": len(episodes),
        "n_retained": len(retained),
        "episodes": episodes,
        "mask": mask,
    }


def sync_source(
    source: Path,
    backend: StorageBackend,
    *,
    run_id: str,
    rho: float = 1.0,
    configs_dir: Path | None = None,
    encode: ReasoningEncoder | None = None,
) -> dict:
    """Sync every ``step_*`` dir under ``source``, then write the whole-run index.

    When ``configs_dir`` is given, also distill the run's ``orchestrator.toml`` into
    ``index/{run_id}/setup.json`` (the declared protocol the ``/setup`` endpoint
    serves). A configs dir that doesn't parse is a real error — it raises, never
    silently skips the setup.

    ``encode`` (``fn(reasoning_text) -> n_tokens``) overrides the tokenizer derived
    from the config — pass it to annotate exact reasoning-token counts for a run
    whose config is elsewhere. When omitted it is built from the config's model.
    """
    step_dirs = sorted(
        (d for d in source.glob(STEP_DIR_GLOB) if d.is_dir()),
        key=_step_of_dir,
    )
    if not step_dirs:
        raise FileNotFoundError(f"no {STEP_DIR_GLOB} dirs under {source}")

    # Load the run setup ONCE: it both feeds the /setup endpoint and names the model
    # whose tokenizer gives exact reasoning-token counts. No config ⇒ no tokenizer ⇒
    # the viewer estimates reasoning tokens client-side (marked ``≈``).
    setup = load_run_setup(configs_dir) if configs_dir is not None else None
    if encode is None:
        encode = _build_reasoning_encoder(setup)

    all_episodes: list[Episode] = []
    transcript_lines: list[int] = []
    per_step: list[dict] = []

    for step_dir in step_dirs:
        result = sync_step(step_dir, backend, run_id=run_id, rho=rho, encode=encode)
        per_step.append({k: v for k, v in result.items() if k not in ("episodes", "mask")})
        # transcript_line: retained episodes get their 0-based shard offset (source
        # order); dropped episodes get NOT_RETAINED.
        line = 0
        for ep, keep in zip(result["episodes"], result["mask"]):
            transcript_lines.append(line if keep else NOT_RETAINED)
            if keep:
                line += 1
        all_episodes.extend(result["episodes"])

    write_index(all_episodes, backend, run_id, transcript_lines)

    setup_logical: str | None = None
    if setup is not None:
        setup_logical = write_setup(setup, backend, run_id)

    _check_soft_limit(backend)

    if isinstance(backend, HuggingFaceBackend):
        backend.flush(commit_message=f"rollout-sync {run_id}: {len(step_dirs)} step(s)")

    return {
        "run_id": run_id,
        "n_steps": len(step_dirs),
        "n_episodes": len(all_episodes),
        "index_path": index_path(run_id),
        "setup_path": setup_logical,
        "per_step": per_step,
    }


def _make_backend(kind: str, dest: str) -> StorageBackend:
    if kind == "local":
        return LocalFSBackend(dest)
    if kind == "hf":
        return HuggingFaceBackend(dest)
    raise ValueError(f"unknown backend {kind!r} (expected 'local' or 'hf')")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="rollout-sync",
        description="Strip + gzip + parquet rollouts into a viewer store.",
    )
    parser.add_argument("--source", required=True, type=Path, help="dir containing step_N/ subdirs")
    parser.add_argument("--run-id", required=True, help="experiment run id (cross-run axis)")
    parser.add_argument("--backend", required=True, choices=("local", "hf"), help="storage backend")
    parser.add_argument("--dest", required=True, help="local store path (local) or HF dataset repo id (hf)")
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help="transcript retention fraction in [0,1] (metrics+diagnostics always kept)",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help="dir with the run's orchestrator/inference/trainer tomls (writes setup.json)",
    )
    args = parser.parse_args()

    backend = _make_backend(args.backend, args.dest)
    summary = sync_source(
        args.source,
        backend,
        run_id=args.run_id,
        rho=args.rho,
        configs_dir=args.configs_dir,
    )
    logger.info(
        "synced run=%s steps=%d episodes=%d → index=%s",
        summary["run_id"],
        summary["n_steps"],
        summary["n_episodes"],
        summary["index_path"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

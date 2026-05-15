"""Remove DCP training-state dirs for completed checkpoints, keeping weights.

prime-rl's SFT checkpoint layout is TWO parallel dirs per step:
  outputs/<run>/weights/step_N/            — HF-format weights, ~15 GB (kept for eval)
  outputs/<run>/checkpoints/step_N/trainer/ — FSDP DCP optimizer/scheduler/dataloader
                                              state, ~90 GB (only needed for resume)

During a run, `[ckpt] keep_interval=N keep_last=1` retains the FULL
training-state ckpt at every N-multiple, which explodes disk usage (e.g.
20 periodic ckpts × 90 GB = 1.8 TB). prime-rl's `weights_only` flag is
global — setting it true breaks mid-training resume.

This script is the post-hoc fix: after training finishes successfully, call
this to delete every `checkpoints/step_N/` except the final one. Weights
(needed for eval) stay untouched under `weights/`.

Safety:
  - Refuses to run while a training job is writing to the dir (checks for
    a sibling `job_*.log` with recent mtime).
  - Dry-run by default; pass --execute to actually delete.
  - Never touches `weights/` — eval artifacts are safe.

Usage:
    # preview
    uv run python scripts/prune_ckpt_training_state.py --run-dir outputs/sft-baseline-marin
    # execute
    uv run python scripts/prune_ckpt_training_state.py --run-dir outputs/sft-baseline-marin --execute
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

_STEP_RE = re.compile(r"^step_(\d+)$")
_RECENT_SECONDS = 900  # 15 min


def _recent_log(run_dir: Path) -> Path | None:
    now = time.time()
    for log in run_dir.glob("job_*.log"):
        if now - log.stat().st_mtime < _RECENT_SECONDS:
            return log
    return None


def run(run_dir: Path, execute: bool = False) -> dict[str, int | list[str]]:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"No checkpoints/ under {run_dir}")

    # Safety: refuse if a job log touched in last 15 min
    recent = _recent_log(run_dir)
    if recent and execute:
        raise RuntimeError(
            f"Recent activity in {recent} (mtime < {_RECENT_SECONDS}s) — "
            "refusing to prune while training may still be writing. "
            "Wait for the job to finish or pass a different --run-dir."
        )

    steps: list[tuple[int, Path]] = []
    for child in ckpt_root.iterdir():
        if not child.is_dir():
            continue
        m = _STEP_RE.match(child.name)
        if m is None:
            continue
        steps.append((int(m.group(1)), child))
    if not steps:
        print(f"[prune] no step_N dirs under {ckpt_root}")
        return {"kept": [], "pruned": [], "bytes_freed": 0}
    steps.sort()
    final_step, final_dir = steps[-1]

    to_prune = [p for (_, p) in steps[:-1]]
    bytes_freed = 0
    for p in to_prune:
        for f in p.rglob("*"):
            if f.is_file():
                bytes_freed += f.stat().st_size

    print(f"[prune] run-dir: {run_dir}")
    print(f"[prune] {len(steps)} ckpts, keeping final step_{final_step} ({final_dir})")
    print(f"[prune] would prune {len(to_prune)} training-state dirs, freeing {bytes_freed / 1e9:.1f} GB")
    for p in to_prune:
        print(f"        - {p}")
    if not execute:
        print("[prune] DRY RUN. Re-run with --execute to actually delete.")
        return {"kept": [str(final_dir)], "pruned": [str(p) for p in to_prune], "bytes_freed": bytes_freed}

    for p in to_prune:
        shutil.rmtree(p)
    print(f"[prune] deleted {len(to_prune)} dirs, freed {bytes_freed / 1e9:.1f} GB")
    return {"kept": [str(final_dir)], "pruned": [str(p) for p in to_prune], "bytes_freed": bytes_freed}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run output dir (containing `checkpoints/` and `weights/`).",
    )
    p.add_argument("--execute", action="store_true", help="Actually delete. Default is dry-run.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run(args.run_dir.resolve(), execute=args.execute)
    return 0


if __name__ == "__main__":
    sys.exit(main())

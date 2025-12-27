import json
from pathlib import Path
from typing import Any

from prime_rl.utils.pathing import get_eval_dir


def _online_eval_state_path(output_dir: Path) -> Path:
    # Keep this inside evals/ so it naturally follows output_dir persistence.
    return get_eval_dir(output_dir) / "online_eval_state.json"


def read_last_completed_online_eval_ckpt_step(output_dir: Path) -> int | None:
    """Return last completed online eval checkpoint step, if recorded."""
    path = _online_eval_state_path(output_dir)
    if not path.exists():
        return None
    try:
        data: Any
        with open(path, "r") as f:
            data = json.load(f)
        value = data.get("last_completed_ckpt_step")
        if value is None:
            return None
        return int(value)
    except Exception:
        # Best-effort; if corrupted, just ignore and proceed.
        return None


def infer_last_completed_online_eval_ckpt_step_from_disk(output_dir: Path) -> int | None:
    """Fallback: infer last eval step by scanning output_dir/evals/step_* folders."""
    eval_dir = get_eval_dir(output_dir)
    if not eval_dir.exists():
        return None
    step_dirs = list(eval_dir.glob("step_*"))
    if not step_dirs:
        return None
    steps: list[int] = []
    for step_dir in step_dirs:
        try:
            steps.append(int(step_dir.name.split("_")[-1]))
        except Exception:
            continue
    return max(steps) if steps else None


def write_last_completed_online_eval_ckpt_step(output_dir: Path, ckpt_step: int) -> None:
    """Persist last completed online eval checkpoint step (atomic write)."""
    eval_dir = get_eval_dir(output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    path = _online_eval_state_path(output_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {"last_completed_ckpt_step": int(ckpt_step)}
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(path)


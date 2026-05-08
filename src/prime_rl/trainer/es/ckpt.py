import shutil
from pathlib import Path

import torch

from prime_rl.trainer.es.lora_materialize import TensorSpec
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path


def save_es_state(
    output_dir: Path,
    step: int,
    theta: torch.Tensor,
    specs: list[TensorSpec],
    metrics: dict,
) -> Path:
    step_dir = get_step_path(get_ckpt_dir(output_dir), step) / "es"
    step_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "theta": theta.detach().cpu(),
            "specs": specs,
            "metrics": metrics,
        },
        step_dir / "es_state.pt",
    )
    (step_dir.parent / "STABLE").touch()
    return step_dir


def load_es_state(output_dir: Path, step: int) -> dict:
    state_path = get_step_path(get_ckpt_dir(output_dir), step) / "es" / "es_state.pt"
    return torch.load(state_path, map_location="cpu", weights_only=False)


def latest_es_step(output_dir: Path) -> int | None:
    ckpt_dir = get_ckpt_dir(output_dir)
    if not ckpt_dir.exists():
        return None
    steps: list[int] = []
    for path in ckpt_dir.glob("step_*"):
        stable = path / "STABLE"
        state = path / "es" / "es_state.pt"
        if stable.exists() and state.exists():
            try:
                steps.append(int(path.name.split("_")[-1]))
            except ValueError:
                continue
    return max(steps) if steps else None


def maybe_clean_es_checkpoints(output_dir: Path, current_step: int, keep_last: int | None) -> None:
    if keep_last is None:
        return
    ckpt_dir = get_ckpt_dir(output_dir)
    if not ckpt_dir.exists():
        return
    keep_from = max(current_step - keep_last + 1, 0)
    for path in ckpt_dir.glob("step_*"):
        try:
            step = int(path.name.split("_")[-1])
        except ValueError:
            continue
        if step < keep_from:
            shutil.rmtree(path, ignore_errors=True)

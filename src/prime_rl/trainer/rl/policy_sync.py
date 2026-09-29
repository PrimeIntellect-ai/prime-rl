"""Model-only policy backbone snapshots for a LoRA value trainer."""

import shutil
from pathlib import Path

import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from prime_rl.trainer.world import get_world


def _backbone_state(model):
    state = {
        name: tensor
        for name, tensor in get_model_state_dict(model).items()
        if name.startswith("model.") and "lora_" not in name
    }
    if not state:
        raise ValueError("Policy backbone snapshot contains no model weights")
    return state


def snapshot_path(directory: Path, step: int) -> Path:
    return directory / f"step_{step}"


def save_policy_backbone(model, directory: Path, step: int) -> None:
    world = get_world()
    path = snapshot_path(directory, step)
    if world.is_master:
        directory.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.rmtree(path)
    dist.barrier()
    dcp_save({"model": _backbone_state(model)}, checkpoint_id=path)
    if world.is_master:
        (path / ".ready").touch()
        for old_path in directory.glob("step_*"):
            if old_path != path and (old_path / ".applied").is_file():
                shutil.rmtree(old_path)
    dist.barrier()


def latest_ready_step(directory: Path, after_step: int, at_most_step: int) -> int | None:
    if not directory.exists():
        return None
    steps = [
        int(path.name.removeprefix("step_"))
        for path in directory.glob("step_*")
        if path.name.removeprefix("step_").isdigit() and (path / ".ready").is_file()
    ]
    return max((step for step in steps if after_step < step <= at_most_step), default=None)


def load_policy_backbone(model, directory: Path, step: int) -> None:
    path = snapshot_path(directory, step)
    state = _backbone_state(model)
    dcp_load({"model": state}, checkpoint_id=path)
    set_model_state_dict(model, state, options=StateDictOptions(strict=False))
    dist.barrier()

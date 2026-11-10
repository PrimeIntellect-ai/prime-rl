"""Keep the weights with the lowest validation loss of an SFT run (``[val] keep_best``).

The snapshot holds only the model (parameters and buffers, no optimizer state, no dataloader): it exists to be
exported or evaluated, while the regular ``[ckpt]`` checkpoints are for resuming. It is taken right after the
validation that measured it, before anything else touches the weights, so it is exactly the weights that produced
the recorded loss.

Every rank saves its own local shards (``rank_<r>-of-<world>.pt``) and only ever reads back its own file. Nothing
depends on files another node wrote, so this works when the volume is staged per node and synced after the run.
The cost is that a snapshot only reloads into the same world size and sharding (the same run, or a resume of it
with the same topology).

Layout under ``root`` (``<checkpoint dir>/best-weights``):
    best.json              the best step, its validation loss, every candidate's loss (written by global rank 0)
    step_<N>/rank_<r>-of-<world>.pt
A new best is written and agreed on by all ranks before ``best.json`` points to it; only then is the previous
snapshot removed. A failed save keeps the previous best and does not stop training.
"""

import json
import math
import os
import shutil
import time
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger

BEST_DIR_NAME = "best-weights"
BEST_JSON = "best.json"
FORMAT = "prime-rl-sft-best-weights-v1"


def is_improvement(val_loss: float, best_loss: float | None) -> bool:
    """A finite loss strictly lower than the best so far (the first finite loss always is)."""
    return math.isfinite(val_loss) and (best_loss is None or val_loss < best_loss)


def persistent_buffers(model: nn.Module) -> list[tuple[str, Tensor]]:
    """The buffers a state dict holds (non-persistent ones, e.g. rotary caches, are derived and skipped)."""
    return [
        (f"{prefix}.{name}" if prefix else name, buffer)
        for prefix, module in model.named_modules()
        for name, buffer in module._buffers.items()
        if buffer is not None and name not in module._non_persistent_buffers_set
    ]


def local_tensors(model: nn.Module) -> dict[str, Tensor]:
    """This rank's part of every parameter and persistent buffer, by module path.

    Read straight from the modules (no state-dict hooks, so packed runtime fusions stay packed), and aliasing the
    live storage: copying into an entry writes the model.
    """
    out: dict[str, Tensor] = {}
    for name, tensor in chain(model.named_parameters(), persistent_buffers(model)):
        if tensor is None:
            continue
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        out[name] = tensor.detach()
    return out


def _all_true(flag: bool, group) -> bool:
    if not dist.is_initialized():
        return flag
    flags: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(flags, bool(flag), group=group)
    return all(flags)


def _from_master(obj: Any, group) -> Any:
    if not dist.is_initialized():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0, group=group)
    return box[0]


def _barrier(group) -> None:
    if dist.is_initialized():
        dist.barrier(group=group)


class BestCheckpoint:
    """Tracks the best validation loss and keeps its weights on disk. Every method is called by all ranks."""

    def __init__(self, root: Path, group=None):
        self.root = root
        self.world = get_world()
        self.logger = get_logger()
        # Object collectives (broadcast_object_list, all_gather_object) need a CPU-capable group.
        if group is None and dist.is_initialized():
            group = dist.new_group(backend="gloo")
        self.group = group
        self.best: dict[str, Any] | None = None
        self.history: dict[int, float] = {}

    # Paths

    def step_dir(self, step: int) -> Path:
        return self.root / f"step_{step}"

    def rank_file(self, step: int) -> Path:
        return self.step_dir(step) / f"rank_{self.world.rank:05d}-of-{self.world.world_size:05d}.pt"

    @property
    def best_step(self) -> int | None:
        return None if self.best is None else self.best["step"]

    @property
    def best_loss(self) -> float | None:
        return None if self.best is None else self.best["val_loss"]

    # Public API

    def consider(self, step: int, val_loss: float, model: nn.Module, *, val_logged_step: int, **info: Any) -> bool:
        """Record the validation loss of the current weights, which are those after ``step`` optimizer updates,
        and snapshot them if it is the lowest so far. Returns whether a new best was saved."""
        # All ranks must take the same branch: decide on the master's value.
        val_loss = float(_from_master(float(val_loss), self.group))
        self.history[step] = val_loss
        if not is_improvement(val_loss, self.best_loss):
            self.logger.info(
                f"Best checkpoint: val loss {val_loss:.4f} after step {step} is not below the best "
                f"{self.best_loss:.4f} (step {self.best_step})"
                if self.best is not None
                else f"Best checkpoint: val loss {val_loss} after step {step} is not finite, not kept"
            )
            self._write_json()
            return False

        start = time.perf_counter()
        self._make_step_dir(step)
        path = self.rank_file(step)
        error = None
        try:
            tmp = path.with_name(path.name + ".tmp")
            torch.save(local_tensors(model), tmp)
            os.replace(tmp, path)
        except Exception as exc:  # disk full, I/O error: keep training with the previous best
            error = repr(exc)
            self.logger.error(f"Best checkpoint: rank {self.world.rank} failed to save {path}: {error}")
        if not _all_true(error is None, self.group):
            self.logger.error(f"Best checkpoint: snapshot of step {step} failed, keeping step {self.best_step}")
            for leftover in (path, path.with_name(path.name + ".tmp")):
                leftover.unlink(missing_ok=True)
            if step != self.best_step:
                self._remove_snapshot(step)
            return False

        previous = self.best
        self.best = {
            "step": step,
            "val_loss": val_loss,
            "val_logged_step": val_logged_step,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **info,
        }
        self._write_json()
        _barrier(self.group)
        if previous is not None and previous["step"] != step:
            self._remove_snapshot(previous["step"])
        self.logger.success(
            f"Best checkpoint: step {step} (val loss {val_loss:.4f}) saved to {self.step_dir(step)} "
            f"in {time.perf_counter() - start:.0f}s"
            + (f", replacing step {previous['step']} ({previous['val_loss']:.4f})" if previous else "")
        )
        return True

    def restore(self, model: nn.Module) -> bool:
        """Load the best snapshot into ``model`` in place. Returns False on every rank if any rank failed."""
        if self.best is None:
            return False
        path = self.rank_file(self.best["step"])
        error = None
        try:
            saved = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            current = local_tensors(model)
            if set(saved) != set(current):
                missing, extra = sorted(set(current) - set(saved)), sorted(set(saved) - set(current))
                raise ValueError(f"tensor names differ: missing {missing[:3]}, extra {extra[:3]}")
            for name, target in current.items():
                source = saved[name]
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise ValueError(
                        f"{name}: saved {source.dtype} {tuple(source.shape)} != model {target.dtype} {tuple(target.shape)}"
                    )
            with torch.no_grad():
                for name, target in current.items():
                    target.copy_(saved[name])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            error = repr(exc)
            self.logger.error(f"Best checkpoint: rank {self.world.rank} failed to load {path}: {error}")
        ok = _all_true(error is None, self.group)
        if ok:
            self.logger.info(
                f"Best checkpoint: loaded step {self.best['step']} from {self.step_dir(self.best['step'])}"
            )
        return ok

    def load_state(self) -> None:
        """On resume: pick up the best so far, if its snapshot is complete for this world size."""
        payload = None
        if self.world.is_master:
            try:
                payload = json.loads((self.root / BEST_JSON).read_text())
            except FileNotFoundError:
                payload = None
            except Exception as exc:
                self.logger.warning(f"Best checkpoint: could not read {self.root / BEST_JSON}: {exc!r}")
        payload = _from_master(payload, self.group)
        if payload is None:
            return
        history = {int(step): float(loss) for step, loss in payload.get("history", {}).items()}
        best = payload.get("best")
        usable = (
            best is not None
            and payload.get("world_size") == self.world.world_size
            and self.rank_file(best["step"]).is_file()
        )
        if _all_true(usable, self.group):
            self.best, self.history = best, history
            self.logger.info(f"Best checkpoint: resuming with step {best['step']} (val loss {best['val_loss']:.4f})")
        else:
            self.history = history
            self.logger.warning(
                f"Best checkpoint: {self.root / BEST_JSON} lists step {best and best.get('step')}, but its snapshot "
                f"is incomplete or from another world size; starting the best-checkpoint search over"
            )

    # Internals

    def _make_step_dir(self, step: int) -> None:
        """Master first, then one rank per node if it does not see it yet (the volume may be staged per node)."""
        path = self.step_dir(step)
        if self.world.is_master:
            path.mkdir(parents=True, exist_ok=True)
        _barrier(self.group)
        if self.world.local_rank == 0 and not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        _barrier(self.group)

    def _remove_snapshot(self, step: int) -> None:
        """Every rank removes its own file, then one rank per node the directory (its node's copy of it)."""
        self.rank_file(step).unlink(missing_ok=True)
        _barrier(self.group)
        if self.world.local_rank == 0:
            shutil.rmtree(self.step_dir(step), ignore_errors=True)
        _barrier(self.group)

    def _write_json(self) -> None:
        if not self.world.is_master:
            return
        payload = {
            "format": FORMAT,
            "best": self.best,
            "world_size": self.world.world_size,
            "history": {str(step): loss for step, loss in sorted(self.history.items())},
            "note": "step = number of optimizer updates applied to the weights that were validated",
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / (BEST_JSON + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(tmp, self.root / BEST_JSON)
        except Exception as exc:
            self.logger.error(f"Best checkpoint: could not write {self.root / BEST_JSON}: {exc!r}")

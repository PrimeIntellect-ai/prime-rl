import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_ckpt_dir, get_step_path


@dataclass
class OrchState:
    """Minimal orchestrator state needed to resume cleanly. Engine version +
    watcher cursor are recoverable from disk on the next watcher tick, so we
    don't persist them. Round-robin scheduler index resets to 0 on resume —
    minor reordering, no correctness loss."""

    step: int = 0
    last_eval_step: int = 0


class CkptManager:
    """Persist and restore the orch state under
    `<output_dir>/checkpoints/step_N/orchestrator/state.pt` so trainer + orch
    ckpts at the same step share a parent dir."""

    def __init__(self, output_dir: Path, config: CheckpointConfig):
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.logger = get_logger()

    def _path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    def save(self, state: OrchState, step: int) -> None:
        path = self._path(step)
        path.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with open(path / "state.pt", "wb") as f:
            torch.save({"state": asdict(state)}, f)
        self.logger.info(f"Saved orch ckpt step {step} in {time.perf_counter() - t0:.2f}s")
        self._prune()

    def load(self, step: int) -> OrchState:
        path = self._path(step)
        f = path / "state.pt"
        if not f.exists():
            raise FileNotFoundError(f"orch checkpoint missing at {f}")
        with open(f, "rb") as fp:
            data = torch.load(fp, weights_only=False)
        return OrchState(**data["state"])

    def latest_step(self) -> int | None:
        if not self.ckpt_dir.exists():
            return None
        steps = [s for s in get_all_ckpt_steps(self.ckpt_dir) if (self._path(s) / "state.pt").exists()]
        return steps[-1] if steps else None

    def _prune(self) -> None:
        # `keep_last` + `keep_interval` semantics match orch1: keep the last N
        # ckpts always, plus any that are at a `keep_interval` multiple.
        if self.config.keep_last is None and self.config.keep_interval is None:
            return
        steps = [s for s in get_all_ckpt_steps(self.ckpt_dir) if (self._path(s) / "state.pt").exists()]
        keep: set[int] = set()
        if self.config.keep_last is not None:
            keep.update(steps[-self.config.keep_last :])
        if self.config.keep_interval is not None:
            keep.update(s for s in steps if s % self.config.keep_interval == 0)
        for s in steps:
            if s in keep:
                continue
            p = self._path(s)
            if p.exists():
                shutil.rmtree(p)
                self.logger.debug(f"Pruned orch ckpt step {s}")


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CkptManager | None:
    return CkptManager(output_dir, config) if config else None

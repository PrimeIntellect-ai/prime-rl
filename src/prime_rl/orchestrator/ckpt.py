"""Checkpoint manager for ``Progress``. Layout:
``<output_dir>/checkpoints/step_N/orchestrator/progress.pt``."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.multi_agent_advantage import RAEState
from prime_rl.orchestrator.types import Progress
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path


class CheckpointManager:
    def __init__(self, output_dir: Path, config: CheckpointConfig) -> None:
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    def save(self, progress: Progress, step: int, *, rae_state: RAEState | None = None) -> None:
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with open(ckpt_path / "progress.pt", "wb") as f:
            torch.save({"progress": progress}, f)
        if rae_state is not None:
            with open(ckpt_path / "rae_state.pt", "wb") as f:
                torch.save(
                    {
                        "baselines": rae_state.baselines,
                        "canonical_members": rae_state.canonical_members,
                        "beta": rae_state.beta,
                        "n_eff": rae_state.n_eff,
                    },
                    f,
                )
        get_logger().debug(
            f"Orchestrator checkpoint saved to {ckpt_path} in {format_time(time.perf_counter() - start)}"
        )

    def load(self, progress: Progress, step: int, *, rae_state: RAEState | None = None) -> None:
        ckpt_path = self.get_ckpt_path(step)
        state_file = ckpt_path / "progress.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"Orchestrator checkpoint not found at {state_file}")
        get_logger().debug(f"Loading checkpoint from {state_file}")
        start = time.perf_counter()
        if self.config.skip_progress:
            get_logger().info("Skipping progress loading from checkpoint")
        else:
            with open(state_file, "rb") as f:
                state = torch.load(f, weights_only=False)
            saved: Progress = state["progress"]
            for key, value in asdict(saved).items():
                if hasattr(progress, key):
                    setattr(progress, key, value)
        if rae_state is not None:
            rae_file = ckpt_path / "rae_state.pt"
            if not rae_file.exists():
                raise FileNotFoundError(
                    f"RAE state not found at {rae_file} but rae advantage is active. "
                    "Resume from a checkpoint with rae_state.pt, or start fresh."
                )
            with open(rae_file, "rb") as f:
                state = torch.load(f, weights_only=False)
            missing = {"baselines", "canonical_members", "beta", "n_eff"} - state.keys()
            if missing:
                raise ValueError(
                    f"RAE state at {rae_file} is missing key(s) {sorted(missing)} — likely the "
                    "retired sequential-EMA format (per-member triple keys + momentum). Rank-7 RAE "
                    "stores (env_name, example_id) baselines plus canonical_members/beta/n_eff; "
                    "old checkpoints are not migrated — start fresh."
                )
            bad_keys = [key for key in state["baselines"] if not (isinstance(key, tuple) and len(key) == 2)]
            if bad_keys:
                raise ValueError(
                    f"RAE state at {rae_file} has non-(env_name, example_id) baseline "
                    f"key(s), e.g. {bad_keys[0]!r}. Rank-7 RAE requires 2-tuple keys — start fresh."
                )
            bad_frames = [
                key
                for key, members in state["canonical_members"].items()
                if not (isinstance(members, tuple) and members and all(isinstance(m, str) for m in members))
            ]
            if bad_frames:
                raise ValueError(
                    f"RAE state at {rae_file} has canonical_members value(s) that are not non-empty member "
                    f"tuples, e.g. {state['canonical_members'][bad_frames[0]]!r} for key {bad_frames[0]!r} — "
                    "likely the retired single-canonical-id format. Rank-7 RAE persists each key's full "
                    "pair set — start fresh."
                )
            orphaned = sorted(set(state["baselines"]) - set(state["canonical_members"]), key=repr)
            if orphaned:
                raise ValueError(
                    f"RAE state at {rae_file} has warm baseline key(s) with no canonical_members entry: "
                    f"{orphaned}. A baseline without its persisted frame would silently re-derive a fresh "
                    "one on the next group — the checkpoint is inconsistent (hand-edited?); start fresh."
                )
            rae_state.baselines = state["baselines"]
            rae_state.canonical_members = state["canonical_members"]
            rae_state.beta = state["beta"]
            rae_state.n_eff = state["n_eff"]
        get_logger().debug(f"Orchestrator checkpoint loaded in {format_time(time.perf_counter() - start)}")


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager | None:
    if config is None:
        return None
    return CheckpointManager(output_dir, config)

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.configs.eval import EvalConfig
    from prime_rl.configs.rl import RLConfig
    from prime_rl.configs.sft import SFTConfig


def check_resume_config(config: EvalConfig | RLConfig | SFTConfig, saved_paths: Iterable[Path]) -> None:
    """Require matching resolved configs before reusing saved run state."""
    if config.resume is not None and config.resume.skip_checks:
        get_logger().warning("Skipping resume compatibility checks; saved state may use different experiment settings.")
        return

    expected = config.model_dump(mode="json")
    for saved_path in saved_paths:
        if not saved_path.is_file():
            raise ValueError(
                f"--resume: no saved config at {saved_path}. "
                "Set --resume.skip-checks to reuse state without a saved config."
            )
        # Re-validating snapshots would fill missing fields with current defaults.
        previous = orjson.loads(saved_path.read_bytes())
        # Generated SLURM scripts run with dry_run=False and mark their attempt when started.
        if previous.get("dry_run"):
            if not saved_path.with_suffix(".started").is_file():
                continue
            previous["dry_run"] = False
        for snapshot in (expected, previous):
            # RL propagates resume into both components; enabling it must not cause a mismatch.
            snapshot.pop("resume", None)
            for component in ("trainer", "orchestrator"):
                if component in snapshot:
                    snapshot[component].pop("resume", None)
        changed = sorted(
            key
            for key in expected.keys() | previous.keys()
            if key not in expected or key not in previous or expected[key] != previous[key]
        )
        if changed:
            raise ValueError(
                f"--resume: config differs from {saved_path} in [{', '.join(changed)}]. "
                "Use the saved config, start a fresh run, or set --resume.skip-checks if the changes are intentional."
            )

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

IMAGE_OFFLOAD_DIR_ENV = "VF_RENDERER_IMAGE_OFFLOAD_DIR"
RUN_DIR_ENV = "PRIME_RL_RUN_DIR"
RUN_ID_ENV = "RUN_ID"


def run_asset_env(output_dir: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Resolve the environment used by subprocesses that share run image assets.

    Hosted runs resolve ``RUN_ID`` to ``/data/outputs/run_${RUN_ID}`` inside
    renderers. Local launches without hosted env vars use the RL output dir as
    the explicit run dir.
    """

    env = dict(os.environ if base is None else base)
    if env.get(IMAGE_OFFLOAD_DIR_ENV) or env.get(RUN_DIR_ENV) or env.get(RUN_ID_ENV):
        return env
    env[RUN_DIR_ENV] = str(output_dir.resolve())
    return env

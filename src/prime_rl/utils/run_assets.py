from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from string import Template

from prime_rl.configs.shared import MultimodalConfig

IMAGE_OFFLOAD_DIR_ENV = "VF_RENDERER_IMAGE_OFFLOAD_DIR"
RUN_ID_ENV = "RUN_ID"

RUN_OUTPUT_ROOT = Path("/data/outputs")
IMAGE_ASSET_SUBDIR = Path("assets/images")


def _expand_path(path: Path, env: Mapping[str, str]) -> Path:
    expanded = Template(os.path.expanduser(str(path))).safe_substitute(env)
    return Path(expanded).resolve()


def _run_id_dir(env: Mapping[str, str]) -> Path | None:
    raw_run_id = env.get(RUN_ID_ENV, "").strip()
    if not raw_run_id:
        return None
    run_id = raw_run_id.removeprefix("run_")
    return RUN_OUTPUT_ROOT / f"run_{run_id}"


def resolve_image_offload_dir(
    output_dir: Path,
    multimodal: MultimodalConfig,
    env: Mapping[str, str],
) -> Path:
    explicit = multimodal.offload_dir
    if explicit is not None:
        return _expand_path(explicit, env)
    hosted_run_dir = _run_id_dir(env)
    if hosted_run_dir is not None:
        return (hosted_run_dir / IMAGE_ASSET_SUBDIR).resolve()
    return (output_dir.resolve() / IMAGE_ASSET_SUBDIR).resolve()


def run_asset_env(
    output_dir: Path,
    multimodal: MultimodalConfig | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve the environment used by subprocesses that share run image assets.

    Prime-RL config owns the multimodal image offload path. Env vars are only the
    transport used by verifiers/renderers running in subprocesses.
    """

    env = dict(os.environ if base is None else base)
    config = multimodal or MultimodalConfig()

    env[IMAGE_OFFLOAD_DIR_ENV] = str(resolve_image_offload_dir(output_dir, config, env))

    return env


def configure_run_asset_env(output_dir: Path, multimodal: MultimodalConfig) -> None:
    if os.environ.get(IMAGE_OFFLOAD_DIR_ENV):
        return
    os.environ.update(run_asset_env(output_dir, multimodal=multimodal))

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from string import Template

from prime_rl.configs.shared import MultimodalConfig

# Contract: must match renderers.mm_store.IMAGE_OFFLOAD_DIR_ENV.
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
    """Resolve image asset dir by precedence: offload_dir, RUN_ID hosted path, then output_dir/assets/images."""
    explicit = multimodal.offload_dir
    if explicit is not None:
        return _expand_path(explicit, env)
    hosted_run_dir = _run_id_dir(env)
    if hosted_run_dir is not None:
        return (hosted_run_dir / IMAGE_ASSET_SUBDIR).resolve()
    return (output_dir.resolve() / IMAGE_ASSET_SUBDIR).resolve()

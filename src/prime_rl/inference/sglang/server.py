import json
import os
import shutil
import sys
from argparse import Namespace
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable


def _format_cli_value(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return str(value)


def _namespace_to_cli_args(namespace: Namespace) -> list[str]:
    args: list[str] = []
    for key, value in vars(namespace).items():
        if value is None or value is False:
            continue

        flag = f"--{key.replace('_', '-')}"
        if value is True:
            args.append(flag)
        else:
            args.extend([flag, _format_cli_value(value)])
    return args


def _sglang_project_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        project_dir = parent / "packages" / "prime-rl-sglang"
        if (project_dir / "pyproject.toml").is_file():
            return project_dir
    raise FileNotFoundError("Could not find packages/prime-rl-sglang for the isolated SGLang runtime")


def _root_nccl_library_path() -> Path | None:
    spec = find_spec("nvidia.nccl")
    if spec is None or spec.submodule_search_locations is None:
        return None

    for location in spec.submodule_search_locations:
        library_path = Path(location) / "lib" / "libnccl.so.2"
        if library_path.is_file():
            return library_path
    return None


def server(config: InferenceConfig) -> None:
    """Launch SGLang's OpenAI-compatible server."""
    logger = get_logger()
    namespace = config.to_sglang()
    sglang_args = ["-m", "prime_rl_sglang.launch_server", *_namespace_to_cli_args(namespace)]
    disable_nccl_p2p_if_unavailable()
    env = os.environ.copy()
    if nccl_library_path := _root_nccl_library_path():
        env.setdefault("SGLANG_NCCL_SO_PATH", str(nccl_library_path))
    if find_spec("prime_rl_sglang") is None and shutil.which("uv") is not None:
        project_dir = _sglang_project_dir()
        env.setdefault("FLASHINFER_WORKSPACE_BASE", str(project_dir))
        command = [
            "uv",
            "run",
            "--frozen",
            "--project",
            str(project_dir),
            "python",
            *sglang_args,
        ]
    else:
        if find_spec("prime_rl_sglang") is None:
            sglang_args[1] = "sglang.launch_server"
        command = [sys.executable, *sglang_args]
    logger.info(f"Starting SGLang server: {' '.join(command)}")
    os.execvpe(command[0], command, env)

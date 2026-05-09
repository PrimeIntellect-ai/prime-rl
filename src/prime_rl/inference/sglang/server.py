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


def server(config: InferenceConfig) -> None:
    """Launch SGLang's OpenAI-compatible server."""
    logger = get_logger()
    namespace = config.to_sglang()
    sglang_args = ["-m", "sglang.launch_server", *_namespace_to_cli_args(namespace)]
    env = os.environ.copy()
    if find_spec("sglang") is None and shutil.which("uv") is not None:
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
        command = [sys.executable, *sglang_args]
    logger.info(f"Starting SGLang server: {' '.join(command)}")
    os.execvpe(command[0], command, env)

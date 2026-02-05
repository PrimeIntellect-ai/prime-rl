import asyncio
import functools
import os
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import wandb

from prime_rl.orchestrator.config import EnvConfig, EvalEnvConfig
from prime_rl.utils.logger import get_logger

# TODO: Change all imports to use utils.pathing
# ruff: noqa: F401
from prime_rl.utils.pathing import (
    get_all_ckpt_steps,
    get_broadcast_dir,
    get_ckpt_dir,
    get_eval_dir,
    get_log_dir,
    get_rollout_dir,
    get_stable_ckpt_steps,
    get_step_path,
    get_weights_dir,
    resolve_latest_ckpt_step,
    sync_wait_for_path,
    wait_for_path,
)


def rgetattr(obj: Any, attr_path: str) -> Any:
    """
    Try to get a (nested) attribute from an object. For example:

    ```python
    class Foo:
        bar = "baz"

    class Bar:
        foo = Foo()

    foo = Foo()
    bar = Bar()
    ```

    Here, the following holds:
    - `getattr(foo, "bar")` will return `"baz"`.
    - `getattr(bar, "foo)` will return an object of type `Foo`.
    - `getattr(bar, "foo.bar")` will error

    This function solves this. `rgetattr(bar, "foo.bar")` will return `"baz"`.

    Args:
        obj: The object to get the attribute from.
        attr_path: The path to the attribute, nested using `.` as separator.

    Returns:
        The attribute
    """
    attrs = attr_path.split(".")
    current = obj

    for attr in attrs:
        if not hasattr(current, attr):
            raise AttributeError(f"'{type(current).__name__}' object has no attribute '{attr}'")
        current = getattr(current, attr)

    return current


def rsetattr(obj: Any, attr_path: str, value: Any) -> None:
    """
    Try to set a (nested) attribute from an object. For example:

    ```python
    class Foo:
        bar = "baz"

    class Bar:
        foo = Foo()

    foo = Foo()
    bar = Bar()
    ```

    Here, the following holds:
    - `rsetattr(bar, "foo.bar", "qux")` will set `bar.foo.bar` to `"qux"`.
    - `rsetattr(bar, "foo.bar", "qux")` will set `bar.foo.bar` to `"qux"`.

    Args:
        obj: The object to set the attribute on.
        attr_path: The path to the attribute, nested using `.` as separator.
        value: The value to set the attribute to.
    """
    if "." not in attr_path:
        return setattr(obj, attr_path, value)
    attr_path, attr = attr_path.rsplit(".", 1)
    obj = rgetattr(obj, attr_path)
    setattr(obj, attr, value)


def capitalize(s: str) -> str:
    """Capitalize the first letter of a string."""
    return s[0].upper() + s[1:]


def clean_exit(func: Callable) -> Callable:
    """
    A decorator that ensures the a torch.distributed process group is properly
    cleaned up after the decorated function runs or raises an exception.
    """
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                ret = await func(*args, **kwargs)
                wandb.finish()
                return ret
            except Exception as e:
                wandb.finish(exit_code=1)
                raise e
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                ret = func(*args, **kwargs)
                wandb.finish()
                return ret
            except Exception as e:
                wandb.finish(exit_code=1)
                raise e
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()

        return sync_wrapper


def to_col_format(list_of_dicts: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Turns a list of dicts to a dict of lists.

    Example:

    ```python
    list_of_dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}] # Row format
    to_col_format(list_of_dicts)
    ```

    Returns:

    ```python
    {"a": [1, 3], "b": [2, 4]} # Column format
    ```
    """
    dict_of_lists = defaultdict(list)
    for row in list_of_dicts:
        for key, value in row.items():
            dict_of_lists[key].append(value)
    return dict(dict_of_lists)


def to_row_format(dict_of_lists: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Turns a dict of lists to a list of dicts.

    Example:

    ```python
    dict_of_lists = {"a": [1, 3], "b": [2, 4]} # Column format
    to_row_format(dict_of_lists)
    ```

    Returns:

    ```python
    [{"a": 1, "b": 2}, {"a": 3, "b": 4}] # Row format
    ```
    """
    return [dict(zip(dict_of_lists.keys(), values)) for values in zip(*dict_of_lists.values())]


def format_time(time_in_seconds: float) -> str:
    """Format a time in seconds to a human-readable format."""
    from datetime import timedelta

    td = timedelta(seconds=time_in_seconds)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format based on magnitude
    if days > 0:
        total_hours = days * 24 + hours
        return f"{total_hours + minutes / 60:.2f}h"
    elif hours > 0:
        return f"{hours + minutes / 60:.2f}h"
    elif minutes > 0:
        return f"{minutes + seconds / 60:.2f}m"
    else:
        # Include microseconds for sub-second precision
        total_seconds = seconds + td.microseconds / 1_000_000
        return f"{total_seconds:.2f}s"


def format_num(num: float | int, precision: int = 2) -> str:
    """
    Format a number in human-readable format with abbreviations.
    """
    sign = "-" if num < 0 else ""
    num = abs(num)
    if num < 1e3:
        return f"{sign}{num:.{precision}f}" if isinstance(num, float) else f"{sign}{num}"
    elif num < 1e6:
        return f"{sign}{num / 1e3:.{precision}f}K"
    elif num < 1e9:
        return f"{sign}{num / 1e6:.{precision}f}M"
    else:
        return f"{sign}{num / 1e9:.{precision}f}B"


def get_free_port() -> int:
    """Find and return a free port"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to any available port
        s.listen(1)
        port = s.getsockname()[1]
    return port


def get_cuda_visible_devices() -> list[int]:
    """Returns the list of availble CUDA devices, taking into account the CUDA_VISIBLE_DEVICES environment variable."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is None:
        # Default to all devices if the environment variable is not set
        return list(range(torch.cuda.device_count()))
    return list(sorted([int(device) for device in cuda_visible.split(",")]))


def get_latest_ckpt_step(weights_dir: Path) -> int | None:
    step_dirs = list(weights_dir.glob("step_*"))
    if len(step_dirs) == 0:
        return None
    steps = sorted([int(step_dir.name.split("_")[-1]) for step_dir in step_dirs])
    for latest_step in steps[::-1]:
        if Path(weights_dir / f"step_{latest_step}" / "STABLE").exists():
            return latest_step
    return None


def mean(values: list[float] | list[int]) -> float:
    """Compute the mean of a list of values."""
    return sum(values) / len(values) if values else 0.0


def mean_normalize(values: list[float] | list[int]) -> list[float]:
    """Mean-Normalize a list of values to 0-1."""
    sum_values = sum(values)
    return [value / sum_values if sum_values > 0 else 0.0 for value in values]


@contextmanager
def default_dtype(dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def install_env(env_id: str) -> None:
    """Install an environment in subprocess."""
    logger = get_logger()
    logger.info(f"Installing environment {env_id}")
    install_cmd = ["uv", "run", "--no-sync", "prime", "env", "install", env_id]
    result = subprocess.run(install_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install environment {env_id} (stdout={result.stdout}, stderr={result.stderr})")
    logger.info(f"Successfully installed environment {env_id}")


def get_env_ids_to_install(env_configs: list[EnvConfig] | list[EvalEnvConfig]) -> set[str]:
    """Get the list of environment IDs to install."""
    env_ids_to_install = set()
    for env_config in env_configs:
        if "/" in env_config.id:
            env_ids_to_install.add(env_config.id)
    return env_ids_to_install


def is_chinese_char(char: str) -> bool:
    """Check if a character is Chinese (CJK Unified Ideographs and extensions)."""
    code_point = ord(char)
    return (
        0x4E00 <= code_point <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code_point <= 0x4DBF  # CJK Extension A
        or 0x20000 <= code_point <= 0x2A6DF  # CJK Extension B
        or 0x2A700 <= code_point <= 0x2B73F  # CJK Extension C
        or 0x2B740 <= code_point <= 0x2B81F  # CJK Extension D
        or 0x2B820 <= code_point <= 0x2CEAF  # CJK Extension E
        or 0xF900 <= code_point <= 0xFAFF  # CJK Compatibility Ideographs
        or 0x2F800 <= code_point <= 0x2FA1F  # CJK Compatibility Ideographs Supplement
    )


def count_chinese_chars(text: str) -> tuple[int, int]:
    """Count Chinese characters in text.

    Returns:
        Tuple of (chinese_char_count, total_char_count)
    """
    chinese_count = sum(1 for char in text if is_chinese_char(char))
    return chinese_count, len(text)

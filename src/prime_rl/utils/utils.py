import functools
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import wandb

from prime_rl.utils.logger import get_logger


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


def clean_exit(func):
    """
    A decorator that ensures the a torch.distributed process group is properly
    cleaned up after the decorated function runs or raises an exception.

    Args:
        func: The function to decorate

    Returns:
        The decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
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

    return wrapper


def wait_for_path(path: Path, interval: int = 1, log_interval: int = 10) -> None:
    logger = get_logger()
    wait_time = 0
    logger.debug(f"Waiting for path `{path}`")
    while True:
        if path.exists():
            logger.debug(f"Found path `{path}`")
            break
        if wait_time % log_interval == 0 and wait_time > 0:  # Every log_interval seconds
            logger.debug(f"Waiting for path `{path}` for {wait_time} seconds")
        time.sleep(interval)
        wait_time += interval


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


def get_step_path(path: Path, step: int) -> Path:
    return path / f"step_{step}"


def get_weight_ckpt_model_path(weight_dir: Path, step: int) -> Path:
    return weight_dir / f"step_{step}" / "pytorch_model.bin"

def get_free_port() -> int:
    """Find and return a free port"""
    import socket
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to any available port
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
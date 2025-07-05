import functools
import time
from pathlib import Path
from typing import Any

import torch.distributed as dist
import wandb

from zeroband.utils.logger import get_logger


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

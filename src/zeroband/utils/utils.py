import functools

import torch.distributed as dist


def ensure_process_group_cleanup(func):
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
            return func(*args, **kwargs)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    return wrapper

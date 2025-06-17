import functools

import torch.distributed as dist
import wandb


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

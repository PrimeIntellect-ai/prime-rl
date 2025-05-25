import os


def cuda_available_devices(gpus_ids: list[int]) -> str:
    return ",".join(map(str, gpus_ids))


class EnvWrapper:
    """
    This class wrap a function call and override the environment variables
    FYI: cannot use a simple function because of pickle issues
    """

    def __init__(self, fn, envs):
        self.fn = fn
        self.envs = envs

    def __call__(self, *args, **kwargs):
        os.environ.update(self.envs)
        return self.fn(*args, **kwargs)

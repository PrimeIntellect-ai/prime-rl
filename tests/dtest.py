import datetime
import inspect
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class DTest:
    """Base class for multi-GPU distributed pytest tests. Subclass and add `test_*` methods;
    each spawns `default_world_size` real GPU ranks with a real NCCL process group. Inside a
    test method, `self.rank` / `self.world_size` / `self.device` are available. Assumes CUDA
    and NCCL are always available; skips if fewer than `default_world_size` GPUs are visible.

    NOTE: `pytest.skip()` inside a test method is not supported — it's translated into a
    `RuntimeError` instead (see `_dist_run`), since `mp.spawn` only catches `Exception` while
    pytest's skip exception subclasses `BaseException` directly. Do availability checks before
    `run()` spawns anything instead. Mid-test skipping can be supported by passing a
    `multiprocessing.Queue` into each worker, switching from `mp.spawn(..., join=True)` to
    `torch.multiprocessing.start_processes(..., join=False)`, and polling that queue in the
    parent.
    """

    default_world_size: int = 2
    _timeout_sec: float = 30.0
    _is_worker: bool = False

    def __call__(self, request):
        test = getattr(self, request.function.__name__)
        test_kwargs = self._get_fixture_kwargs(request, test)
        self.run(test, test_kwargs, self.default_world_size)

    def _get_fixture_kwargs(self, request, func):
        # Reused near-verbatim from dtest (`_dtest.py:202-214`) — lets a DTest test method
        # also request ordinary pytest fixtures alongside `self.rank`/`self.world_size`.
        params = inspect.getfullargspec(func).args
        params.remove("self")
        kwargs = {}
        for p in params:
            try:
                kwargs[p] = request.getfixturevalue(p)
            except pytest.FixtureLookupError:
                pass
        return kwargs

    def run(self, test, test_kwargs: dict, world_size: int) -> None:
        if torch.cuda.device_count() < world_size:
            pytest.skip(
                f"{type(self).__name__}:{test.__name__} requires {world_size} GPUs, found {torch.cuda.device_count()}"
            )
        port = _free_port()
        try:
            mp.spawn(
                self._dist_run,
                args=(world_size, port, test.__name__, test_kwargs),
                nprocs=world_size,
                join=True,
            )
        except (mp.ProcessRaisedException, mp.ProcessExitedException) as e:
            pytest.fail(str(e), pytrace=False)

    def _dist_run(self, local_rank: int, world_size: int, port: int, test_name: str, test_kwargs: dict) -> None:
        self._is_worker = True
        self._rank = local_rank
        self._world_size = world_size
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=local_rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=self._timeout_sec),
        )
        test = getattr(self, test_name)
        try:
            test(**test_kwargs)
        except pytest.skip.Exception as e:
            raise RuntimeError(
                f"pytest.skip() is not supported inside a {type(self).__name__} worker "
                f"(skip reason was: {e.msg!r}). Do availability checks before the test runs "
                f"(e.g. by overriding `default_world_size` or checking torch.cuda.device_count())."
            ) from e
        except BaseException:
            # No destroy_process_group() here: other ranks may be blocked in a collective this
            # rank never reaches, and destroy() would wait for them, turning a real error into a
            # hang until the NCCL timeout instead of surfacing it.
            raise
        else:
            dist.destroy_process_group()

    @property
    def rank(self) -> int:
        if not self._is_worker:
            raise RuntimeError(f"{type(self).__name__}.rank is only available inside a spawned worker")
        return self._rank

    @property
    def world_size(self) -> int:
        if not self._is_worker:
            raise RuntimeError(f"{type(self).__name__}.world_size is only available inside a spawned worker")
        return self._world_size

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def print_rank(self, *args, **kwargs) -> None:
        print(f"[rank {self.rank}]", *args, **kwargs)

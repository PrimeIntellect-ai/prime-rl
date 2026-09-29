from itertools import chain

import torch
from dion import Muon as DionMuon
from dion.opt_utils import AsyncRuntime, create_param_batches

from prime_rl.trainer.optim.state_offload import StreamingMuonCPUOffloadOptimizer


class Muon(DionMuon):
    """Dion Muon with configurable optimizer-task concurrency."""

    def __init__(self, *args, max_concurrent_tasks: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be at least 1")
        self.max_concurrent_tasks = max_concurrent_tasks

    @torch.no_grad()
    def step(self, closure=None, state_offloader: StreamingMuonCPUOffloadOptimizer | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_groups = []
        lion_groups = []
        adamw_groups = []
        for group in self.param_groups:
            group["step"] += 1
            match group["algorithm"]:
                case "muon":
                    muon_groups.append(group)
                case "lion":
                    lion_groups.append(group)
                case "adamw":
                    adamw_groups.append(group)
                case algorithm:
                    raise ValueError(f"Unknown algorithm: {algorithm}")

        if state_offloader is None:
            tasks = chain(
                self._create_muon_tasks(muon_groups),
                self._create_lion_tasks(lion_groups),
                self._create_adamw_tasks(adamw_groups),
            )
            AsyncRuntime(tasks, max_concurrent_tasks=self.max_concurrent_tasks).run()
        else:
            for group in muon_groups:
                _, world_size, _, _ = self._get_group_mesh_info(group)
                params = [param for param in group["params"] if state_offloader.has_gradient(param)]
                for batch in create_param_batches(params, world_size, self._matrix_partitions):
                    self._run_streamed_batch(group, batch, self._create_muon_tasks, state_offloader)

            for groups, create_tasks in (
                (lion_groups, self._create_lion_tasks),
                (adamw_groups, self._create_adamw_tasks),
            ):
                for group in groups:
                    for param in group["params"]:
                        if state_offloader.has_gradient(param):
                            self._run_streamed_batch(group, [param], create_tasks, state_offloader)
        return loss

    def _run_streamed_batch(self, group, params, create_tasks, state_offloader):
        state_offloader.load_gradients(params)
        state_offloader._move_states("cuda", params)
        batch_group = {**group, "params": params}
        AsyncRuntime(create_tasks([batch_group]), max_concurrent_tasks=1).run()
        state_offloader.release_gradients(params)
        state_offloader._move_states("cpu", params)

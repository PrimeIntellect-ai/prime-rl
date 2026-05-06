import asyncio
from dataclasses import dataclass

import verifiers as vf

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.group import Group, setup_group
from prime_rl.orchestrator.scheduler import Kind, Scheduler, Task
from prime_rl.utils.logger import get_logger


@dataclass
class GroupOutput:
    """One produced + scored group, queued from engine to batcher."""

    example: dict
    env_id: str
    kind: Kind
    rollouts: list[vf.RolloutOutput]
    policy_version: int
    eval_step: int | None = None  # trigger step for eval groups; None for train


@dataclass
class Inflight:
    """Lightweight tracker for the max_off_policy_level metric. Not used for
    cancellation — once a group is dispatched, it runs to completion or times
    out, and gets filtered post-hoc by per-sample max_off_policy."""

    version: int
    kind: Kind


@dataclass
class EngineInputs:
    """Pre-built inputs for the RolloutEngine. `setup_rollout_engine` produces
    this from config; tests construct it directly with stub scheduler + group."""

    scheduler: Scheduler
    out_q: asyncio.Queue[GroupOutput]
    group: Group
    max_off_policy: int
    concurrency: int
    lora_name: str | None = None


class RolloutEngine:
    def __init__(self, inputs: EngineInputs):
        self.scheduler = inputs.scheduler
        self.out_q = inputs.out_q
        self.group = inputs.group
        # When set, swap rollouts to the LoRA adapter name on the first
        # successful weight update (step 0 rollouts always use the base model
        # since no adapter is loaded yet).
        self.lora_name = inputs.lora_name
        self.max_off_policy = inputs.max_off_policy
        self.concurrency = inputs.concurrency
        self.policy_version = 0
        self._inflight: list[Inflight] = []
        self.logger = get_logger()

    async def run(self) -> None:
        sem = asyncio.Semaphore(self.concurrency)
        while True:
            await sem.acquire()
            dispatch = self.scheduler.next_task()
            if dispatch is None:
                sem.release()
                return
            asyncio.create_task(self._run_group(dispatch.task, dispatch.example, sem, dispatch.eval_step))

    async def _run_group(
        self,
        task: Task,
        example: dict,
        sem: asyncio.Semaphore,
        eval_step: int | None,
    ) -> None:
        try:
            version = self.policy_version  # snapshot; current version may advance during await
            inflight = Inflight(version=version, kind=task.kind)
            self._inflight.append(inflight)
            try:
                rollouts = await self.group.run(task, example)
            finally:
                self._inflight.remove(inflight)

            # correctness guard: group may have finished just as a new version arrived.
            # Only enforced for train — eval rollouts always ship, they're tagged with
            # the trigger step regardless of which weights produced each rollout.
            if task.kind == "train" and self.policy_version - version > self.max_off_policy:
                return
            # Train groups with no surviving rollouts (all timed out) are dropped;
            # eval groups still ship empty so the batcher's expected-count resolves.
            if task.kind == "train" and not rollouts:
                return

            await self.out_q.put(
                item=GroupOutput(
                    example=example,
                    env_id=task.id,
                    kind=task.kind,
                    rollouts=list(rollouts),
                    policy_version=version,
                    eval_step=eval_step,
                )
            )
        finally:
            sem.release()

    async def on_new_version(self, step: int) -> None:
        self.policy_version = step
        # First successful adapter load: switch rollout target from the base
        # model to the LoRA adapter name so future rollouts hit the trained
        # adapter on vLLM.
        if self.lora_name and self.group.model != self.lora_name:
            self.logger.info(f"Switching rollouts to LoRA adapter '{self.lora_name}' (was '{self.group.model}')")
            self.group.model = self.lora_name

    def max_off_policy_level(self) -> int:
        """Max lag across currently in-flight train groups."""
        train = [i for i in self._inflight if i.kind == "train"]
        if not train:
            return 0
        return max(self.policy_version - i.version for i in train)


def setup_rollout_engine(
    cfg: OrchestratorConfig,
    *,
    scheduler: Scheduler,
    out_q: asyncio.Queue[GroupOutput],
    client: vf.ClientConfig,
    concurrency: int,
    lora_name: str | None = None,
) -> RolloutEngine:
    """Translate config → RolloutEngine. `concurrency` is computed by the
    caller because it's also needed to size out_q. Tests should construct
    `RolloutEngine(EngineInputs(...))` directly."""
    return RolloutEngine(
        EngineInputs(
            scheduler=scheduler,
            out_q=out_q,
            group=setup_group(cfg, client=client),
            max_off_policy=cfg.max_off_policy_steps,
            concurrency=concurrency,
            lora_name=lora_name,
        )
    )

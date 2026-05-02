import asyncio
from dataclasses import dataclass

import verifiers as vf
from aiolimiter import AsyncLimiter

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.scheduler import Kind, Scheduler, Task
from prime_rl.utils.logger import get_logger


@dataclass
class Group:
    example: dict
    env_id: str
    kind: Kind
    rollouts: list[vf.RolloutOutput]
    policy_version: int
    eval_step: int | None = None  # trigger step for eval groups; None for train


@dataclass
class Inflight:
    version: int
    gather: asyncio.Future
    kind: Kind


@dataclass
class EngineInputs:
    """Pre-built inputs for the RolloutEngine. `setup_rollout_engine` produces
    this from config; tests construct it directly with stub scheduler + env."""

    scheduler: Scheduler
    out_q: asyncio.Queue[Group]
    client: vf.ClientConfig
    model: str
    max_off_policy: int
    concurrency: int
    tasks_per_minute: int | None = None
    max_rollout_time_seconds: float | None = None
    lora_name: str | None = None


class RolloutEngine:
    def __init__(self, inputs: EngineInputs):
        self.scheduler = inputs.scheduler
        self.out_q = inputs.out_q
        self.client = inputs.client
        self.model = inputs.model
        # When set, swap rollouts to the LoRA adapter name on the first
        # successful weight update (step 0 rollouts always use the base model
        # since no adapter is loaded yet).
        self.lora_name = inputs.lora_name
        self.max_off_policy = inputs.max_off_policy
        self.concurrency = inputs.concurrency
        self.rate_limiter = (
            AsyncLimiter(max_rate=inputs.tasks_per_minute, time_period=60) if inputs.tasks_per_minute else None
        )
        self.max_rollout_time_seconds = inputs.max_rollout_time_seconds
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
            gather = asyncio.gather(*(self._rollout(task, example) for _ in range(task.rollouts_per_group)))
            inflight = Inflight(version=version, gather=gather, kind=task.kind)
            self._inflight.append(inflight)

            timed_out = False
            try:
                if self.max_rollout_time_seconds is not None:
                    rollouts = await asyncio.wait_for(gather, timeout=self.max_rollout_time_seconds)
                else:
                    rollouts = await gather
            except asyncio.TimeoutError:
                timed_out = True
                self.logger.warning(
                    f"Rollout group timed out after {self.max_rollout_time_seconds}s "
                    f"(task={task.id}, kind={task.kind}, version={version})"
                )
            except asyncio.CancelledError:
                return
            finally:
                self._inflight.remove(inflight)

            if timed_out:
                # Train groups are dropped on timeout. Eval emits an empty-rollouts
                # group so the batcher's expected-count check can resolve and flush
                # the partial epoch; otherwise it would wait forever.
                if task.kind == "eval":
                    await self.out_q.put(
                        Group(
                            example=example,
                            env_id=task.id,
                            kind="eval",
                            rollouts=[],
                            policy_version=version,
                            eval_step=eval_step,
                        )
                    )
                return

            # correctness guard: group may have finished just as a new version arrived.
            # Only enforced for train — eval rollouts always ship, they're tagged with
            # the trigger step regardless of which weights produced each rollout.
            if task.kind == "train" and self.policy_version - version > self.max_off_policy:
                return

            await self.out_q.put(
                Group(
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

    async def _rollout(self, task: Task, example: dict) -> vf.RolloutOutput:
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()
        return await task.env.run_rollout(
            vf.RolloutInput(**example),
            client=self.client,
            model=self.model,
            sampling_args=task.sampling_args,
            state_columns=["trajectory", "sampling_args"],
        )

    async def on_new_version(self, step: int) -> None:
        self.policy_version = step
        # First successful adapter load: switch rollout target from the base
        # model to the LoRA adapter name so future rollouts hit the trained
        # adapter on vLLM.
        if self.lora_name and self.model != self.lora_name:
            self.logger.info(f"Switching rollouts to LoRA adapter '{self.lora_name}' (was '{self.model}')")
            self.model = self.lora_name
        for inflight in list(self._inflight):
            if inflight.kind == "eval":
                continue  # never cancel eval; it's tagged with its trigger step
            if step - inflight.version > self.max_off_policy:
                inflight.gather.cancel()

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
    out_q: asyncio.Queue[Group],
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
            client=client,
            model=cfg.model.name,
            max_off_policy=cfg.max_off_policy_steps,
            concurrency=concurrency,
            tasks_per_minute=cfg.tasks_per_minute,
            max_rollout_time_seconds=(cfg.max_rollout_time_minutes * 60.0) if cfg.max_rollout_time_minutes else None,
            lora_name=lora_name,
        )
    )

"""The replay taskset: turns saved rollouts back into training tasks.

Reads a run's saved rollout files (``<output>/rollouts/step_*/train_rollouts.jsonl``,
plain WireTrace lines) and materializes one of three derived task kinds:

- ``continue``: resume from a context-compaction point — the seed is the post-compaction
  prompt (system + summary user message) recovered from the trace's message graph; scored
  by the original taskset.
- ``recheck``: the finished conversation plus an appended check-your-work turn; scored by
  the original taskset.
- ``judge``: a rendered transcript and the question "was this correct?"; the reward is
  whether the verdict matches the reward the rollout actually received. Self-contained —
  no original taskset needed.

Tasks bind lazily: ``load_tasks`` returns stubs fixing the index range and
``resolve_task`` reads one saved line per request. Online buffers (``online = true``,
typically with ``buffer_dir = "self"`` under prime-rl) sample a fresh source rollout per
request and therefore force group dispatch, so all GRPO group members share one source.

Pair with a harness that supports message prompts and taskset tools — the built-in
``null`` chat-loop harness fits all three kinds.
"""

import re
from collections.abc import Mapping
from typing import ClassVar, Literal

from pydantic import SerializeAsAny, model_validator
from verifiers.v1 import Runtime, Task, Taskset, TasksetConfig, Toolset, Trace, metric, reward, task_type
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.loaders import narrow_plugin_field, taskset_class, taskset_config_type
from verifiers.v1.state import State, state_cls

from replay_v1.buffer import ReplayBuffer
from replay_v1.surgery import build_children, continue_seed, main_tree, recheck_seed, render_transcript

ReplayKind = Literal["continue", "recheck", "judge"]

RECHECK_INSTRUCTION = (
    "Carefully check your work above. Re-verify the reasoning and the final answer, and "
    "fix any mistakes you find. Then state your final answer."
)

JUDGE_INSTRUCTION = (
    "Below is the transcript of an AI assistant attempting a task.\n\n"
    "{transcript}\n\n"
    "Did the assistant solve the task correctly? Analyze the transcript, then answer on "
    "the final line with exactly `VERDICT: CORRECT` or `VERDICT: INCORRECT`."
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)", re.IGNORECASE)


def parse_verdict(reply: str) -> bool | None:
    """The model's verdict, from the last `VERDICT:` line; None when unparseable."""
    matches = _VERDICT_RE.findall(reply)
    if not matches:
        return None
    return matches[-1].upper() == "CORRECT"


class ReplayTasksetConfig(TasksetConfig):
    """Config for the replay taskset. `buffer_dir`, `mode`, and (for continue/recheck)
    `inner` are the load-bearing fields; the rest tune candidate selection."""

    buffer_dir: str = ""
    """The saved-rollout dir to replay: a run's `rollouts` dir (or the run dir containing
    it). Under prime-rl the literal `"self"` resolves to this run's own rollout dir (an
    online buffer over the run's freshly written rollouts)."""

    mode: ReplayKind = "judge"
    """Which derived task this env serves. One mode per env entry — mix modes (and set
    their ratios) with multiple `[[orchestrator.train.env]]` entries."""

    inner: SerializeAsAny[TasksetConfig] | None = None
    """The original taskset's config — continue/recheck rollouts are scored by it, so it
    must reproduce the source run's taskset config. Judge scoring is self-contained and
    forbids this field."""

    online: bool = False
    """Treat the buffer as growing (this run's own rollouts): steps are rescanned during
    training, only barrier-complete steps are read, and every request samples a fresh
    source rollout (which forces whole-group dispatch so GRPO groups share one source).
    Offline buffers are indexed once, deterministically per task index."""

    num_slots: int = 1024
    """Online only: the advertised task count. The orchestrator fixes an env's task count
    at startup, so an online buffer serves a virtual index space of this size."""

    max_candidates: int = 4096
    """Cap on indexed candidates. Steps are scanned newest-first, so the cap keeps the
    most recent rollouts (and bounds startup scan time on large buffers)."""

    max_steps_back: int | None = None
    """Recency window: only replay rollouts from the last N steps (None = no window).
    For online buffers this is also the eviction policy."""

    stop_conditions: list[str] | None = None
    """Only replay rollouts with these stop conditions (None = any non-error rollout).
    E.g. `["agent_completed"]` restricts recheck to conversations that actually finished."""

    env_name: str | None = None
    """Only replay rollouts stamped with this env name (`info.prime_rl.env_name`) — for
    buffers written by multi-env runs. Records without the stamp never match."""

    allow_container: bool = False
    """Allow continue/recheck over rollouts whose task ran in a container image. The
    container state the transcript references is gone — a fresh container is provisioned
    from the same image, so the model resumes in a reset world. Off by default; judge
    never provisions a container."""

    success_threshold: float = 0.5
    """Judge: the source rollout counts as correct when its reward exceeds this."""

    balance_labels: bool = True
    """Judge: interleave correct/incorrect source rollouts 1:1 (truncating to the smaller
    label) so a constant verdict can't score above chance."""

    recheck_instruction: str = RECHECK_INSTRUCTION
    """Recheck: the user turn appended to the finished conversation."""

    judge_instruction: str = JUDGE_INSTRUCTION
    """Judge: the prompt template; `{transcript}` is replaced with the rendered rollout."""

    max_message_chars: int = 2000
    """Judge: per-message truncation for the rendered transcript (single tool results in
    real buffers reach 150K chars)."""

    max_transcript_chars: int = 60000
    """Judge: total transcript budget; over it, middle messages are elided (the task
    statement and the trailing conversation are kept)."""

    seed: int = 0
    """Seed for online buffer sampling."""

    @model_validator(mode="before")
    @classmethod
    def _narrow_inner(cls, data):
        if isinstance(data, dict) and data.get("inner"):
            narrow_plugin_field(data, "inner", taskset_config_type)
        return data

    @model_validator(mode="after")
    def _validate_mode(self):
        if not self.buffer_dir:
            raise ValueError('replay taskset needs `buffer_dir` (a run\'s rollouts dir, or "self" under prime-rl)')
        if self.mode == "judge":
            if self.inner is not None:
                raise ValueError("judge scoring is self-contained; drop `inner` (or pick mode continue/recheck)")
        elif self.inner is None:
            raise ValueError(f"mode {self.mode!r} is scored by the original taskset; set `inner` to its config")
        return self


class ReplayTask(Task):
    """A derived task plus the lazy handle onto its source rollout. Stubs (from
    `load_tasks`) carry `prompt=None` and an empty `source_task`; `resolve_task`
    materializes the real thing."""

    kind: ReplayKind = "judge"
    source_task: dict = {}
    """The source rollout's saved task dict, verbatim — rebuilt into the original typed
    task for scoring/tool delegation."""
    original_reward: float = 0.0
    """The reward the source rollout actually received (the judge's ground truth)."""
    source_id: str = ""
    source_step: int = -1


class ReplayTaskset(Taskset[ReplayTask, ReplayTasksetConfig]):
    NEEDS_CONTAINER: ClassVar[bool] = False

    def __init__(self, config: ReplayTasksetConfig) -> None:
        super().__init__(config)
        self.inner: Taskset | None = None
        self.inner_task_type: type[Task] = Task
        if config.inner is not None:
            self.inner = taskset_class(config.inner.id)(config.inner)
            if discover_decorated(self.inner, "group_reward"):
                raise ValueError(
                    f"taskset {config.inner.id!r} defines @group_reward(s); replay cannot delegate group "
                    "scoring (the group would score against the replay task, not the original)"
                )
            if state_cls(type(self.inner)) is not State:
                raise ValueError(
                    f"taskset {config.inner.id!r} uses a custom State; replay rollouts build the base "
                    "State, so its typed state would never be populated"
                )
            if type(self.inner).user is not Taskset.user:
                raise ValueError(f"taskset {config.inner.id!r} defines a user simulator; replay does not support one")
            self.inner_task_type = task_type(config.inner.id)
            # The Environment reads both flags off the instance, so shadowing works.
            self.NEEDS_CONTAINER = type(self.inner).NEEDS_CONTAINER
        # Online buffers sample a fresh source per request; the whole GRPO group must
        # share that draw, so all its rollouts must arrive as one run_group request.
        self.REQUIRES_GROUP_ROLLOUTS = config.online
        self.buffer = ReplayBuffer(
            buffer_dir=config.buffer_dir,
            mode=config.mode,
            online=config.online,
            stop_conditions=config.stop_conditions,
            env_name=config.env_name,
            allow_container=config.allow_container,
            success_threshold=config.success_threshold,
            balance_labels=config.balance_labels,
            max_candidates=config.max_candidates,
            max_steps_back=config.max_steps_back,
            seed=config.seed,
        )

    # ------------------------------------------------------------------ tasks

    def load_tasks(self) -> list[ReplayTask]:
        candidates = self.buffer.scan()
        if self.config.online:
            num_tasks = self.config.num_slots
        else:
            if not candidates:
                raise ValueError(
                    f"replay buffer at {self.buffer.rollout_dir} has no replayable {self.config.mode!r} candidates"
                )
            num_tasks = len(candidates)
        return [ReplayTask(idx=i, kind=self.config.mode, prompt=None) for i in range(num_tasks)]

    async def resolve_task(self, idx: int, tasks: list[ReplayTask]) -> ReplayTask:
        candidate = await self.buffer.sample() if self.config.online else self.buffer.pick(idx)
        record = await self.buffer.read_record(candidate)
        return self._materialize(idx, candidate, record)

    def _materialize(self, idx: int, candidate, record: dict) -> ReplayTask:
        nodes = record["nodes"]
        children, _ = build_children(nodes)
        tree = main_tree(children)
        mode = self.config.mode
        if mode == "continue":
            prompt = continue_seed(nodes, candidate.fork_node)
        elif mode == "recheck":
            prompt = recheck_seed(nodes, children, tree, self.config.recheck_instruction)
        else:
            transcript = render_transcript(
                nodes, children, tree, self.config.max_message_chars, self.config.max_transcript_chars
            )
            prompt = self.config.judge_instruction.format(transcript=transcript)
        source_task = record["task"]
        if self.inner is not None:
            # Rebuild the typed original task now, so inner-taskset schema drift fails
            # loudly here — before any generation is spent on the rollout.
            self.inner_task_type.model_validate(source_task)
        provision = (
            {key: source_task.get(key) for key in ("image", "workdir", "timeout", "resources")}
            if mode != "judge"
            else {}
        )
        return ReplayTask(
            idx=idx,
            name=f"{mode}:{candidate.source_id[:8]}",
            prompt=prompt,
            kind=mode,
            source_task=source_task,
            original_reward=candidate.original_reward,
            source_id=candidate.source_id,
            source_step=candidate.step,
            **{key: value for key, value in provision.items() if value is not None},
        )

    def _inner_task(self, task: ReplayTask) -> Task:
        return self.inner_task_type.model_validate(task.source_task)

    # ---------------------------------------------------- original-world hooks

    def tools(self, task: ReplayTask) -> list[Toolset]:
        # Stubs (env serving inspects tools(tasks[0]) for shared placement) get none;
        # inner tasksets with shared-placement toolsets are therefore unsupported.
        if task.kind == "judge" or self.inner is None or not task.source_task:
            return []
        return self.inner.tools(self._inner_task(task))

    async def setup(self, task: ReplayTask, runtime: Runtime) -> None:
        if task.kind != "judge" and self.inner is not None:
            await self.inner.setup(self._inner_task(task), runtime)

    async def finalize(self, task: ReplayTask, trace: Trace, runtime: Runtime) -> None:
        if task.kind == "judge" or self.inner is None:
            return
        inner_task = self._inner_task(task)
        # The shallow copy shares nodes/info/state with the real trace, so anything the
        # inner taskset scrapes for its rewards lands where scoring will read it.
        await self.inner.finalize(inner_task, trace.model_copy(update={"task": inner_task}), runtime)

    # ------------------------------------------------------------------ scoring

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        task = trace.task
        if task.kind != "judge" and self.inner is not None:
            # Score with the original taskset, seen through a view whose `task` is the
            # rebuilt original: the shallow copy shares the rewards/metrics/info dicts,
            # so the inner rewards land on the real trace with their original names and
            # weights, while `trace.task` stays the ReplayTask for everything else.
            inner_task = self._inner_task(task)
            await self.inner.score(trace.model_copy(update={"task": inner_task}), runtime)
        await super().score(trace, runtime)

    @reward
    async def judge_correct(self, task: ReplayTask, trace: Trace) -> Mapping[str, float]:
        if task.kind != "judge":
            return {}
        verdict = parse_verdict(trace.last_reply or "")
        was_correct = task.original_reward > self.config.success_threshold
        return {"judge_correct": float(verdict is not None and verdict == was_correct)}

    @metric
    async def replay_stats(self, task: ReplayTask, trace: Trace) -> Mapping[str, float]:
        stats = {"replay/source_reward": task.original_reward, "replay/source_step": float(task.source_step)}
        if task.kind == "judge":
            stats["replay/judge_parseable"] = float(parse_verdict(trace.last_reply or "") is not None)
        return stats

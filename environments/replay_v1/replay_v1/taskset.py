"""The replay taskset: turns saved rollouts back into training tasks.

Reads a run's saved rollout files (``<output>/rollouts/step_*/train_rollouts.jsonl``,
plain WireTrace lines) and materializes one derived task kind per env entry, chosen by
``derivation.type``:

- ``continue``: resume from a context-compaction point — the seed is the post-compaction
  prompt (system + summary user message) recovered from the trace's message graph; scored
  by the original taskset.
- ``recheck``: the finished conversation plus an appended check-your-work turn; scored by
  the original taskset.
- ``judge``: a rendered transcript and the question "was this correct?"; the reward is
  whether the verdict matches the reward the rollout actually received. Self-contained —
  no original taskset needed.

Tasks bind lazily: ``load_tasks`` returns stubs fixing the index range and
``resolve_task`` reads one saved line per request. Online buffers (``buffer_dir =
"self"`` under prime-rl) sample a fresh source rollout per request and therefore force
group dispatch, so all GRPO group members share one source.

Run continue/recheck under the source env's harness (it must support message prompts);
judge under the tool-less ``null`` chat-loop harness.
"""

import asyncio
import logging
import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, PrivateAttr, SerializeAsAny, model_validator
from pydantic_config import BaseConfig
from verifiers.v1 import Runtime, Task, Taskset, TasksetConfig, Toolset, Trace, metric, reward, task_type
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.loaders import narrow_plugin_field, taskset_class, taskset_config_type
from verifiers.v1.state import State, state_cls

from replay_v1.buffer import ReplayBuffer
from replay_v1.surgery import (
    build_children,
    continue_seed,
    main_tree,
    recheck_seed,
    render_transcript,
    unwrap_source_task,
)

logger = logging.getLogger(__name__)

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
    """The model's verdict: the last line carrying exactly one verdict token. Lines
    quoting both options (an echo of the instruction) are not answers and are skipped;
    no unambiguous verdict line means unparseable (None)."""
    for line in reversed(reply.splitlines()):
        verdicts = {match.upper() for match in _VERDICT_RE.findall(line)}
        if len(verdicts) == 1:
            return verdicts.pop() == "CORRECT"
    return None


# ------------------------------------------------------------------ derivations


class _DelegatingDerivationConfig(BaseConfig):
    """Shared shape of the derivations scored by the original taskset."""

    inner: SerializeAsAny[TasksetConfig]
    """The original taskset's config — it scores the derived rollouts and provides
    their tools, so it must reproduce the source run's taskset config."""

    allow_container: bool = False
    """Allow sources whose task ran in a container image. The container state the
    transcript references is gone — a fresh container is provisioned from the same
    image, so the model resumes in a reset world. Off until that's fair (sandbox
    snapshotting)."""

    @model_validator(mode="before")
    @classmethod
    def _narrow_inner(cls, data):
        if isinstance(data, dict) and data.get("inner"):
            narrow_plugin_field(data, "inner", taskset_config_type)
        return data


class ContinueConfig(_DelegatingDerivationConfig):
    """Resume sources from their context-compaction points (one task per compaction)."""

    type: Literal["continue"] = "continue"


class RecheckConfig(_DelegatingDerivationConfig):
    """Append a check-your-work turn to finished sources and re-roll."""

    type: Literal["recheck"] = "recheck"

    instruction: str = RECHECK_INSTRUCTION
    """The user turn appended to the finished conversation."""


class JudgeConfig(BaseConfig):
    """Ask the model whether a source rollout was correct; reward = the verdict matches
    the reward that rollout actually received. Self-contained — no original taskset."""

    type: Literal["judge"] = "judge"

    instruction: str = JUDGE_INSTRUCTION
    """The prompt template; `{transcript}` is replaced with the rendered rollout."""

    success_threshold: float = 0.5
    """The source rollout counts as correct when its reward exceeds this."""

    max_transcript_chars: int = 60000
    """Total transcript budget, sized to the model's context; over it, middle messages
    are elided (the task statement and the trailing conversation are kept). Single
    messages are capped at 1/20 of it, so one huge tool result can't eat the budget."""


DerivationConfig = Annotated[ContinueConfig | RecheckConfig | JudgeConfig, Field(discriminator="type")]


# ------------------------------------------------------------------ config


class ReplayTasksetConfig(TasksetConfig):
    """Config for the replay taskset: where the buffer is, whose rollouts to replay,
    and which derivation this env serves (each derivation carries its own knobs)."""

    buffer_dir: str = ""
    """The saved-rollout dir to replay: a run's `rollouts` dir (or the run dir containing
    it). Under prime-rl the literal `"self"` resolves to this run's own rollout dir (an
    online buffer over the run's freshly written rollouts)."""

    derivation: DerivationConfig
    """The derived task this env serves: `{ type = "continue" | "recheck" | "judge", ... }`.
    One derivation per env entry — mix them (and set their ratios) with multiple
    `[[orchestrator.train.env]]` entries."""

    source_envs: list[str] | None = None
    """Which envs' rollouts to replay, by their stamped name (`info.prime_rl.env_name`).
    None (the default) replays every env except replay envs — deriving from a replay
    env's own outputs is a feedback loop unless chosen deliberately. Listing env names
    replays exactly those, and naming a replay env is that deliberate choice: chained
    derivations (recheck a recheck) are expressed as one replay env sourcing another.
    With an explicit list, records without the stamp never match."""

    online: bool = False
    """Treat the buffer as growing: rescan for new steps during training, read only
    barrier-complete steps, and sample a fresh source per request (which forces
    whole-group dispatch so GRPO groups share one source). Set automatically for
    `buffer_dir = "self"`; set it yourself only to watch another still-running run's
    dir. Offline buffers are indexed once, deterministically per task index."""

    @model_validator(mode="after")
    def _resolve(self):
        if not self.buffer_dir:
            raise ValueError('replay taskset needs `buffer_dir` (a run\'s rollouts dir, or "self" under prime-rl)')
        if self.buffer_dir == "self":
            # The orchestrator rewrites the sentinel to a resolved path before the env
            # server spawns, so onlineness must be pinned while the sentinel is visible.
            self.online = True
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

    # The rebuilt typed original task, cached at materialization so the per-rollout
    # hooks (tools/setup/finalize/score) don't re-validate the dict. Private attrs
    # bypass frozen and never serialize.
    _inner: Task | None = PrivateAttr(default=None)


class ReplayTaskset(Taskset[ReplayTask, ReplayTasksetConfig]):
    def __init__(self, config: ReplayTasksetConfig) -> None:
        super().__init__(config)
        self.derivation = config.derivation
        self.inner: Taskset | None = None
        self.inner_task_type: type[Task] = Task
        inner_config = getattr(self.derivation, "inner", None)
        if inner_config is not None:
            self.inner = taskset_class(inner_config.id)(inner_config)
            if discover_decorated(self.inner, "group_reward"):
                raise ValueError(
                    f"taskset {inner_config.id!r} defines @group_reward(s); replay cannot delegate group "
                    "scoring (the group would score against the replay task, not the original)"
                )
            if state_cls(type(self.inner)) is not State:
                raise ValueError(
                    f"taskset {inner_config.id!r} uses a custom State; replay rollouts build the base "
                    "State, so its typed state would never be populated"
                )
            if type(self.inner).user is not Taskset.user:
                raise ValueError(f"taskset {inner_config.id!r} defines a user simulator; replay does not support one")
            self.inner_task_type = task_type(inner_config.id)
            self.NEEDS_CONTAINER = self.inner.NEEDS_CONTAINER
        # Online buffers sample a fresh source per request; the whole GRPO group must
        # share that draw, so all its rollouts must arrive as one run_group request.
        self.REQUIRES_GROUP_ROLLOUTS = config.online
        self.buffer = ReplayBuffer(
            buffer_dir=config.buffer_dir,
            mode=self.derivation.type,
            online=config.online,
            source_envs=config.source_envs,
            allow_container=getattr(self.derivation, "allow_container", False),
            success_threshold=getattr(self.derivation, "success_threshold", 0.5),
        )

    # ------------------------------------------------------------------ tasks

    def load_tasks(self) -> list[ReplayTask]:
        if self.config.buffer_dir == "self":
            raise ValueError(
                'buffer_dir = "self" is resolved by the prime-rl orchestrator before env servers '
                "spawn; when serving this taskset standalone, pass an explicit rollouts dir"
            )
        candidates = self.buffer.scan()
        if self.config.online:
            # Sampling ignores the task index, so an online buffer serves exactly one
            # virtual task. (The orchestrator's no-ratio fallback weights envs by task
            # count — replay envs need an explicit ratio regardless.)
            num_tasks = 1
        else:
            if not candidates:
                raise ValueError(
                    f"replay buffer at {self.buffer.rollout_dir} has no replayable {self.derivation.type!r} candidates"
                )
            num_tasks = len(candidates)
        return [ReplayTask(idx=i, kind=self.derivation.type, prompt=None) for i in range(num_tasks)]

    async def resolve_task(self, task: ReplayTask) -> ReplayTask:
        # The read (~1MB line), graph walk, and prompt build all run off the event loop.
        if not self.config.online:
            candidate = self.buffer.pick(task.idx)
            return await asyncio.to_thread(self._materialize, task.idx, candidate)
        # An online source line can vanish under us (its run resumed and cleaned future
        # steps): drop the dangling candidate and draw again instead of failing forever.
        for _ in range(8):
            candidate = await self.buffer.sample()
            try:
                return await asyncio.to_thread(self._materialize, task.idx, candidate)
            except FileNotFoundError:
                logger.warning("replay source %s vanished; discarding candidate", candidate.path)
                self.buffer.discard(candidate)
        raise RuntimeError(f"replay buffer at {self.buffer.rollout_dir} keeps serving vanished source files")

    def _materialize(self, idx: int, candidate) -> ReplayTask:
        derivation = self.derivation
        record = self.buffer.read_record(candidate)
        nodes = record["nodes"]
        if derivation.type == "continue":
            prompt = continue_seed(nodes, candidate.fork_node)
        else:
            children, _ = build_children(nodes)
            tree = main_tree(children)
            if derivation.type == "recheck":
                prompt = recheck_seed(nodes, children, tree, derivation.instruction)
            else:
                total = derivation.max_transcript_chars
                transcript = render_transcript(nodes, children, tree, total // 20, total)
                prompt = derivation.instruction.format(transcript=transcript)
        # A chained source (a replay env sourcing another replay env) nests its lineage;
        # scoring, tools, and provisioning are always keyed on the innermost original.
        source_task = unwrap_source_task(record["task"])
        provision = (
            {key: value for key in ("image", "workdir", "timeout", "resources") if (value := source_task.get(key))}
            if derivation.type != "judge"
            else {}
        )
        task = ReplayTask(
            idx=idx,
            name=f"{derivation.type}:{candidate.source_id[:8]}",
            prompt=prompt,
            kind=derivation.type,
            source_task=source_task,
            original_reward=candidate.original_reward,
            source_id=candidate.source_id,
            source_step=candidate.step,
            **provision,
        )
        if self.inner is not None:
            # Rebuild the typed original task now — inner-taskset schema drift fails
            # loudly here, before any generation is spent — and cache it for the
            # per-rollout hooks.
            task._inner = self.inner_task_type.model_validate(source_task)
        return task

    def _inner_task(self, task: ReplayTask) -> Task:
        if task._inner is None:
            task._inner = self.inner_task_type.model_validate(task.source_task)
        return task._inner

    # ---------------------------------------------------- original-world hooks

    # `inner` exists exactly on the derivations the original taskset scores, so
    # `self.inner is not None` alone selects the delegating derivations.

    def tools(self, task: ReplayTask) -> list[Toolset]:
        # Stubs (env serving inspects tools(tasks[0]) for shared placement) get none;
        # inner tasksets with shared-placement toolsets are therefore unsupported.
        if self.inner is None or not task.source_task:
            return []
        return self.inner.tools(self._inner_task(task))

    async def setup(self, task: ReplayTask, runtime: Runtime) -> None:
        if self.inner is not None:
            await self.inner.setup(self._inner_task(task), runtime)

    async def finalize(self, task: ReplayTask, trace: Trace, runtime: Runtime) -> None:
        if self.inner is None:
            return
        inner_task = self._inner_task(task)
        # The shallow copy shares nodes/info/state with the real trace, so anything the
        # inner taskset scrapes for its rewards lands where scoring will read it.
        await self.inner.finalize(inner_task, trace.model_copy(update={"task": inner_task}), runtime)

    # ------------------------------------------------------------------ scoring

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        if self.inner is not None:
            # Score with the original taskset, seen through a view whose `task` is the
            # rebuilt original: the shallow copy shares the rewards/metrics/info dicts,
            # so the inner rewards land on the real trace with their original names and
            # weights, while `trace.task` stays the ReplayTask for everything else.
            inner_task = self._inner_task(trace.task)
            await self.inner.score(trace.model_copy(update={"task": inner_task}), runtime)
        await super().score(trace, runtime)

    @reward
    async def judge_correct(self, task: ReplayTask, trace: Trace) -> Mapping[str, float]:
        if task.kind != "judge":
            return {}
        verdict = parse_verdict(trace.last_reply or "")
        was_correct = task.original_reward > self.derivation.success_threshold
        return {"judge_correct": float(verdict is not None and verdict == was_correct)}

    @metric
    async def replay_stats(self, task: ReplayTask, trace: Trace) -> Mapping[str, float]:
        stats = {"replay/source_reward": task.original_reward, "replay/source_step": float(task.source_step)}
        if task.kind == "judge":
            stats["replay/judge_parseable"] = float(parse_verdict(trace.last_reply or "") is not None)
        return stats

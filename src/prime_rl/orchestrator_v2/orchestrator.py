"""Async-pipelined RL orchestrator v2.

``Orchestrator`` is a class with ``__init__(config)`` / ``start()`` / ``stop()``:

- ``__init__`` does the cheap synchronous setup (logger, ckpt manager, monitor).
- ``start()`` does the async setup (inference pools, env workers, weight
  broadcast, resume), constructs the four long-lived components
  (``WeightWatcher``, ``RolloutDispatcher``, ``TrainBatcher``, ``IntervalLogger``),
  spawns each as an ``asyncio.Task``, and then blocks on the shared
  ``stopped`` event. Components signal completion by setting that event (e.g.
  the batcher sets it when ``progress.step >= max_steps``).
- ``stop()`` just sets the shared event. Cleanup is driven from inside
  ``start()`` after the event fires, so callers get one obvious path.

No control-flow exceptions — components and the orchestrator coordinate via
``orch.stopped: asyncio.Event``.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import time

import tomli_w

import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive imports
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.filters import setup_filters
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.utils import get_weight_dir, set_default_executor
from prime_rl.orchestrator.vf_utils import intercept_vf_logging
from prime_rl.orchestrator_v2.batcher import TrainBatcher
from prime_rl.orchestrator_v2.ckpt import Progress, setup_ckpt_manager
from prime_rl.orchestrator_v2.dispatcher import RolloutDispatcher
from prime_rl.orchestrator_v2.log_loop import IntervalLogger
from prime_rl.orchestrator_v2.policy import Policy
from prime_rl.orchestrator_v2.watcher import WeightWatcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transport import setup_training_batch_sender
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.client import init_nccl_broadcast, setup_inference_pool
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import get_log_dir
from prime_rl.utils.usage_reporter import UsageReporter
from prime_rl.utils.utils import (
    clean_exit,
    get_env_ids_to_install,
    install_env,
    resolve_latest_ckpt_step,
)

monkey_patch_oai_iterable_types()
monkey_patch_chat_completion_logprobs()


# Hard wall-clock budget for the v2 orchestrator's post-training cleanup.
# Mirrors the legacy orchestrator's safety: persist artifacts before this point,
# then force-exit if graceful shutdown wedges (env-server ZMQ recv, vLLM admin
# aclose, etc).
SHUTDOWN_TIMEOUT_S = 300


class Orchestrator:
    """v2 orchestrator. Use ``await Orchestrator(config).start()`` to run; call
    ``stop()`` from outside (or rely on the batcher to set ``stopped`` once
    ``max_steps`` is reached). Cleanup happens at the tail of ``start()``."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
        intercept_vf_logging(logger="verifiers.serve", level="WARN")
        self.logger.info(f"Starting orchestrator v2 ({config.training_mode})")

        if config.bench:
            self.logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

        # Cheap synchronous setup. Heavy async work (inference pools, env
        # workers, weight broadcast, resume) happens in ``start()``.
        self.progress = Progress()
        self.ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)

        # Stopped event — set by the batcher on max_steps, by ``stop()``, or
        # by a fatal error. ``start()`` waits on it and then drives cleanup.
        self.stopped = asyncio.Event()

        # All component / shared-state references — populated in ``start()``.
        self.tokenizer = None
        self.renderer = None
        self.mm_token_type_ids_mapping: dict[int, int] | None = None
        self.student_inference = None
        self.teacher_inference = None
        self.monitor = None
        self.heart: Heartbeat | None = None
        self.usage_reporter: UsageReporter | None = None
        self.inference_metrics: InferenceMetricsCollector | None = None
        self.pre_filters = []  # populated in start()
        self.post_filters = []
        self.train_envs: TrainEnvs | None = None
        self.eval_envs: EvalEnvs | None = None
        self.sender = None
        self.policy = Policy(version=0, model_name="")
        self.lora_name: str | None = None
        self.resume_step: int | None = None

        self.dispatcher: RolloutDispatcher | None = None
        self.batcher: TrainBatcher | None = None
        self.watcher: WeightWatcher | None = None
        self.log_loop: IntervalLogger | None = None

        # Spawned tasks — owned by the orchestrator so ``stop()`` can cancel them.
        self.tasks: list[asyncio.Task] = []

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Async setup: install envs, load models/pools, prepare env workers,
        resume from checkpoint, instantiate components. Idempotent-safe — the
        orchestrator class is single-use, but the setup is broken into pieces
        in case a caller wants to override one."""
        config = self.config
        set_default_executor()

        # Persist the resolved config alongside the run.
        config_dir = config.output_dir / "control"
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "orch.toml", "wb") as f:
            tomli_w.dump(config.model_dump(exclude_none=True, mode="json"), f)

        # Install envs (train + eval).
        env_ids_to_install = set(get_env_ids_to_install(config.train.env))
        if config.eval is not None:
            env_ids_to_install.update(get_env_ids_to_install(config.eval.env))
        for env_id in env_ids_to_install:
            install_env(env_id, prerelease=config.env_install_prerelease)

        self.logger.info(f"Initializing tokenizer ({config.tokenizer})")
        self.tokenizer = setup_tokenizer(config.tokenizer)

        # Student inference pool.
        self.logger.info(
            f"Initializing student inference pool (base_url={', '.join(config.student.client.base_url)}, "
            f"model={config.student.model.name})"
        )
        self.renderer, self.student_inference = await setup_student_inference_pool(
            config=config, tokenizer=self.tokenizer
        )
        self.mm_token_type_ids_mapping = (
            getattr(self.renderer, "mm_token_type_id_map", None) if self.renderer is not None else None
        )
        if self.mm_token_type_ids_mapping == {}:
            self.mm_token_type_ids_mapping = None

        if config.teacher is not None:
            self.logger.info(
                f"Initializing teacher inference pool (base_url={', '.join(config.teacher.client.base_url)}, "
                f"model={config.teacher.model.name})"
            )
            self.teacher_inference = await setup_inference_pool(
                config.teacher.client,
                model_name=config.teacher.model.name,
                train_client_type="openai_chat_completions",
            )

        self.logger.info(f"Initializing monitor (wandb={config.wandb}, prime_monitor={config.prime_monitor})")
        self.monitor = setup_monitor(
            wandb_config=config.wandb,
            prime_config=config.prime_monitor,
            output_dir=config.output_dir,
            tokenizer=self.tokenizer,
            run_config=config,
            keep_full_history=config.bench,
        )

        if config.heartbeat is not None:
            self.heart = Heartbeat(config.heartbeat.url)

        usage_base_url = os.environ.get("PI_USAGE_BASE_URL")
        usage_api_key = os.environ.get("PI_USAGE_API_KEY")
        if usage_base_url and usage_api_key:
            self.usage_reporter = UsageReporter()

        # Filters.
        self.pre_filters = setup_filters(config.pre_batch_filters, vocab_size=self.tokenizer.vocab_size)
        self.post_filters = setup_filters(config.post_batch_filters, vocab_size=self.tokenizer.vocab_size)

        # Envs.
        self.logger.info("Loading training environments")
        self.train_envs = TrainEnvs(config.train.env)
        if config.training_mode == "sft":
            for env in self.train_envs:
                env.sampling_args.pop("logprobs", None)
        self.logger.info(f"Loaded {len(self.train_envs)} training environment(s) ({', '.join(self.train_envs.names)})")
        await self.train_envs.start(
            log_dir=get_log_dir(config.output_dir.parent) / "envs" / "train",
            log_level=config.log.vf_level,
            json_logging=config.log.json_logging,
        )
        self.logger.success("Train environment(s) ready")

        if config.eval is not None:
            self.logger.info("Loading eval environment(s)")
            self.eval_envs = EvalEnvs(config.eval.env)
            self.logger.info(f"Loaded {len(self.eval_envs)} eval environment(s) ({', '.join(self.eval_envs.names)})")
            await self.eval_envs.start(
                log_dir=get_log_dir(config.output_dir.parent) / "envs" / "eval",
                log_level=config.log.vf_level,
                json_logging=config.log.json_logging,
            )
            self.logger.success("Eval environment(s) ready")

        # Resume.
        if config.ckpt is not None and config.ckpt.resume_step is not None and self.ckpt_manager is not None:
            if config.ckpt.resume_step == -1:
                self.resume_step = resolve_latest_ckpt_step(self.ckpt_manager.ckpt_dir)
            else:
                self.resume_step = config.ckpt.resume_step

        # Initial policy state — student model name; on resume below we may
        # advance ``policy.version`` and the LoRA-adapter model name.
        self.policy.model_name = self.student_inference.model_name

        # Wait for inference pools to be reachable.
        self.logger.info("Waiting for student inference pool to be ready")
        await self.student_inference.wait_for_ready(config.student.model.name)
        self.logger.success("Student inference pool ready")
        if self.teacher_inference is not None:
            assert config.teacher is not None
            self.logger.info("Waiting for teacher inference pool to be ready")
            await self.teacher_inference.wait_for_ready(config.teacher.model.name)
            self.logger.success("Teacher inference pool ready")

        # Inference metrics collector.
        if config.wandb is not None and config.collect_inference_metrics:
            self.inference_metrics = InferenceMetricsCollector(self.student_inference.admin_clients)
            await self.inference_metrics.start()

        # Weight broadcast init.
        self.logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
        if config.weight_broadcast.type == "nccl":
            await init_nccl_broadcast(
                self.student_inference.admin_clients,
                config.weight_broadcast.host,
                config.weight_broadcast.port,
                config.weight_broadcast.timeout,
                inference_world_size=config.weight_broadcast.inference_world_size,
                quantize_in_weight_transfer=config.weight_broadcast.quantize_in_weight_transfer,
            )

        # Training-batch sender.
        self.logger.info(f"Initializing training batch sender ({config.rollout_transport})")
        self.sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)

        self.lora_name = config.student.model.lora.name if config.student.model.lora else None

        # Restore from checkpoint (if resuming).
        if self.resume_step is not None and self.ckpt_manager is not None:
            self.ckpt_manager.load(self.progress, step=self.resume_step)
            self.logger.info(f"Resuming v2 orchestrator from checkpoint step {self.resume_step}")
            check_exists = config.weight_broadcast.type != "nccl"
            wait_timeout = config.ckpt.wait_for_weights_timeout if config.ckpt else None
            weights_path = get_weight_dir(
                config.output_dir, self.progress.step, check_exists=check_exists, wait_timeout=wait_timeout
            )
            await self.student_inference.update_weights(weights_path, lora_name=self.lora_name, step=self.progress.step)
            if self.lora_name is not None:
                self.student_inference.update_model_name(self.lora_name)
                self.policy.model_name = self.lora_name
            self.policy.version = self.progress.step
        else:
            self.logger.info("Training from scratch")

        # Wire components.
        self.dispatcher = RolloutDispatcher(
            config=config,
            train_envs=self.train_envs,
            eval_envs=self.eval_envs,
            student_inference=self.student_inference,
            teacher_inference=self.teacher_inference,
            policy=self.policy,
            resume_step=self.resume_step,
        )
        self.batcher = TrainBatcher(self)
        self.watcher = WeightWatcher(
            config=config,
            policy=self.policy,
            student_inference=self.student_inference,
            observers=[self.dispatcher],
            lora_name=self.lora_name,
            ckpt_step=self.progress.step,
        )
        self.log_loop = IntervalLogger(self)

    async def start(self) -> None:
        """Run the orchestrator until shutdown. Drives setup, spawns the
        component tasks, blocks on ``self.stopped``, then cleans up."""
        await self.setup()
        config = self.config
        self.logger.info(f"Starting orchestrator v2 loop (max_steps={config.max_steps or 'infinite'})")
        start_time = time.perf_counter()

        # Spawn the component loops. Each component's ``start()`` is its main
        # loop; we run them as concurrent tasks.
        assert self.dispatcher and self.batcher and self.watcher and self.log_loop
        self.tasks = [
            asyncio.create_task(self.dispatcher.start(), name="dispatcher"),
            asyncio.create_task(self.batcher.start(), name="batcher"),
            asyncio.create_task(self.watcher.start(), name="watcher"),
            asyncio.create_task(self.log_loop.start(), name="log_loop"),
        ]

        # Block until something sets ``stopped``: the batcher on max_steps,
        # an external ``stop()`` call, or a future fatal-error path.
        await self.stopped.wait()
        self.logger.success(f"Orchestrator v2 step loop done in {time.perf_counter() - start_time:.1f}s")

        # Final evals before shutdown. Uses the legacy ``EvalEnv.evaluate``
        # directly — the dispatcher / batcher tasks are already winding down.
        if self.eval_envs is not None and config.eval is not None:
            self.logger.info("Running final evals")
            await asyncio.gather(
                *(
                    eval_env.evaluate(
                        model_name=self.student_inference.model_name,
                        get_client=self.student_inference.get_eval_client,
                        step=self.progress.step,
                        cache_salt=str(self.progress.step),
                    )
                    for eval_env in self.eval_envs
                ),
                return_exceptions=True,
            )

        assert self.monitor is not None
        self.monitor.save_final_summary()

        if self.ckpt_manager is not None:
            self.logger.info("Writing final v2 checkpoint")
            self.ckpt_manager.save(self.progress, step=self.progress.step)

        await self.shutdown()
        self.logger.success("Orchestrator v2 finished.")

        # Return free glibc heap pages to the OS so the launcher's exit isn't
        # held by malloc bookkeeping from numpy/pandas allocations.
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception as e:
            get_logger().debug(f"malloc_trim(0) failed: {e}")

    async def stop(self) -> None:
        """Signal the run to wind down. ``start()`` will see this and drive cleanup."""
        self.stopped.set()

    async def shutdown(self) -> None:
        """Bounded best-effort cleanup. Stops each component, drains the inflight
        rollouts, and tears down env workers. Has a global timeout so a wedged
        peer can't keep the process alive forever — training artifacts are
        already persisted before this is reached."""

        async def do_shutdown() -> None:
            if self.sender is not None:
                self.sender.close()
            if self.batcher is not None:
                await self.batcher.stop()
            if self.dispatcher is not None:
                await self.dispatcher.stop()
            if self.watcher is not None:
                await self.watcher.stop()
            if self.log_loop is not None:
                await self.log_loop.stop()
            for task in self.tasks:
                await safe_cancel(task)
            self.tasks.clear()
            if self.inference_metrics is not None:
                await self.inference_metrics.stop()
            if self.student_inference is not None:
                await self.student_inference.stop()
            if self.teacher_inference is not None:
                await self.teacher_inference.stop()
            if self.train_envs is not None:
                self.train_envs.shutdown()
            if self.eval_envs is not None:
                self.eval_envs.shutdown()
            if self.usage_reporter is not None:
                self.usage_reporter.close()

        task = asyncio.create_task(do_shutdown())
        _, pending = await asyncio.wait({task}, timeout=SHUTDOWN_TIMEOUT_S)
        if pending:
            self.logger.warning(
                f"Orchestrator v2 shutdown did not complete within {SHUTDOWN_TIMEOUT_S}s; "
                "forcing process exit. Training artifacts are already persisted."
            )
            os._exit(0)
        await task


@clean_exit
async def run_orchestrator(config: OrchestratorConfig) -> None:
    """Top-level entrypoint: instantiate ``Orchestrator`` and run it.

    Wrapped in ``@clean_exit`` so wandb is flushed on exit (success or crash);
    keeps that cleanup out of the ``Orchestrator`` class proper.
    """
    await Orchestrator(config).start()


async def setup_student_inference_pool(*, config: OrchestratorConfig, tokenizer):
    """Mirror of ``prime_rl.orchestrator.orchestrator.setup_student_inference_pool``.

    Kept inline so the v2 orchestrator can evolve client setup independently
    (e.g. drop the teacher_inference for sft mode in a follow-up) without
    touching the legacy code path.
    """
    from renderers.base import create_renderer

    client_config = config.student.client
    model_name = config.student.model.name

    if config.use_renderer:
        renderer = create_renderer(
            tokenizer,
            renderer=config.renderer.name,
            tool_parser=config.renderer.tool_parser,
            reasoning_parser=config.renderer.reasoning_parser,
            preserve_all_thinking=config.renderer.preserve_all_thinking,
            preserve_thinking_between_tool_calls=config.renderer.preserve_thinking_between_tool_calls,
        )
        get_logger().info(f"Initialized {type(renderer).__name__} for {model_name}")
        inference_pool = await setup_inference_pool(
            client_config,
            model_name=model_name,
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_name=config.renderer.name,
            tool_parser=config.renderer.tool_parser,
            reasoning_parser=config.renderer.reasoning_parser,
            renderer_pool_size=config.renderer.pool_size,
            preserve_all_thinking=config.renderer.preserve_all_thinking,
            preserve_thinking_between_tool_calls=config.renderer.preserve_thinking_between_tool_calls,
        )
        get_logger().info("Using direct renderer rollout client")
        return renderer, inference_pool

    get_logger().info("Using MITO (openai_chat_completions) for rollouts")
    inference_pool = await setup_inference_pool(
        client_config,
        model_name=model_name,
        train_client_type="openai_chat_completions",
        eval_client_type="openai_chat_completions",
    )
    return None, inference_pool


def main() -> None:
    """Entrypoint for direct invocation via ``python -m prime_rl.orchestrator_v2.orchestrator``."""
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    set_proc_title("OrchestratorV2")
    import uvloop

    uvloop.install()
    asyncio.run(run_orchestrator(cli(OrchestratorConfig)))


if __name__ == "__main__":
    main()

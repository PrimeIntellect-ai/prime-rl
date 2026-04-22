import asyncio
import os
import time
from pathlib import Path
from typing import Iterator

import httpx
import tomli_w
import torch
import verifiers as vf

from prime_rl.configs.orchestrator import (
    DefaultAdvantageConfig,
    OrchestratorConfig,
    TrainEnvConfig,
)
from prime_rl.orch2.engine import EnvWorker, Group, Pool, RolloutEngine
from prime_rl.orch2.watcher import WeightWatcher
from prime_rl.orchestrator.advantage import AdvantageInputs, default_advantage_fn
from prime_rl.orchestrator.trajectories import (
    interleave_rollout,
    pretokenize_rollout_trajectory,
)
from prime_rl.orchestrator.vf_utils import get_completion_len
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transport import TrainingBatch, TrainingSample, setup_training_batch_sender
from prime_rl.transport.base import TrainingBatchSender
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import get_monitor, setup_monitor
from prime_rl.utils.pathing import get_broadcast_dir

# ---------- inference admin (single endpoint) ----------


class InferenceAdmin:
    """Single-endpoint admin client for health, model check, and weight update."""

    def __init__(self, base_url: str, api_key: str | None = None):
        base_url = base_url.rstrip("/").removesuffix("/v1")
        headers = {}
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(None),
        )

    async def wait_healthy(self, timeout: float = 1800.0, interval: float = 1.0) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                r = await self.client.get("/health")
                if r.status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(interval)
        raise TimeoutError(f"Inference server {self.base_url} not healthy after {timeout}s")

    async def check_model(self, model_name: str) -> None:
        r = await self.client.get("/v1/models")
        r.raise_for_status()
        models = r.json().get("data", [])
        if not any(m["id"] == model_name for m in models):
            raise ValueError(f"Model '{model_name}' not found on {self.base_url}")

    async def update_weights(self, weight_dir: Path) -> None:
        # vLLM's update_weights_from_path passes the string straight to HF's
        # DefaultModelLoader, which first validates as a repo ID. A relative
        # path with multiple slashes trips that check before the local-path
        # fallback. Resolve to absolute here.
        path = weight_dir.resolve().as_posix()
        (await self.client.post("/pause", params={"mode": "keep", "clear_cache": "false"})).raise_for_status()
        try:
            (await self.client.post("/update_weights", json={"weight_dir": path})).raise_for_status()
        finally:
            (await self.client.post("/resume")).raise_for_status()


# ---------- train batcher ----------
#
# Reads groups, scores advantages, accumulates a batch, converts to
# TrainingBatch and ships to trainer. Single coroutine, single loop.


class _Done(Exception):
    """Raised by the batcher when max_steps has been reached. Caught by run()."""


class TrainBatcher:
    def __init__(
        self,
        in_q: asyncio.Queue[Group],
        tokenizer,
        sender: TrainingBatchSender,
        engine: RolloutEngine,
        batch_size: int,
        advantage_cfg: DefaultAdvantageConfig,
        max_steps: int | None = None,
        max_async_level: int = 1,
    ):
        self.in_q = in_q
        self.tokenizer = tokenizer
        self.sender = sender
        self.engine = engine
        self.batch_size = batch_size
        self.advantage_cfg = advantage_cfg
        self.max_steps = max_steps
        self.max_async_level = max_async_level
        self.step = 0
        self.logger = get_logger()

    async def run(self) -> None:
        buf: list[vf.RolloutOutput] = []
        while True:
            group = await self.in_q.get()
            if group.pool_id.startswith("eval"):
                continue
            self._score(group)
            buf.extend(group.rollouts)
            while len(buf) >= self.batch_size:
                rollouts, buf = buf[: self.batch_size], buf[self.batch_size :]
                # Async-level barrier: don't ship more than max_async_level batches
                # ahead of the latest policy version. This throttles orch2 when
                # weight updates fall behind, and cascades backpressure through
                # the groups queue to the engine.
                while self.step - self.engine.policy_version > self.max_async_level:
                    await asyncio.sleep(0.1)
                await self._ship(rollouts)
                if self.max_steps is not None and self.step >= self.max_steps:
                    raise _Done()

    def _score(self, group: Group) -> None:
        rewards = torch.tensor([[r.get("reward", 0.0) for r in group.rollouts]], dtype=torch.float32)
        lens = torch.tensor([[get_completion_len(r) for r in group.rollouts]], dtype=torch.int64)
        out = default_advantage_fn(
            AdvantageInputs(rewards=rewards, completion_lengths=lens),
            length_shaping=self.advantage_cfg.length_shaping,
        )
        for r, a in zip(group.rollouts, out.advantages[0].tolist()):
            r["advantage"] = a

    async def _ship(self, rollouts: list[vf.RolloutOutput]) -> None:
        t0 = time.perf_counter()
        samples = await asyncio.to_thread(self._convert, rollouts)
        convert_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        batch = TrainingBatch(examples=samples, step=self.step)
        await asyncio.to_thread(self.sender.send, batch)
        send_time = time.perf_counter() - t1

        rewards = [r.get("reward", 0.0) for r in rollouts]
        advs = [r.get("advantage") or 0.0 for r in rollouts]
        seq_lens = [get_completion_len(r) for r in rollouts]
        reward_mean = sum(rewards) / len(rewards)
        adv_abs = sum(abs(a) for a in advs) / len(advs)
        seq_mean = sum(seq_lens) / len(seq_lens)

        self.logger.success(
            f"Step {self.step} | "
            f"Batch: {len(samples)} samples ({len(rollouts)} rollouts) | "
            f"Reward: {reward_mean:+.4f} | |Adv|: {adv_abs:.4f} | "
            f"Seq: {seq_mean:.0f} tok | "
            f"Version: {self.engine.policy_version} | "
            f"Convert: {convert_time:.2f}s | Ship: {send_time:.2f}s"
        )
        get_monitor().log(
            {
                "train/reward/mean": reward_mean,
                "train/advantage/abs_mean": adv_abs,
                "train/seq_len/mean": seq_mean,
                "train/batch_size": len(samples),
                "train/policy_version": self.engine.policy_version,
                "time/convert": convert_time,
                "time/ship": send_time,
            },
            step=self.step,
        )
        self.step += 1

    def _convert(self, rollouts: list[vf.RolloutOutput]) -> list[TrainingSample]:
        samples: list[TrainingSample] = []
        for r in rollouts:
            pretokenize_rollout_trajectory(r, self.tokenizer)
            out = interleave_rollout(r)
            if out is None:
                continue
            for s in out:
                s.advantage = r.get("advantage")
                s.reward = r.get("reward")
            samples.extend(out)
        return samples


# ---------- entrypoint ----------


def cycle_forever(ds) -> Iterator[dict]:
    i = 0
    while True:
        for row in ds:
            r = dict(row)
            r["example_id"] = i
            i += 1
            yield r


def build_pool(env_cfg: TrainEnvConfig, cfg: OrchestratorConfig, concurrency: int) -> Pool:
    env = vf.load_environment(env_cfg.stripped_id, **env_cfg.args)
    client = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=cfg.client.base_url[0],
        api_key_var=cfg.client.api_key_var,
        timeout=cfg.client.timeout,
        connect_timeout=cfg.client.connect_timeout,
    )
    worker = EnvWorker(
        env=env,
        client=client,
        model=cfg.model.name,
        sampling_args=env_cfg.sampling.to_sampling_args(),
    )
    return Pool(
        id=env_cfg.resolved_name,
        dataset=cycle_forever(env.get_dataset()),
        env=worker,
        concurrency=concurrency,
        rollouts_per_group=cfg.rollouts_per_example,
        max_off_policy=cfg.max_off_policy_steps,
    )


async def run(cfg: OrchestratorConfig) -> None:
    assert cfg.max_inflight_rollouts is not None
    assert cfg.batch_size is not None
    assert isinstance(cfg.advantage, DefaultAdvantageConfig), "orch2 minimal only supports default advantage"
    assert len(cfg.client.base_url) == 1, "orch2 assumes a single inference endpoint"

    logger = get_logger()
    logger.info(f"Output dir: {cfg.output_dir}")
    logger.info(f"Model: {cfg.model.name}")

    # Trainer reads orchestrator config from output_dir/control/orch.toml
    # (see prime_rl.trainer.runs.RunManager.get_orchestrator_config).
    control_dir = cfg.output_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(cfg.model_dump(exclude_none=True, mode="json"), f)
    logger.info(f"Wrote orch config to {control_dir / 'orch.toml'}")
    logger.info(
        f"Batch size: {cfg.batch_size} | Rollouts/example: {cfg.rollouts_per_example} | "
        f"Max in-flight: {cfg.max_inflight_rollouts} | Max off-policy: {cfg.max_off_policy_steps}"
    )

    tokenizer = setup_tokenizer(cfg.tokenizer)
    logger.info(f"Tokenizer ready: {cfg.tokenizer.name}")

    setup_monitor(
        wandb_config=cfg.wandb,
        output_dir=cfg.output_dir,
        tokenizer=tokenizer,
        run_config=cfg,
        prime_config=cfg.prime_monitor,
    )
    if cfg.wandb is not None:
        logger.info(f"Wandb monitor ready (project={cfg.wandb.project}, name={cfg.wandb.name})")

    num_envs = len(cfg.train.env)
    per_env_concurrency = max(1, cfg.max_inflight_rollouts // num_envs)
    pools = [build_pool(env_cfg, cfg, per_env_concurrency) for env_cfg in cfg.train.env]
    for pool in pools:
        logger.info(
            f"Pool '{pool.id}' ready | concurrency={pool.concurrency} | "
            f"rollouts_per_group={pool.rollouts_per_group} | max_off_policy={pool.max_off_policy}"
        )

    groups_q: asyncio.Queue[Group] = asyncio.Queue()

    engine = RolloutEngine(pools, groups_q)
    training_sender = setup_training_batch_sender(cfg.output_dir, cfg.rollout_transport)
    batcher = TrainBatcher(
        groups_q,
        tokenizer,
        training_sender,
        engine,
        cfg.batch_size,
        cfg.advantage,
        cfg.max_steps,
        cfg.max_async_level,
    )
    logger.info(f"Training batch sender ready ({cfg.rollout_transport.type})")

    admin = InferenceAdmin(cfg.client.base_url[0], os.getenv(cfg.client.api_key_var, "EMPTY"))
    logger.info(f"Admin client ready ({admin.base_url})")

    logger.info(f"Weight broadcast mode: {cfg.weight_broadcast.type}")
    if cfg.weight_broadcast.type == "nccl":
        logger.warning("NCCL weight broadcast is not wired in orch2 yet — falling back to filesystem polling")

    logger.info("Waiting for inference server to be healthy...")
    t0 = time.perf_counter()
    await admin.wait_healthy()
    logger.success(f"Inference server ready ({time.perf_counter() - t0:.1f}s)")

    await admin.check_model(cfg.model.name)
    logger.success(f"Model '{cfg.model.name}' loaded on inference server")

    async def on_new_weights(step: int, weight_dir: Path) -> None:
        t0 = time.perf_counter()
        await admin.update_weights(weight_dir)
        engine.on_new_version(step)
        logger.success(f"Weights updated to step {step} in {time.perf_counter() - t0:.2f}s ({weight_dir})")

    watcher = WeightWatcher(get_broadcast_dir(cfg.output_dir), on_new_weights)

    logger.success("Orch2 starting — producing rollouts")
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(engine.run())
            tg.create_task(batcher.run())
            tg.create_task(watcher.run())
    except* _Done:
        logger.success(f"Orch2 finished: reached max_steps={cfg.max_steps}")


def main() -> None:
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")
    cfg = cli(OrchestratorConfig)
    setup_logger(
        log_level=cfg.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=cfg.log.json_logging,
    )
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()

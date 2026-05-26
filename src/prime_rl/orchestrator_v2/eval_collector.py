"""EvalCollector: aggregates eval trajectories and flushes metrics per epoch.

Single responsibility. The orchestrator calls ``handle(traj, expected, fired_envs)``
for each ``kind == "eval"`` trajectory pulled from the dispatcher queue.
Trajectories are bucketed by ``eval_step``; once the count for that step
reaches ``expected`` the collector computes per-env reward / completion-len /
pass@k stats and logs them to the monitor.

No dispatcher reach-ins, no orchestrator back-pointer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.eval_utils import compute_pass_at_k
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.vf_utils import get_seq_len, save_rollouts
from prime_rl.orchestrator_v2.dispatcher import Trajectory
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_rollout_dir, get_step_path


class EvalCollector:
    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        monitor,
        eval_envs: EvalEnvs | None,
        post_filters: list[RolloutFilter],
    ) -> None:
        self.config = config
        self.monitor = monitor
        self.eval_envs = eval_envs
        self.post_filters = post_filters
        self.logger = get_logger()

        self.buf: dict[int, list[Trajectory]] = defaultdict(list)
        self.received: dict[int, int] = defaultdict(int)
        self.last_flushed_step: int | None = None

    def handle(self, traj: Trajectory, expected: int | None, fired_envs: set[str]) -> None:
        """Bucket the trajectory; flush per-env eval metrics if the epoch is complete."""
        assert traj.eval_step is not None, "eval Trajectory missing eval_step"

        if self.post_filters:
            apply_filters(self.post_filters, traj.rollouts)

        self.buf[traj.eval_step].append(traj)
        self.received[traj.eval_step] += len(traj.rollouts)

        if expected is None or self.received[traj.eval_step] < expected:
            return

        trajs = self.buf.pop(traj.eval_step)
        received = self.received.pop(traj.eval_step)
        self.flush(traj.eval_step, trajs, expected, received, fired_envs)
        self.last_flushed_step = traj.eval_step

    def flush(
        self,
        eval_step: int,
        trajs: list[Trajectory],
        expected: int,
        received: int,
        envs_fired: set[str],
    ) -> None:
        per_env: dict[str, list[Trajectory]] = defaultdict(list)
        for t in trajs:
            per_env[t.env_name].append(t)

        to_log: dict[str, Any] = {"step": eval_step}
        all_rewards: list[float] = []
        all_lens: list[int] = []

        for env_name, env_trajs in per_env.items():
            rollouts = [r for t in env_trajs for r in t.rollouts]
            if not rollouts:
                continue
            rewards = [r.get("reward", 0.0) for r in rollouts]
            lens = [get_seq_len(r) for r in rollouts]
            all_rewards.extend(rewards)
            all_lens.extend(lens)

            no_response_rate = sum(1 for r in rollouts if not r.get("completion")) / len(rollouts)
            truncation_rate = sum(1 for r in rollouts if r.get("is_truncated")) / len(rollouts)
            prefix = f"eval/{env_name}"
            group_size = 1
            if self.eval_envs is not None:
                try:
                    group_size = self.eval_envs.get(env_name).config.group_size
                except KeyError:
                    pass
            to_log[f"{prefix}/avg@{group_size}"] = float(sum(rewards) / len(rewards))
            to_log[f"{prefix}/reward/mean"] = float(sum(rewards) / len(rewards))
            to_log[f"{prefix}/completion_len/mean"] = float(sum(lens) / len(lens))
            to_log[f"{prefix}/completion_len/max"] = float(max(lens))
            to_log[f"{prefix}/completion_len/min"] = float(min(lens))
            to_log[f"{prefix}/is_truncated/mean"] = float(truncation_rate)
            to_log[f"{prefix}/no_response/mean"] = float(no_response_rate)
            to_log[f"{prefix}/n_rollouts"] = float(len(rollouts))
            to_log[f"{prefix}/n_examples"] = float(len(env_trajs))

            unique_rewards = {float(r) for r in rewards}
            could_be_binary = unique_rewards.issubset({0.0, 1.0})
            if could_be_binary:
                per_example_rewards = [[float(r.get("reward", 0.0)) for r in t.rollouts] for t in env_trajs]
                pass_at_k_per_example = [compute_pass_at_k(rs) for rs in per_example_rewards]
                if pass_at_k_per_example:
                    keys = set().union(*(d.keys() for d in pass_at_k_per_example))
                    for k in keys:
                        values = [d.get(k, 0.0) for d in pass_at_k_per_example]
                        to_log[f"{prefix}/{k}"] = float(sum(values) / len(values))

            step_path = get_step_path(get_rollout_dir(self.config.output_dir), eval_step)
            save_rollouts(rollouts, step_path / "eval_rollouts.jsonl", exclude_keys={"trajectory"})
            self.monitor.log_eval_samples(rollouts, env_name=env_name, step=eval_step)

        if all_rewards:
            to_log["eval/reward/mean"] = float(sum(all_rewards) / len(all_rewards))
            to_log["eval/completion_len/mean"] = float(sum(all_lens) / len(all_lens))
            to_log["eval/n_rollouts"] = float(len(all_rewards))

        envs_str = ",".join(sorted(envs_fired))
        coverage = f"{received}/{expected}"
        self.logger.success(
            f"Eval @ step={eval_step} | Envs: {envs_str} | "
            f"Reward: {to_log.get('eval/reward/mean', float('nan')):.4f} | "
            f"Rollouts: {coverage}"
        )
        self.monitor.log(to_log, step=eval_step)

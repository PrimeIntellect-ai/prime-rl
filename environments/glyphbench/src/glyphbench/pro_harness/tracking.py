"""Optional W&B tracking for standalone normal/Pro long-horizon runs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


class WandbHarnessTracker:
    """Log per-episode science metrics, native videos, and replay artifacts."""

    def __init__(self, cfg: Any, run_dir: str | None) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir) if run_dir else None
        self.run: Any | None = None
        self._wandb: Any | None = None
        self._episode_rows: list[dict[str, Any]] = []
        if not cfg.wandb_project:
            return

        import wandb

        public_cfg = cfg.public_dict()
        run_name = cfg.wandb_name or "-".join(
            filter(
                None,
                [
                    _slug(cfg.model or "model"),
                    cfg.harness_variant,
                    cfg.reasoning_effort or "default",
                    cfg.reasoning_mode or "standard",
                ],
            )
        )
        self._wandb = wandb
        suite = cfg.task_id.rsplit("/", 1)[-1].split("-", 1)[0]
        self.run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            id=cfg.wandb_run_id,
            resume="allow" if cfg.wandb_run_id else None,
            name=run_name,
            group=cfg.wandb_group,
            job_type=f"{suite}-eval",
            config=public_cfg,
            tags=[
                "glyphbench",
                suite,
                cfg.harness_variant,
                cfg.model or "unknown-model",
                f"effort-{cfg.reasoning_effort or 'default'}",
                f"reasoning-mode-{cfg.reasoning_mode or 'standard'}",
                f"observation-{cfg.observation_mode}",
                "uncapped-output" if cfg.max_output_tokens is None else f"output-cap-{cfg.max_output_tokens}",
                "native-horizon" if cfg.max_turns is None else f"turn-cap-{cfg.max_turns}",
            ],
        )

    def log_episode(self, result: Any) -> None:
        if self.run is None or self._wandb is None:
            return
        row = asdict(result)
        self._episode_rows.append(row)
        metrics: dict[str, Any] = {
            "episode/index": result.episode,
            "episode/seed": result.seed,
            "episode/return": result.episode_return,
            "episode/raw_score": result.raw_score,
            "episode/achievements": result.achievements,
            "episode/steps": result.steps,
            "episode/terminated": int(result.terminated),
            "episode/truncated": int(result.truncated),
            "episode/forfeits": result.forfeits,
            "episode/parse_failures": result.parse_failures,
            "episode/content_filter_blocks": result.content_filter_blocks,
            "episode/softened_calls": result.softened_calls,
            "episode/input_tokens": result.total_input_tokens,
            "episode/cached_input_tokens": result.total_cached_input_tokens,
            "episode/output_tokens": result.total_output_tokens,
            "episode/reasoning_tokens": result.total_reasoning_tokens,
            "episode/llm_attempts": result.total_llm_attempts,
            "episode/transient_retries": result.total_transient_retries,
            "episode/content_filter_retries": result.total_content_filter_retries,
            "episode/unsupported_param_retries": result.total_unsupported_param_retries,
            "episode/blocked_movement_attempts": result.blocked_movement_attempts,
            "episode/llm_seconds": result.llm_seconds,
            "episode/action_turns": result.num_action_turns,
            "episode/memory_turns": result.num_memory_turns,
            "episode/memory_updates_skipped": result.memory_updates_skipped,
            "episode/image_inputs": result.num_image_inputs,
            "episode/error": int(result.error is not None),
        }
        for achievement in result.achievement_names:
            metrics[f"achievement/{_slug(achievement)}"] = 1
        if self.run_dir is not None:
            video = self.run_dir / f"episode_{result.episode}.mp4"
            if video.exists() and video.stat().st_size > 0:
                metrics[f"video/episode_{result.episode}"] = self._wandb.Video(
                    str(video), fps=self.cfg.video_fps, format="mp4"
                )
        self.run.log(metrics, step=result.episode)

    def finish(self, summary: dict[str, Any]) -> None:
        if self.run is None or self._wandb is None:
            return
        prefixed = {f"eval/{key}": value for key, value in summary.items()}
        self.run.summary.update(prefixed)
        if self._episode_rows:
            columns = list(self._episode_rows[0])
            table = self._wandb.Table(
                columns=columns,
                data=[
                    [
                        json.dumps(row[key]) if isinstance(row[key], (list, dict)) else row[key]
                        for key in columns
                    ]
                    for row in self._episode_rows
                ],
            )
            self.run.log({"episodes": table})
        if self.run_dir is not None and self.run_dir.exists():
            suite = self.cfg.task_id.rsplit("/", 1)[-1].split("-", 1)[0]
            artifact = self._wandb.Artifact(
                name=f"{suite}-{_slug(self.run.id)}",
                type=f"glyphbench-{suite}-run",
                metadata={
                    "model": self.cfg.model,
                    "harness_variant": self.cfg.harness_variant,
                    "reasoning_effort": self.cfg.reasoning_effort,
                    "reasoning_mode": self.cfg.reasoning_mode,
                    "observation_mode": self.cfg.observation_mode,
                    "image_detail": self.cfg.image_detail,
                    "episodes": len(self._episode_rows),
                },
            )
            artifact.add_dir(str(self.run_dir))
            self.run.log_artifact(artifact)
        self.run.finish()
        self.run = None

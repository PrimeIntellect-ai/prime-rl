"""The Pro harness rollout loop.

Drives a GlyphBench ``BaseGlyphEnv`` with a grounded action/memory protocol:

  ACTION TURN  -> parse (with a focused re-ask on failure) -> env.step
              -> MEMORY TURN when due -> apply grounded state updates -> repeat

It owns all robustness: content-filter-resilient calls (via the client),
parse-failure recovery + a feedback channel back to the model, graceful
degradation on a blocked/empty response, and zero reasoning-budget
truncation. Per-episode metrics and full interaction logs are written to disk
after every episode so long runs are never lost.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.registry import make_env
from glyphbench.pro_harness.adapters import EnvAdapter, get_adapter
from glyphbench.pro_harness.clients import LLMClient, build_client
from glyphbench.pro_harness.config import SUPPORTED_TASK_IDS, ProConfig
from glyphbench.pro_harness.media import PixelVideoRecorder, frame_to_png_data_url
from glyphbench.pro_harness.memory import Scratchpad, SpatialMemory, apply_memory
from glyphbench.pro_harness.navigation import ExplorationTracker
from glyphbench.pro_harness.parser import parse_action
from glyphbench.pro_harness.prompting import (
    build_system_prompt as build_pro_system_prompt,
)
from glyphbench.pro_harness.prompting import (
    render_action_user,
    render_memory_user,
)
from glyphbench.pro_harness.tracking import WandbHarnessTracker
from glyphbench.protocol import (
    build_system_prompt as build_normal_system_prompt,
)
from glyphbench.protocol import (
    parse_action_response,
)
from glyphbench.protocol import (
    render_user_turn as render_normal_user_turn,
)


def _render_memory_state(scratchpad: Scratchpad, spatial: SpatialMemory) -> str:
    """Compact scratchpad + landmark snapshot for the replay memory panel."""
    return f"Scratchpad:\n{scratchpad.render()}\n\nLandmarks:\n{spatial.render()}"


def _coerce_name_list(raw: Any) -> list[str]:
    """Coerce an ``achievements_unlocked`` field (list / tuple / numpy array /
    JSON-ish string like ``"['collect_wood']"`` / ``"[]"``) to a name list."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s in {"[]", "()"}:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x) for x in v]
        except (json.JSONDecodeError, ValueError):
            pass
        return [p.strip().strip("'\"") for p in s.strip("[]()").split(",") if p.strip()]
    # numpy array or other iterable.
    try:
        return [str(x) for x in raw]
    except TypeError:
        return []


def _unlocked_achievements(info: dict[str, Any]) -> set[str]:
    """Achievements unlocked so far, robust across env variants.

    The pure-Python Craftax env exposes per-name ``Achievements/<name>`` flags;
    the JAX fork exposes an immediate ``achievements_unlocked`` list (its
    per-name flags lag by a step). We union both so the count is correct and
    not delayed regardless of which env backs the task."""
    out: set[str] = set()
    for key, val in info.items():
        if key.startswith("Achievements/") and val:
            out.add(key.split("/", 1)[1])
    out.update(_coerce_name_list(info.get("achievements_unlocked")))
    return out


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    steps: int
    # Native cumulative reward returned by the environment. CraftaxFull uses
    # upstream weighted achievement + health rewards and is not normalized.
    episode_return: float
    # RAW, un-normalized env score (Craftax: achievement count; NetHack: game
    # score). This is the headline metric — never normalized.
    raw_score: float | None
    raw_score_label: str
    terminated: bool
    truncated: bool
    achievements: int
    achievement_names: list[str]
    forfeits: int
    parse_failures: int
    parse_recoveries: int
    action_truncations: int
    memory_parse_failures: int
    content_filter_blocks: int
    softened_calls: int
    total_input_tokens: int
    total_cached_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int
    total_llm_attempts: int
    llm_seconds: float
    num_action_turns: int
    num_memory_turns: int
    memory_updates_skipped: int = 0
    total_transient_retries: int = 0
    total_content_filter_retries: int = 0
    total_unsupported_param_retries: int = 0
    blocked_movement_attempts: int = 0
    num_image_inputs: int = 0
    # Set only when the episode aborted on an unexpected (e.g. env-internal)
    # error; the rest of the sweep continues. None on a normal episode.
    error: str | None = None


class ProHarness:
    def __init__(self, cfg: ProConfig, client: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.client = client or build_client(cfg)
        self._run_dir: str | None = None

    # ---------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        prefix = f"[{cfg.harness_variant}]"
        if cfg.task_id not in SUPPORTED_TASK_IDS and cfg.verbose:
            print(
                f"{prefix} WARNING: task {cfg.task_id!r} is outside the supported "
                f"set {sorted(SUPPORTED_TASK_IDS)}; running anyway."
            )
        self._run_dir = self._make_run_dir()
        adapter = get_adapter(cfg.task_id)
        if cfg.verbose:
            print(f"{prefix} {cfg.describe()}")
            print(f"{prefix} client: {self.client.describe()}")
            if self._run_dir:
                print(f"{prefix} output: {self._run_dir}")

        results, all_interactions = self._load_checkpointed_results()
        if cfg.verbose and results:
            print(f"{prefix} resuming after {len(results)}/{cfg.num_episodes} completed episodes")
        tracker = WandbHarnessTracker(cfg, self._run_dir)
        try:
            result_by_episode = {result.episode: result for result in results}
            interaction_by_episode = {
                int(row["episode"]): row for row in all_interactions
            }

            def run_one(ep: int):
                seed = cfg.seed + ep
                game = self._make_game()
                try:
                    res, interactions = self.run_episode(game, adapter, seed, ep)
                except Exception as exc:  # noqa: BLE001
                    # The LLM client is no-raise by contract; this guards against a
                    # third-party env backend (Craftax/JAX, NLE) crashing internally.
                    # Isolate it to this one episode so the rest of the sweep — and
                    # the already-saved episodes — survive.
                    import traceback

                    if cfg.verbose:
                        print(f"{prefix} episode {ep} aborted: {type(exc).__name__}: {exc}")
                        traceback.print_exc()
                    res = self._failed_episode(ep, seed, f"{type(exc).__name__}: {exc}")
                    interactions = []
                finally:
                    with suppress(Exception):
                        game.close()
                return ep, seed, res, interactions

            def commit_episode(outcome) -> None:
                nonlocal results, all_interactions
                ep, seed, res, interactions = outcome
                result_by_episode[ep] = res
                interaction_by_episode[ep] = {
                    "episode": ep,
                    "seed": seed,
                    "interactions": interactions,
                }
                results = [result_by_episode[i] for i in sorted(result_by_episode)]
                all_interactions = [
                    interaction_by_episode[i] for i in sorted(interaction_by_episode)
                ]
                if cfg.verbose:
                    rs = "n/a" if res.raw_score is None else f"{res.raw_score:g}"
                    print(
                        f"{prefix} episode {ep + 1}/{cfg.num_episodes} done: "
                        f"raw_{res.raw_score_label}={rs} steps={res.steps} "
                        f"ach={res.achievements} return={res.episode_return:+.4f} "
                        f"forfeits={res.forfeits} blocks={res.content_filter_blocks}"
                    )
                self._save(results, all_interactions)

            pending = [
                ep for ep in range(cfg.num_episodes) if ep not in result_by_episode
            ]
            next_log_position = 0

            def log_completed_in_seed_order() -> None:
                nonlocal next_log_position
                while next_log_position < len(pending):
                    episode = pending[next_log_position]
                    result = result_by_episode.get(episode)
                    if result is None:
                        return
                    tracker.log_episode(result)
                    next_log_position += 1

            workers = min(cfg.max_concurrent_episodes, len(pending))
            if workers <= 1:
                for ep in pending:
                    commit_episode(run_one(ep))
                    log_completed_in_seed_order()
            else:
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="glyphbench-episode",
                ) as pool:
                    futures = [pool.submit(run_one, ep) for ep in pending]
                    for future in as_completed(futures):
                        commit_episode(future.result())
                        # W&B requires monotonically increasing explicit
                        # steps, even when higher-numbered seeds finish first.
                        log_completed_in_seed_order()
            summary = self._summarize(results)
            if cfg.verbose:
                print(f"{prefix} SUMMARY: {json.dumps(summary, indent=2)}")
            tracker.finish(summary)
            return {"summary": summary, "episodes": [asdict(r) for r in results]}
        except Exception:
            # Ensure a W&B run is not left live if local serialization fails.
            tracker.finish(self._summarize(results))
            raise

    # ---------------------------------------------------------------------
    def run_episode(
        self, game: BaseGlyphEnv, adapter: EnvAdapter, seed: int, episode_idx: int
    ) -> tuple[EpisodeResult, list[dict[str, Any]]]:
        cfg = self.cfg
        pro_mode = cfg.harness_variant == "pro"
        pixel_mode = cfg.observation_mode == "pixels"
        spec = game.action_spec
        synonyms = adapter.synonyms()
        system_prompt = (
            build_pro_system_prompt(
                game,
                adapter,
                observation_mode=cfg.observation_mode,
            )
            if pro_mode
            else build_normal_system_prompt(
                game,
                None,
                use_memory=False,
                forfeit_mode=cfg.forfeit_mode,
                enable_reasoning=True,
            )
        )

        raw_obs, _info = game.reset(seed)
        if cfg.observation_mode == "native_text":
            obs_text = adapter.native_text_observation(game, raw_obs)
        else:
            obs_text = adapter.observation(game, raw_obs) if pro_mode else raw_obs
        current_pixel_frame = (
            self._require_pixel_frame(adapter, game) if pixel_mode else None
        )
        scratchpad = Scratchpad(adapter.initial_scratchpad())
        spatial = SpatialMemory()
        exploration = ExplorationTracker()
        recent: deque[tuple[int, str, float]] = deque(maxlen=cfg.recent_actions_window)
        normal_frames: deque[tuple[str, str, float]] = deque(maxlen=cfg.recent_actions_window)
        # Match the public verifier's linear conversation aggregation. Pro mode
        # deliberately replaces this unbounded history with compact state.
        normal_conversation: list[dict[str, str]] = []
        pending_lookups = ""
        feedback = ""
        unlocked: set[str] = set()

        episode_return = 0.0
        raw_score: float | None = None
        forfeits = parse_failures = parse_recoveries = 0
        action_truncations = memory_parse_failures = 0
        content_filter_blocks = softened_calls = 0
        in_tok = cached_in_tok = out_tok = reason_tok = total_attempts = 0
        llm_seconds = 0.0
        num_action_turns = num_memory_turns = 0
        memory_updates_skipped = 0
        transient_retries = filter_retries = unsupported_retries = 0
        blocked_movement_attempts = 0
        num_image_inputs = 0
        interactions: list[dict[str, Any]] = []
        transcript: list[dict[str, Any]] = []

        max_turns = cfg.max_turns if cfg.max_turns is not None else game.max_turns
        terminated = truncated = False

        video = (
            PixelVideoRecorder(
                Path(self._run_dir) / f"episode_{episode_idx}.mp4",
                fps=cfg.video_fps,
                max_frames=cfg.video_max_frames,
            )
            if cfg.save_video and self._run_dir is not None
            else None
        )

        def add_call(res) -> None:
            nonlocal llm_seconds, in_tok, cached_in_tok, out_tok, reason_tok
            nonlocal total_attempts, softened_calls, content_filter_blocks
            nonlocal transient_retries, filter_retries, unsupported_retries
            llm_seconds += res.latency_s
            in_tok += res.input_tokens
            cached_in_tok += res.cached_input_tokens
            out_tok += res.output_tokens
            reason_tok += res.reasoning_tokens
            total_attempts += max(1, res.attempts)
            transient_retries += res.transient_retries
            filter_retries += res.content_filter_retries
            unsupported_retries += res.unsupported_param_retries
            if res.softened:
                softened_calls += 1
            if res.blocked:
                content_filter_blocks += 1

        def parse_reply(text: str):
            if pro_mode:
                return parse_action(text, spec, noop=game.noop_action_name, synonyms=synonyms)
            parsed = parse_action_response(text, spec, noop=game.noop_action_name)
            return parsed.index, parsed.name, parsed.failed, parsed.failure_reason

        def pixel_request(text: str, frame: Any) -> list[dict[str, Any]]:
            nonlocal num_image_inputs
            num_image_inputs += 1
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                        {
                            "type": "input_image",
                            "image_url": frame_to_png_data_url(frame),
                            "detail": cfg.image_detail,
                        },
                    ],
                }
            ]

        self._record_video_frame(video, adapter, game)
        if pro_mode and cfg.show_exploration_aid:
            exploration.observe(adapter.area_key(game, obs_text), obs_text)
        try:
            for _ in range(int(max_turns)):
                area_label = adapter.area_label(game, obs_text)
                # ---------------- ACTION TURN ----------------
                if pro_mode:
                    action_user = render_action_user(
                        adapter=adapter,
                        game=game,
                        obs_text=obs_text,
                        turn=game.turn,
                        recent_actions=recent,
                        scratchpad_text=scratchpad.render(),
                        landmarks_text=spatial.render(),
                        pending_lookups=pending_lookups,
                        feedback=feedback,
                        exploration_text=(
                            exploration.render(adapter.area_key(game, obs_text))
                            if cfg.show_exploration_aid
                            else ""
                        ),
                        show_focus=cfg.show_floor_focus,
                        include_text_observation=not pixel_mode,
                    )
                else:
                    action_user = render_normal_user_turn(
                        game,
                        frames=normal_frames,
                        current_obs=obs_text,
                        turn=game.turn,
                        max_output_tokens=None,
                        enable_reasoning=True,
                    )
                    if feedback:
                        action_user = f"[System Feedback]\n{feedback}\n\n{action_user}"
                pending_lookups = ""
                feedback = ""
                request_input: str | list[dict[str, Any]] = action_user
                if pixel_mode:
                    request_input = pixel_request(action_user, current_pixel_frame)
                elif not pro_mode:
                    request_input = [
                        *normal_conversation,
                        {"role": "user", "content": action_user},
                    ]
                res = self.client.complete(system_prompt, request_input, label="action")
                num_action_turns += 1
                add_call(res)
                if not pro_mode:
                    normal_conversation.append({"role": "user", "content": action_user})
                    if res.text:
                        normal_conversation.append({"role": "assistant", "content": res.text})
                if res.truncated:
                    action_truncations += 1

                idx, name, pf, reason = parse_reply(res.text)

                # Focused re-ask on parse failure (cheap, strongly constrained).
                recovered = False
                if pf and res.text and cfg.parse_retries > 0:
                    for _r in range(cfg.parse_retries):
                        retry_user = self._focused_action_user(spec)
                        retry_input: str | list[dict[str, Any]] = retry_user
                        if pixel_mode:
                            retry_input = pixel_request(retry_user, current_pixel_frame)
                        elif not pro_mode:
                            retry_input = [
                                *normal_conversation,
                                {"role": "user", "content": retry_user},
                            ]
                        rres = self.client.complete(
                            system_prompt, retry_input, label="action_retry"
                        )
                        add_call(rres)
                        if not pro_mode:
                            normal_conversation.append({"role": "user", "content": retry_user})
                            if rres.text:
                                normal_conversation.append(
                                    {"role": "assistant", "content": rres.text}
                                )
                        ridx, rname, rpf, _rreason = parse_reply(rres.text)
                        if not rpf:
                            idx, name, pf, reason = ridx, rname, False, None
                            recovered = True
                            parse_recoveries += 1
                            break

                if pf:
                    parse_failures += 1
                    forfeits += 1

                # ---------------- ENV STEP ----------------
                if pf and cfg.forfeit_mode == "freeze":
                    next_raw, reward, term, trunc, info = game.forfeit_turn()
                    applied_name = "FORFEIT"
                else:
                    next_raw, reward, term, trunc, info = game.step(idx)
                    applied_name = "FORFEIT" if pf else name
                if cfg.observation_mode == "native_text":
                    next_obs = adapter.native_text_observation(game, next_raw)
                else:
                    next_obs = adapter.observation(game, next_raw) if pro_mode else next_raw
                next_pixel_frame = (
                    self._require_pixel_frame(adapter, game) if pixel_mode else None
                )
                movement_feedback = self._blocked_movement_feedback(
                    applied_name, obs_text, next_obs
                )
                if movement_feedback:
                    blocked_movement_attempts += 1
                if pro_mode and cfg.show_exploration_aid:
                    exploration.observe(adapter.area_key(game, next_obs), next_obs)

                episode_return += float(reward)
                self._record_video_frame(video, adapter, game)
                new_unlocked = sorted(_unlocked_achievements(info) - unlocked)
                unlocked |= set(new_unlocked)
                sc = adapter.raw_score(info)
                if sc is not None:
                    raw_score = sc if raw_score is None else max(raw_score, sc)
                recent.append((game.turn, applied_name, float(reward)))
                normal_frames.append((obs_text, applied_name, float(reward)))

                # Replay transcript: the observation the agent acted on + outcome.
                # Full output and reasoning are retained for later analysis.
                transcript.append(
                    {
                        "turn": game.turn,
                        "area": area_label,
                        "observation": obs_text,
                        "model_observation_mode": cfg.observation_mode,
                        "pixel_image_detail": cfg.image_detail if pixel_mode else None,
                        "pixel_frame_index": max(0, game.turn - 1) if pixel_mode else None,
                        "action": applied_name,
                        "reward": float(reward),
                        "total_return": round(episode_return, 4),
                        "raw_score": raw_score,
                        "raw_score_label": adapter.raw_score_label(),
                        "achievements": len(unlocked),
                        "new_achievements": new_unlocked,
                        "terminated": bool(term),
                        "truncated": bool(trunc),
                        "movement_blocked": bool(movement_feedback),
                        "reasoning": res.reasoning or "",
                        "output": res.text or "",
                        "scratchpad": scratchpad.to_dict() if pro_mode else {},
                        "memory_before": (
                            _render_memory_state(scratchpad, spatial) if pro_mode else None
                        ),
                        "memory_after": None,
                        "memory_output": None,
                        "pending_lookups": "",
                    }
                )

                interactions.append(
                    {
                        "step": game.turn,
                        "role": "action",
                        "area": area_label,
                        "action": applied_name,
                        "action_idx": int(idx),
                        "parse_failed": bool(pf),
                        "parse_reason": reason,
                        "parse_recovered": recovered,
                        "reward": float(reward),
                        "total_return": round(episode_return, 4),
                        "achievements": len(unlocked),
                        "new_achievements": new_unlocked,
                        "finish_reason": res.finish_reason,
                        "softened": res.softened,
                        "blocked": res.blocked,
                        "truncated": res.truncated,
                        "input_tokens": res.input_tokens,
                        "cached_input_tokens": res.cached_input_tokens,
                        "output_tokens": res.output_tokens,
                        "reasoning_tokens": res.reasoning_tokens,
                        "latency_s": round(res.latency_s, 2),
                        "reasoning": res.reasoning or "",
                        "output": res.text or "",
                        "error": res.error,
                        "transient_retries": res.transient_retries,
                        "content_filter_retries": res.content_filter_retries,
                        "unsupported_param_retries": res.unsupported_param_retries,
                        "movement_blocked": bool(movement_feedback),
                        "model_observation_mode": cfg.observation_mode,
                        "pixel_image_detail": cfg.image_detail if pixel_mode else None,
                        "pixel_frame_index": max(0, game.turn - 1) if pixel_mode else None,
                    }
                )

                if self.cfg.render_terminal:
                    self._render(
                        game,
                        episode_idx,
                        applied_name,
                        reward,
                        episode_return,
                        len(unlocked),
                        res,
                        raw_score,
                        adapter.raw_score_label(),
                    )

                terminated, truncated = bool(term), bool(trunc)
                if terminated or truncated:
                    break

                # Build the feedback string for the *next* action turn.
                feedback = self._action_feedback(pf and not recovered, res)
                if movement_feedback:
                    feedback = "\n".join(filter(None, [feedback, movement_feedback]))

                if not pro_mode:
                    obs_text = next_obs
                    if cfg.checkpoint_every and game.turn % cfg.checkpoint_every == 0:
                        self._checkpoint_episode(
                            episode_idx,
                            transcript,
                            self._progress(
                                game,
                                adapter,
                                raw_score,
                                unlocked,
                                episode_return,
                                forfeits,
                                action_truncations,
                                memory_parse_failures,
                                content_filter_blocks,
                                softened_calls,
                                in_tok,
                                cached_in_tok,
                                out_tok,
                                reason_tok,
                                total_attempts,
                                memory_updates_skipped,
                                transient_retries,
                                filter_retries,
                                unsupported_retries,
                                blocked_movement_attempts,
                            ),
                        )
                    continue

                next_area_label = adapter.area_label(game, next_obs)
                memory_due = (
                    cfg.memory_update_every == 1
                    or game.turn % cfg.memory_update_every == 0
                    or bool(new_unlocked)
                    or float(reward) != 0.0
                    or next_area_label != area_label
                    or bool(pf)
                    or bool(res.truncated)
                )
                if not memory_due:
                    memory_updates_skipped += 1
                    obs_text = next_obs
                    current_pixel_frame = next_pixel_frame
                    if cfg.checkpoint_every and game.turn % cfg.checkpoint_every == 0:
                        self._checkpoint_episode(
                            episode_idx,
                            transcript,
                            self._progress(
                                game, adapter, raw_score, unlocked, episode_return,
                                forfeits, action_truncations, memory_parse_failures,
                                content_filter_blocks, softened_calls, in_tok,
                                cached_in_tok, out_tok, reason_tok, total_attempts,
                                memory_updates_skipped,
                                transient_retries,
                                filter_retries,
                                unsupported_retries,
                                blocked_movement_attempts,
                            ),
                        )
                    continue

                # ---------------- MEMORY TURN ----------------
                memory_user = render_memory_user(
                    action_reasoning=res.reasoning or res.text,
                    action_name=applied_name,
                    parse_failed=bool(pf),
                    parse_reason=reason,
                    forfeit_mode=cfg.forfeit_mode,
                    action_truncated=res.truncated,
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    new_achievements=new_unlocked,
                    next_obs=next_obs,
                    scratchpad_text=scratchpad.render(),
                    landmarks_text=spatial.render(),
                    exploration_text=(
                        exploration.render(adapter.area_key(game, next_obs))
                        if cfg.show_exploration_aid
                        else ""
                    ),
                    include_text_observation=not pixel_mode,
                )
                memory_input: str | list[dict[str, Any]] = memory_user
                if pixel_mode:
                    memory_input = pixel_request(memory_user, next_pixel_frame)
                mres = self.client.complete(system_prompt, memory_input, label="memory")
                num_memory_turns += 1
                add_call(mres)

                upd = apply_memory(mres.text, scratchpad, spatial)
                pending_lookups = upd.query_results
                if upd.parse_failed:
                    memory_parse_failures += 1

                # Backfill this turn's transcript row with the memory-turn result
                # so `gb replay` can show the scratchpad/landmark memory panel.
                if transcript:
                    transcript[-1]["memory_after"] = _render_memory_state(scratchpad, spatial)
                    transcript[-1]["memory_output"] = mres.text or ""
                    transcript[-1]["pending_lookups"] = pending_lookups

                interactions.append(
                    {
                        "step": game.turn,
                        "role": "memory",
                        "memory_parse_failed": upd.parse_failed,
                        "memory_updated": upd.updated,
                        "landmark_count": spatial.count(),
                        "finish_reason": mres.finish_reason,
                        "softened": mres.softened,
                        "blocked": mres.blocked,
                        "truncated": mres.truncated,
                        "input_tokens": mres.input_tokens,
                        "cached_input_tokens": mres.cached_input_tokens,
                        "output_tokens": mres.output_tokens,
                        "reasoning_tokens": mres.reasoning_tokens,
                        "latency_s": round(mres.latency_s, 2),
                        "scratchpad": scratchpad.to_dict(),
                        "reasoning": mres.reasoning or "",
                        "output": mres.text or "",
                        "error": mres.error,
                        "transient_retries": mres.transient_retries,
                        "content_filter_retries": mres.content_filter_retries,
                        "unsupported_param_retries": mres.unsupported_param_retries,
                    }
                )

                obs_text = next_obs
                current_pixel_frame = next_pixel_frame

                # Periodic checkpoint so a long uncapped run isn't lost on a kill.
                if cfg.checkpoint_every and game.turn % cfg.checkpoint_every == 0:
                    self._checkpoint_episode(
                        episode_idx,
                        transcript,
                        self._progress(
                            game,
                            adapter,
                            raw_score,
                            unlocked,
                            episode_return,
                            forfeits,
                            action_truncations,
                            memory_parse_failures,
                            content_filter_blocks,
                            softened_calls,
                            in_tok,
                            cached_in_tok,
                            out_tok,
                            reason_tok,
                            total_attempts,
                            memory_updates_skipped,
                            transient_retries,
                            filter_retries,
                            unsupported_retries,
                            blocked_movement_attempts,
                        ),
                    )
        finally:
            if video is not None:
                video.close()
                if cfg.verbose:
                    if video.error:
                        print(f"[{cfg.harness_variant}] video recording failed: {video.error}")
                    elif video.exists:
                        print(
                            f"[{cfg.harness_variant}] saved native pixel video "
                            f"({video.frames} frames): "
                            f"{video.path}"
                        )

        self._write_transcript(transcript, episode_idx)

        result = EpisodeResult(
            episode=episode_idx,
            seed=seed,
            steps=game.turn,
            episode_return=round(episode_return, 4),
            raw_score=raw_score,
            raw_score_label=adapter.raw_score_label(),
            terminated=terminated,
            truncated=truncated,
            achievements=len(unlocked),
            achievement_names=sorted(unlocked),
            forfeits=forfeits,
            parse_failures=parse_failures,
            parse_recoveries=parse_recoveries,
            action_truncations=action_truncations,
            memory_parse_failures=memory_parse_failures,
            content_filter_blocks=content_filter_blocks,
            softened_calls=softened_calls,
            total_input_tokens=in_tok,
            total_cached_input_tokens=cached_in_tok,
            total_output_tokens=out_tok,
            total_reasoning_tokens=reason_tok,
            total_llm_attempts=total_attempts,
            llm_seconds=round(llm_seconds, 2),
            num_action_turns=num_action_turns,
            num_memory_turns=num_memory_turns,
            memory_updates_skipped=memory_updates_skipped,
            total_transient_retries=transient_retries,
            total_content_filter_retries=filter_retries,
            total_unsupported_param_retries=unsupported_retries,
            blocked_movement_attempts=blocked_movement_attempts,
            num_image_inputs=num_image_inputs,
        )
        return result, interactions

    # ---------------------------------------------------------------------
    @staticmethod
    def _require_pixel_frame(adapter: EnvAdapter, game: BaseGlyphEnv) -> Any:
        """Return the native frame or fail closed instead of leaking text.

        A pixel-modality eval must never silently fall back to the ASCII
        observation, because that would invalidate the modality ablation.
        """
        frame = adapter.render_frame(game)
        if frame is None:
            raise RuntimeError(
                "pixel observation requested, but the environment did not "
                "produce a native render frame"
            )
        return frame

    # ---------------------------------------------------------------------
    @staticmethod
    def _record_video_frame(video, adapter, game) -> None:
        if video is None:
            return
        with suppress(Exception):
            frame = adapter.render_frame(game)
            if frame is not None:
                video.append(frame)

    def _write_transcript(
        self, transcript: list, episode_idx: int, *, announce: bool = True
    ) -> None:
        """Write one JSON line per turn (observation + action + outcome) so the
        episode can be replayed in the terminal via ``gb replay <run-dir>``."""
        if not transcript or not self._run_dir:
            return
        path = os.path.join(self._run_dir, f"transcript_{episode_idx}.jsonl")
        with open(path, "w") as fh:
            for row in transcript:
                fh.write(json.dumps(row) + "\n")
        if announce and self.cfg.verbose:
            print(
                f"[{self.cfg.harness_variant}] saved transcript ({len(transcript)} turns): {path}"
            )

    def _checkpoint_episode(self, episode_idx: int, transcript: list, progress: dict) -> None:
        """Mid-episode flush so a long uncapped run survives a kill / 24h cap.

        Writes the full current transcript + a small progress snapshot. The
        native MP4 recorder streams independently and closes at episode end.
        """
        if not self._run_dir:
            return
        self._write_transcript(transcript, episode_idx, announce=False)
        with (
            open(os.path.join(self._run_dir, f"progress_{episode_idx}.json"), "w") as fh,
        ):
            json.dump(progress, fh, indent=2)
        if self.cfg.verbose:
            print(
                f"[{self.cfg.harness_variant}] checkpoint @ turn "
                f"{progress.get('steps')}: "
                f"raw_{progress.get('raw_score_label')}={progress.get('raw_score')} "
                f"({len(transcript)} turns saved)"
            )

    @staticmethod
    def _progress(
        game: BaseGlyphEnv,
        adapter: EnvAdapter,
        raw_score: float | None,
        unlocked: set[str],
        episode_return: float,
        forfeits: int,
        action_truncations: int,
        memory_parse_failures: int,
        content_filter_blocks: int,
        softened_calls: int,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        llm_attempts: int,
        memory_updates_skipped: int = 0,
        transient_retries: int = 0,
        content_filter_retries: int = 0,
        unsupported_param_retries: int = 0,
        blocked_movement_attempts: int = 0,
    ) -> dict[str, Any]:
        return {
            "in_progress": True,
            "steps": game.turn,
            "episode_return": round(episode_return, 4),
            "raw_score": raw_score,
            "raw_score_label": adapter.raw_score_label(),
            "achievements": len(unlocked),
            "achievement_names": sorted(unlocked),
            "forfeits": forfeits,
            "action_truncations": action_truncations,
            "memory_parse_failures": memory_parse_failures,
            "content_filter_blocks": content_filter_blocks,
            "softened_calls": softened_calls,
            "total_input_tokens": input_tokens,
            "total_cached_input_tokens": cached_input_tokens,
            "total_output_tokens": output_tokens,
            "total_reasoning_tokens": reasoning_tokens,
            "total_llm_attempts": llm_attempts,
            "memory_updates_skipped": memory_updates_skipped,
            "total_transient_retries": transient_retries,
            "total_content_filter_retries": content_filter_retries,
            "total_unsupported_param_retries": unsupported_param_retries,
            "blocked_movement_attempts": blocked_movement_attempts,
        }

    def _focused_action_user(self, spec) -> str:
        return (
            "Your previous reply did not contain a parseable action, so the "
            "turn would be forfeited. Reply with ONLY one tag and nothing else:\n"
            "  <action>NAME</action>\n"
            "where NAME is exactly one of: " + ", ".join(spec.names) + "."
        )

    @staticmethod
    def _blocked_movement_feedback(action: str, before: str, after: str) -> str:
        """Ground a failed move using only observations already shown to the model.

        A first movement command in a new direction can merely turn the
        avatar. It is definitely blocked only when the avatar was already
        facing that direction and the absolute position still did not change.
        """
        directions = {
            "MOVE_UP": "up", "UP": "up",
            "MOVE_DOWN": "down", "DOWN": "down",
            "MOVE_LEFT": "left", "LEFT": "left",
            "MOVE_RIGHT": "right", "RIGHT": "right",
        }
        direction = directions.get(action.upper())
        if direction is None:
            return ""
        pos_re = re.compile(r"Pos:\((-?\d+),\s*(-?\d+)\)")
        facing_re = re.compile(r"Facing:([a-z]+)", re.IGNORECASE)
        before_pos = pos_re.search(before or "")
        after_pos = pos_re.search(after or "")
        before_facing = facing_re.search(before or "")
        if not before_pos or not after_pos or not before_facing:
            return ""
        if before_pos.groups() != after_pos.groups():
            return ""
        if before_facing.group(1).lower() != direction:
            return ""
        position = f"({before_pos.group(1)},{before_pos.group(2)})"
        return (
            f"GROUNDED NAVIGATION FAILURE: {action} while already facing {direction} "
            f"left Pos:{position} unchanged, so that adjacent edge is blocked. "
            f"Do NOT repeat {action} from Pos:{position}; route around it using a "
            "different direction, or interact only if the visible blocking tile is a "
            "resource you intentionally need to mine."
        )

    @staticmethod
    def _action_feedback(unrecovered_parse_fail: bool, res) -> str:
        if unrecovered_parse_fail:
            return (
                "Your previous turn was forfeited because no valid "
                "<action>NAME</action> tag was found. End your reply with "
                "exactly one such tag using a name from the action list."
            )
        if res.blocked:
            return (
                "The previous response was blocked or empty and the turn was "
                "forfeited. Continue playing the game normally and reply with a "
                "valid <action>NAME</action>."
            )
        if res.truncated:
            return (
                "Your previous response was cut off before the <action> tag. "
                "You have ample output budget — make sure to finish with the "
                "<action>NAME</action> tag."
            )
        return ""

    def _render(
        self, game, ep, action, reward, total, ach, res, raw_score=None, score_label="score"
    ) -> None:
        rs = "n/a" if raw_score is None else f"{raw_score:g}"
        print(
            f"[{self.cfg.harness_variant}] ep{ep + 1} t{game.turn} "
            f"act={action} r={reward:+.3f} "
            f"raw_{score_label}={rs} ach={ach} ret={total:+.3f} "
            f"fin={res.finish_reason} out_tok={res.output_tokens}"
        )

    # ---------------------------------------------------------------------
    def _make_game(self) -> BaseGlyphEnv:
        kw: dict[str, Any] = {}
        if self.cfg.max_turns is not None:
            kw["max_turns"] = int(self.cfg.max_turns)
        return make_env(self.cfg.task_id, **kw)

    def _make_run_dir(self) -> str | None:
        if not self.cfg.save_results:
            return None
        base = self.cfg.output_dir or os.path.join("outputs", "pro_harness")
        slug = self.cfg.task_id.split("/")[-1]
        model = (self.cfg.model or "model").replace("/", "_")
        effort = self.cfg.reasoning_effort or "default"
        reasoning_mode = self.cfg.reasoning_mode or "standard"
        name = (
            f"{slug}-{self.cfg.backend}-{model}-{self.cfg.harness_variant}-"
            f"{effort}-{reasoning_mode}"
        )
        if self.cfg.observation_mode != "text":
            name += f"-{self.cfg.observation_mode}-observation"
        if self.cfg.run_tag:
            name += f"-{self.cfg.run_tag}"
        run_dir = os.path.join(base, name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _save(self, results, all_interactions) -> None:
        if not self._run_dir:
            return
        # Commit metrics last: they mark the episodes eligible for resume.
        # A failed write must be visible, and must not truncate an older file.
        for name, value in (
            ("config.json", self.cfg.public_dict()),
            ("interactions.json", all_interactions),
            ("metrics.json", [asdict(r) for r in results]),
        ):
            path = Path(self._run_dir) / name
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
            temporary.replace(path)

    def _load_checkpointed_results(
        self,
    ) -> tuple[list[EpisodeResult], list[dict[str, Any]]]:
        """Resume fully completed seeds for stable, ID-addressed grid runs.

        A partly written current episode is deliberately replayed from its seed;
        only entries already committed to ``metrics.json`` are considered done.
        Automatic resume is restricted to runs with an explicit W&B run id,
        which prevents an incidental same-name local run from being reused.
        """
        if not self._run_dir or not self.cfg.wandb_run_id:
            return [], []
        metrics_path = os.path.join(self._run_dir, "metrics.json")
        interactions_path = os.path.join(self._run_dir, "interactions.json")
        config_path = os.path.join(self._run_dir, "config.json")
        try:
            with open(config_path) as fh:
                saved_config = json.load(fh)
            if not isinstance(saved_config, dict):
                return [], []
            with open(metrics_path) as fh:
                raw_metrics = json.load(fh)
            results = [EpisodeResult(**row) for row in raw_metrics]
            if len(results) > self.cfg.num_episodes:
                return [], []
            with open(interactions_path) as fh:
                interactions = json.load(fh)
            if (
                not isinstance(interactions, list)
                or not all(isinstance(row, dict) for row in interactions)
            ):
                return [], []
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return [], []
        # A shared output name is not enough to identify an experiment: changing
        # seeds, budgets, or sampling settings must not reuse old scores.
        runtime_options = {
            "num_episodes", "max_concurrent_episodes", "output_dir", "verbose",
            "render_terminal", "save_results", "save_video", "video_fps",
            "video_max_frames", "checkpoint_every", "wandb_project", "wandb_entity",
            "wandb_group", "wandb_name", "run_tag",
        }
        changed = [
            key for key, value in self.cfg.public_dict().items()
            if key not in runtime_options and saved_config.get(key) != value
        ]
        if changed:
            raise ValueError(f"checkpoint configuration differs: {', '.join(changed)}")
        episodes = {result.episode for result in results}
        if len(episodes) != len(results) or any(
            result.episode not in range(self.cfg.num_episodes)
            or result.seed != self.cfg.seed + result.episode
            for result in results
        ):
            raise ValueError("checkpoint episode indices or seeds do not match this run")
        # The interactions file is replaced before metrics. An interrupted
        # save can therefore include an episode that was never committed.
        # Keep only the entries acknowledged by metrics, while still requiring
        # exactly one matching interaction record for each completed episode.
        committed_interactions = {}
        for row in interactions:
            episode = row.get("episode")
            if not isinstance(episode, int):
                raise ValueError("checkpoint interaction episode index is invalid")
            if episode not in episodes:
                continue
            if (
                episode in committed_interactions
                or row.get("seed") != self.cfg.seed + episode
                or not isinstance(row.get("interactions"), list)
            ):
                raise ValueError("checkpoint interactions do not match the recorded episodes")
            committed_interactions[episode] = row
        if set(committed_interactions) != episodes:
            raise ValueError("checkpoint interactions do not match the recorded episodes")
        return results, [committed_interactions[result.episode] for result in results]

    @staticmethod
    def _failed_episode(episode_idx: int, seed: int, error: str) -> EpisodeResult:
        return EpisodeResult(
            episode=episode_idx,
            seed=seed,
            steps=0,
            episode_return=0.0,
            raw_score=None,
            raw_score_label="score",
            terminated=False,
            truncated=False,
            achievements=0,
            achievement_names=[],
            forfeits=0,
            parse_failures=0,
            parse_recoveries=0,
            action_truncations=0,
            memory_parse_failures=0,
            content_filter_blocks=0,
            softened_calls=0,
            total_input_tokens=0,
            total_cached_input_tokens=0,
            total_output_tokens=0,
            total_reasoning_tokens=0,
            total_llm_attempts=0,
            llm_seconds=0.0,
            num_action_turns=0,
            num_memory_turns=0,
            memory_updates_skipped=0,
            total_transient_retries=0,
            total_content_filter_retries=0,
            total_unsupported_param_retries=0,
            blocked_movement_attempts=0,
            error=error,
        )

    @staticmethod
    def _summarize(results: list[EpisodeResult]) -> dict[str, Any]:
        if not results:
            return {}
        n = len(results)
        rets = [r.episode_return for r in results]
        achs = [r.achievements for r in results]
        raws = [r.raw_score for r in results if r.raw_score is not None]
        score_label = next((r.raw_score_label for r in results), "score")

        def mean(xs):
            return sum(xs) / len(xs) if xs else 0.0

        return {
            "episodes": n,
            "errored_episodes": sum(1 for r in results if r.error),
            # RAW (un-normalized) score is the headline metric.
            f"mean_raw_{score_label}": round(mean(raws), 3) if raws else None,
            f"max_raw_{score_label}": round(max(raws), 3) if raws else None,
            "raw_scores": raws,
            "mean_episode_return": round(mean(rets), 4),
            "max_episode_return": round(max(rets), 4),
            "episode_returns": rets,
            "mean_achievements": round(mean(achs), 2),
            "max_achievements": max(achs),
            "mean_steps": round(mean([r.steps for r in results]), 1),
            "total_forfeits": sum(r.forfeits for r in results),
            "total_parse_recoveries": sum(r.parse_recoveries for r in results),
            "total_content_filter_blocks": sum(r.content_filter_blocks for r in results),
            "total_softened_calls": sum(r.softened_calls for r in results),
            "total_action_truncations": sum(r.action_truncations for r in results),
            "total_memory_parse_failures": sum(r.memory_parse_failures for r in results),
            "total_memory_updates_skipped": sum(r.memory_updates_skipped for r in results),
            "total_input_tokens": sum(r.total_input_tokens for r in results),
            "total_cached_input_tokens": sum(r.total_cached_input_tokens for r in results),
            "total_output_tokens": sum(r.total_output_tokens for r in results),
            "total_reasoning_tokens": sum(r.total_reasoning_tokens for r in results),
            "total_llm_attempts": sum(r.total_llm_attempts for r in results),
            "total_transient_retries": sum(r.total_transient_retries for r in results),
            "total_content_filter_retries": sum(
                r.total_content_filter_retries for r in results
            ),
            "total_unsupported_param_retries": sum(
                r.total_unsupported_param_retries for r in results
            ),
            "total_blocked_movement_attempts": sum(
                r.blocked_movement_attempts for r in results
            ),
            "achievement_frequency": {
                name: sum(name in r.achievement_names for r in results)
                for name in sorted({name for r in results for name in r.achievement_names})
            },
        }

"""GlyphbenchMultiTurnEnv + load_environment entry point."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset
from verifiers.types import Response, TrajectoryStep
from verifiers.utils.response_utils import parse_response_message, parse_response_tokens

from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.registry import REGISTRY, default_use_memory, make_env
from glyphbench.core.task_selection import list_task_ids
from glyphbench.craftax_media import render_native_craftax_frame, write_gif
from glyphbench.verifiers_integration.memory import (
    ACTION_STOP,
    action_reasoning_text,
    build_memory_update_user,
    extract_memory_update,
    memory_sampling_args,
    xml_stop_sampling_args,
)
from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
from glyphbench.verifiers_integration.prompting import (
    build_system_prompt,
    render_user_turn,
)
from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric

DEFAULT_MAX_OUTPUT_TOKENS = 4096
# Stateless per turn by default — every observation must be readable off
# the current grid alone. Frame stacking is opt-in via load_environment.
DEFAULT_N_FRAMES = 0
DEFAULT_NUM_EPISODES = 3
DEFAULT_BASE_SEED = 42
AGENTICK_MARKOV_REASONER_HARNESS = "agentick_markov_reasoner"


def _coerce_optional_bool(value: bool | str | None, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "auto", "default", "task"}:
            return None
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise TypeError(
        f"{name} must be bool, None, or one of auto/on/off; got {value!r}"
    )


def _coerce_harness(value: str | None) -> str:
    if value is None:
        return "default"
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"", "auto", "default", "standard", "memory"}:
        return "default"
    if normalized in {
        AGENTICK_MARKOV_REASONER_HARNESS,
        "agentick_reasoner",
        "markovian_reasoner",
        "markov_reasoner",
    }:
        return AGENTICK_MARKOV_REASONER_HARNESS
    raise TypeError(
        "harness must be one of default/agentick_markov_reasoner "
        "(aliases: auto/standard/memory for default; agentick_reasoner/"
        "markovian_reasoner/markov_reasoner for agentick_markov_reasoner); "
        f"got {value!r}"
    )


def load_environment(
    task_id: str | list[str] | None = None,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    n_frames: int = DEFAULT_N_FRAMES,
    max_turns: int | None = None,
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
    seed: int = DEFAULT_BASE_SEED,
    use_memory: bool | None = None,
    memory_update_max_tokens: int | None = 4096,
    stop_after_xml: bool = True,
    forfeit_mode: str = "freeze",
    harness: str | None = None,
    save_gif: bool = False,
    gif_output_dir: str | None = None,
    gif_fps: int = 8,
    gif_max_frames: int = 4000,
    parse_retries: int = 1,
    include_suites: list[str] | None = None,
    exclude_suites: list[str] | None = None,
    include_tasks: list[str] | None = None,
    exclude_tasks: list[str] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Entry point consumed by ``prime eval run`` and ``prime-rl`` orchestrator.

    Args:
        task_id: single glyphbench env id (e.g. ``"glyphbench/__dummy-v0"``),
                a list of ids, or ``None`` for all registered envs (dummy envs
                excluded when id is ``None``). Named ``task_id`` (not
                ``env_id``) because verifiers reserves ``env_id`` for the
                package name passed via ``vf.load_environment``.
        num_episodes: rollouts per env.
        n_frames: history window shown in each user turn.
        max_turns: per-episode turn cap; ``None`` uses each game's own max_turns.
        max_output_tokens: per-turn LLM budget communicated in the system
                prompt. ``None`` imposes no explicit completion cap.
        seed: base seed; each episode uses ``seed + episode_idx`` as the
                per-rollout seed.
        use_memory: ``None`` uses each task's registered default. When true or
                false, overrides all selected tasks. Memory mode uses an action
                generation followed by a memory-update generation per env step.
        memory_update_max_tokens: optional generation limit for the memory
                update call. ``None`` reuses the action sampling limit.
        stop_after_xml: when true, stop action/memory generations immediately
                after the closing protocol tag while keeping that tag in the
                returned text/tokens for strict parsing and training.
        forfeit_mode: parse-failure transition. ``"freeze"`` preserves the
                public eval default: advance only the turn counter, leave env
                state unchanged, reward 0. ``"noop"`` applies the env's
                configured noop/fallback action, so dynamics and reward advance
                as if the agent had chosen that action.
        harness: optional rollout harness. ``"default"`` preserves the current
                GlyphBench memory/action behavior.
        save_gif: save a native-renderer Craftax GIF for each rollout.
        gif_output_dir: directory for GIFs (required when ``save_gif`` is true).
        gif_fps: playback frame rate for saved GIFs.
        gif_max_frames: maximum sampled frames, spread across the full horizon.
        parse_retries: focused same-turn retries after an invalid action reply.
        include_suites: when ``task_id`` is ``None``, only registered envs
                whose first hyphen-segment matches one of these suite names
                pass.
        exclude_suites: removes any env whose suite is in this list. Always
                wins over includes.
        include_tasks: list of exact env IDs OR fnmatch patterns. Either
                this or ``include_suites`` is sufficient for an env to pass.
        exclude_tasks: list of exact env IDs OR fnmatch patterns to drop.
                Always wins over includes.

    ``**kwargs`` is intentional: verifiers' generic loader injects
    ``env_id="<package-name>"``. We absorb it (and reject anything else) so
    callers can either go through ``vf.load_environment`` or call
    ``glyphbench.load_environment`` directly.
    """
    # verifiers injects env_id=<package name>; ignore that one specific kwarg.
    kwargs.pop("env_id", None)
    if kwargs:
        raise TypeError(
            f"load_environment() got unexpected keyword arguments: "
            f"{sorted(kwargs)!r}. Did you mean 'task_id' for the glyphbench "
            f"env id?"
        )
    _ensure_envs_loaded()
    env_ids = _resolve_env_ids(
        task_id, include_suites, exclude_suites, include_tasks, exclude_tasks,
    )
    harness_mode = _coerce_harness(harness)
    dataset = _build_dataset(env_ids, num_episodes, seed)

    parser = GlyphbenchXMLParser()
    rubric = EpisodicReturnRubric(parser=parser)

    return GlyphbenchMultiTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        n_frames=n_frames,
        max_turns_override=max_turns,
        max_output_tokens=max_output_tokens,
        use_memory=_coerce_optional_bool(use_memory, name="use_memory"),
        memory_update_max_tokens=memory_update_max_tokens,
        stop_after_xml=stop_after_xml,
        forfeit_mode=forfeit_mode,
        harness=harness_mode,
        save_gif=save_gif,
        gif_output_dir=gif_output_dir,
        gif_fps=gif_fps,
        gif_max_frames=gif_max_frames,
        parse_retries=parse_retries,
    )


def _ensure_envs_loaded() -> None:
    """Force-import all suite __init__.py files so the registry is populated."""
    from glyphbench.envs import _import_all_suites

    _import_all_suites()


def _resolve_env_ids(
    task_id: str | list[str] | None,
    include_suites: list[str] | None,
    exclude_suites: list[str] | None,
    include_tasks: list[str] | None,
    exclude_tasks: list[str] | None,
) -> list[str]:
    """Resolve the final list of env_ids to operate on.

    If ``task_id`` is given (string or list), filter kwargs must all be
    ``None`` — they are mutually exclusive with the explicit task id.
    Otherwise, the filter kwargs are passed through list_task_ids to filter
    the registry.
    """
    if task_id is not None:
        if any(
            x is not None
            for x in (include_suites, exclude_suites, include_tasks, exclude_tasks)
        ):
            raise TypeError(
                "task_id is mutually exclusive with include_suites/exclude_suites/"
                "include_tasks/exclude_tasks. Pass either an explicit task_id (or list) "
                "OR filter kwargs, not both."
            )
        ids = [task_id] if isinstance(task_id, str) else list(task_id)
        missing = [i for i in ids if i not in REGISTRY]
        if missing:
            raise KeyError(
                f"unknown task_id(s): {missing!r}. "
                f"Known ids (sample): {sorted(REGISTRY)[:5]}…"
            )
        return ids
    return list_task_ids(
        include_suites=include_suites,
        exclude_suites=exclude_suites,
        include_tasks=include_tasks,
        exclude_tasks=exclude_tasks,
    )


def _build_dataset(env_ids: list[str], num_episodes: int, base_seed: int) -> Dataset:
    rows = []
    for env_id in env_ids:
        for ep in range(num_episodes):
            seed_val = int(base_seed) + ep
            info = {"env_id": env_id, "seed": seed_val}
            prompt = [
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
            ]
            rows.append(
                {
                    "info": json.dumps(info),
                    "task": _task_payload(
                        info=info,
                        prompt=prompt,
                        answer="",
                        example_id=-1,
                    ),
                    # Placeholder — filled in setup_state (verifiers allows
                    # dynamic prompt construction via state["prompt"] mutation).
                    "prompt": prompt,
                    "answer": "",
                }
            )
    return Dataset.from_list(rows)


def _parse_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _route_info(value: Any) -> dict[str, Any] | None:
    payload = _parse_mapping(value)
    if payload is None:
        return None
    if "env_id" in payload and "seed" in payload:
        return {"env_id": str(payload["env_id"]), "seed": int(payload["seed"])}
    for key in ("info", "task", "input"):
        nested = _route_info(payload.get(key))
        if nested is not None:
            return nested
    return None


def _task_payload(
    *,
    info: Mapping[str, Any],
    prompt: Any,
    answer: Any,
    example_id: Any,
) -> dict[str, Any]:
    route = _route_info(info)
    if route is None:
        raise ValueError(f"GlyphBench task payload is missing env_id/seed: {info!r}")
    return {
        "env_id": route["env_id"],
        "seed": route["seed"],
        "info": route,
        "prompt": prompt,
        "answer": "" if answer is None else answer,
        "example_id": example_id,
    }


def _jsonable_env_info(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable_env_info(item())
    if isinstance(value, dict):
        return {str(k): _jsonable_env_info(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_env_info(v) for v in value]
    return repr(value)


class GlyphbenchMultiTurnEnv(vf.MultiTurnEnv):
    """Verifiers MultiTurnEnv that drives a glyphbench game per rollout."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        rubric: vf.Rubric,
        parser: GlyphbenchXMLParser,
        n_frames: int,
        max_turns_override: int | None,
        max_output_tokens: int | None,
        use_memory: bool | None = None,
        memory_update_max_tokens: int | None = None,
        stop_after_xml: bool = True,
        forfeit_mode: str = "freeze",
        harness: str = "default",
        save_gif: bool = False,
        gif_output_dir: str | None = None,
        gif_fps: int = 8,
        gif_max_frames: int = 4000,
        parse_retries: int = 1,
        **kwargs: Any,
    ) -> None:
        if forfeit_mode not in {"freeze", "noop"}:
            raise ValueError("forfeit_mode must be 'freeze' or 'noop'")
        harness_mode = _coerce_harness(harness)
        effective_max_turns = max_turns_override if max_turns_override is not None else -1
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            parser=parser,
            max_turns=effective_max_turns,
            **kwargs,
        )
        self.n_frames = int(n_frames)
        self._max_turns_override = max_turns_override
        self._max_output_tokens = (
            None if max_output_tokens is None else int(max_output_tokens)
        )
        self._use_memory_override = _coerce_optional_bool(
            use_memory, name="use_memory"
        )
        self._memory_update_max_tokens = (
            None
            if memory_update_max_tokens is None
            else int(memory_update_max_tokens)
        )
        self._stop_after_xml = bool(stop_after_xml)
        self._forfeit_mode = forfeit_mode
        self._harness = harness_mode
        self._save_gif = bool(save_gif)
        self._gif_output_dir = Path(gif_output_dir) if gif_output_dir else None
        if self._save_gif and self._gif_output_dir is None:
            raise ValueError("gif_output_dir is required when save_gif=true")
        self._gif_fps = max(1, int(gif_fps))
        self._gif_max_frames = max(1, int(gif_max_frames))
        self._parse_retries = max(0, int(parse_retries))
        self.parser: GlyphbenchXMLParser = parser  # narrow type

    def _format_dataset(
        self,
        dataset: Dataset,
        system_prompt: str | None = None,
        few_shot: Any = None,
        question_key: str = "question",
        answer_key: str = "answer",
        map_kwargs: dict[str, Any] | None = None,
    ) -> Dataset:
        dataset = super()._format_dataset(
            dataset,
            system_prompt=system_prompt,
            few_shot=few_shot,
            question_key=question_key,
            answer_key=answer_key,
            map_kwargs=map_kwargs or {},
        )

        def attach_task_payload(row: dict[str, Any]) -> dict[str, Any]:
            info = _route_info(row.get("info")) or _route_info(row.get("task"))
            if info is None:
                raise ValueError(f"GlyphBench dataset row is missing env_id/seed: {row!r}")
            return {
                "task": _task_payload(
                    info=info,
                    prompt=row.get("prompt"),
                    answer=row.get("answer", ""),
                    example_id=row.get("example_id"),
                )
            }

        return dataset.map(attach_task_payload, **(map_kwargs or {}))

    def _resolve_use_memory(self, env_id: str) -> bool:
        if self._use_memory_override is not None:
            return self._use_memory_override
        return default_use_memory(env_id)

    def _state_uses_memory(self, state: dict[str, Any]) -> bool:
        if "memory_enabled" in state:
            return bool(state["memory_enabled"])
        game = state.get("game")
        if isinstance(game, BaseGlyphEnv):
            return self._resolve_use_memory(game.env_id())
        info = _route_info(state.get("info")) or _route_info(state.get("task"))
        if info is not None:
            return self._resolve_use_memory(info["env_id"])
        return bool(self._use_memory_override)

    async def setup_state(self, state: dict[str, Any]) -> dict[str, Any]:
        info = _route_info(state.get("info")) or _route_info(state.get("task"))
        if info is None:
            raise ValueError(
                "GlyphBench rollout state is missing env_id/seed in info or task"
            )
        env_id = info["env_id"]
        seed_val = int(info["seed"])
        use_memory = self._resolve_use_memory(env_id)
        state["info"] = {"env_id": env_id, "seed": seed_val}

        kw: dict[str, Any] = {}
        if self._max_turns_override is not None:
            kw["max_turns"] = self._max_turns_override

        game = make_env(env_id, **kw)
        obs_text, _ = game.reset(seed_val)

        state["game"] = game
        state["frames"] = deque(maxlen=self.n_frames)
        state["current_obs"] = obs_text
        state["done"] = False
        state["terminated"] = False
        state["truncated"] = False
        state["parse_failures"] = 0
        state["parse_recoveries"] = 0
        state["forfeit_count"] = 0
        state["action_completion_truncations"] = 0
        state["memory_completion_truncations"] = 0
        state["memory_parse_failures"] = 0
        state["num_action_turns"] = 0
        state["num_memory_turns"] = 0
        state["episode_return"] = 0.0
        state["num_turns"] = 0
        state["memory_enabled"] = use_memory
        state["harness"] = self._harness
        state["memory"] = ""
        state["gif_frames"] = []
        state["gif_stride"] = max(
            1,
            math.ceil(int(game.max_turns) / self._gif_max_frames),
        )
        self._record_gif_frame(state, game, force=True)

        # Populate the prompt now that we have the game instance.
        system_text = build_system_prompt(
            game,
            self._max_output_tokens,
            use_memory=use_memory,
            memory_update_max_tokens=self._memory_update_max_tokens,
            forfeit_mode=self._forfeit_mode,
            harness=self._harness,
        )
        initial_user_text = self._render_action_user(game, state, turn=0)
        state["prompt"] = [
            vf.SystemMessage(content=system_text),
            vf.UserMessage(content=initial_user_text),
        ]
        if self._stop_after_xml:
            state["sampling_args"] = xml_stop_sampling_args(
                state.get("sampling_args"), ACTION_STOP
            )
        await super().setup_state(state)
        return state

    def _render_action_user(
        self, game: BaseGlyphEnv, state: dict[str, Any], *, turn: int
    ) -> str:
        memory = state.get("memory", "") if self._state_uses_memory(state) else None
        return self._render_observation_user(game, state, turn=turn, memory=memory)

    def _render_observation_user(
        self,
        game: BaseGlyphEnv,
        state: dict[str, Any],
        *,
        turn: int,
        memory: str | None,
    ) -> str:
        return render_user_turn(
            game,
            frames=state["frames"],
            current_obs=state["current_obs"],
            turn=turn,
            max_output_tokens=self._max_output_tokens,
            memory=memory,
            enable_reasoning=(
                self._harness == AGENTICK_MARKOV_REASONER_HARNESS
                or self._state_uses_memory(state)
            ),
        )

    def _last_assistant_text(self, messages: list[dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "assistant":
                content = m.get("content", "") or ""
                return content if isinstance(content, str) else str(content)
        return ""

    def _apply_action_response(
        self,
        messages: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        game: BaseGlyphEnv = state["game"]
        last_assistant = self._last_assistant_text(messages)

        action_idx, action_name, parse_failed, parse_failure_reason = (
            self.parser.parse_action(
                last_assistant, game.action_spec, noop=game.noop_action_name
            )
        )

        pre_obs = state["current_obs"]
        if parse_failed:
            state["parse_failures"] += 1
            state["forfeit_count"] = state.get("forfeit_count", 0) + 1
            if self._forfeit_mode == "noop":
                obs_text, reward, term, trunc, env_info = game.step(action_idx)
            else:
                obs_text, reward, term, trunc, env_info = game.forfeit_turn()
            applied_action_name = "FORFEIT"
            applied_action_idx = -1
            forfeit = True
        else:
            obs_text, reward, term, trunc, env_info = game.step(action_idx)
            applied_action_name = action_name
            applied_action_idx = action_idx
            forfeit = False

        state["frames"].append((pre_obs, applied_action_name, float(reward)))
        state["current_obs"] = obs_text
        state["episode_return"] += float(reward)
        state["terminated"] = bool(term)
        state["truncated"] = bool(trunc)
        state["done"] = bool(term or trunc)
        result_info = _jsonable_env_info(env_info)
        state["info"].update(
            {
                "last_env_info": result_info,
                "turns": int(game.turn),
                "terminated": bool(term),
                "truncated": bool(trunc),
            }
        )
        if "craftax" in str(state["info"].get("env_id", "")):
            raw_score = result_info.get("num_achievements")
            achievement_names = result_info.get("achievements")
            if (
                raw_score is None
                and isinstance(achievement_names, list)
            ):
                raw_score = len(achievement_names)
            if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                state["info"]["raw_score"] = float(raw_score)
                state["info"]["raw_score_label"] = "achievements"
            if isinstance(achievement_names, list):
                state["info"]["achievement_names"] = sorted(
                    {str(name) for name in achievement_names}
                )
        self._record_gif_frame(state, game, force=bool(term or trunc))

        return {
            "action_idx": applied_action_idx,
            "action_name": applied_action_name,
            "action_chosen": applied_action_name,
            "parse_failed": parse_failed,
            "parse_failure_reason": parse_failure_reason,
            "forfeit_mode": self._forfeit_mode if parse_failed else None,
            "forfeit_action_name": action_name if parse_failed else None,
            "forfeit": forfeit,
            "pre_obs": pre_obs,
            "next_obs": obs_text,
            "reward": float(reward),
            "terminated": bool(term),
            "truncated": bool(trunc),
            "env_info": result_info,
        }

    async def env_response(
        self,
        messages: list[dict[str, Any]],
        state: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        game: BaseGlyphEnv = state["game"]

        messages = await self._recover_invalid_action(messages, state, game)

        result = self._apply_action_response(messages, state)

        # Set the per-turn reward and action-step extras on the trajectory step
        # verifiers appended before calling env_response.
        traj = state.get("trajectory", [])
        if traj:
            last_step = traj[-1]
            last_step["reward"] = float(result["reward"])
            if not self._state_uses_memory(state):
                existing_extras = last_step.get("extras") or {}
                existing_extras.update({
                    "glyphbench_step_role": "action",
                    "parse_failed": bool(result["parse_failed"]),
                    "parse_failure_reason": result["parse_failure_reason"],
                    "action_chosen": result["action_chosen"],
                    "forfeit": bool(result["forfeit"]),
                    "env_info": result["env_info"],
                })
                last_step["extras"] = existing_extras

        next_user = self._render_action_user(game, state, turn=game.turn)
        response_msg = [vf.UserMessage(content=next_user)]
        if state["done"]:
            # Game ended this turn — signal the rollout loop to skip the
            # otherwise-wasted final model call.
            state["final_env_response"] = response_msg
        return response_msg

    async def _recover_invalid_action(
        self,
        messages: list[dict[str, Any]],
        state: dict[str, Any],
        game: BaseGlyphEnv,
    ) -> list[dict[str, Any]]:
        """Focused same-turn re-ask so a formatting slip need not cost a turn."""

        if self._parse_retries <= 0:
            return messages
        last_step = state.get("trajectory", [])[-1]
        retry = await self._retry_invalid_completion(
            list(last_step["prompt"]),
            list(last_step["completion"]),
            state,
            game,
        )
        if retry is None:
            return messages
        response, completion, tokens, details = retry
        extras = last_step.setdefault("extras", {})
        extras.update(details)
        if response is not None:
            last_step["completion"] = completion
            last_step["response"] = response
            last_step["tokens"] = tokens
            state["parse_recoveries"] += 1
            return [*list(last_step["prompt"]), *completion]
        return messages

    async def _retry_invalid_completion(
        self,
        prompt_messages: list[Any],
        completion: list[Any],
        state: dict[str, Any],
        game: BaseGlyphEnv,
    ) -> tuple[Response | None, list[Any], Any, dict[str, Any]] | None:
        original_text = self._last_assistant_text(completion)
        parsed = self.parser.parse_action(
            original_text,
            game.action_spec,
            noop=game.noop_action_name,
        )
        if not parsed[2] or self._parse_retries <= 0:
            return None

        retry_messages: list[Any] = [
            *prompt_messages,
            *completion,
            vf.UserMessage(
                content=(
                    "Your previous reply could not be parsed, so the turn would "
                    "be forfeited. Retry the SAME turn now. Reply with exactly "
                    "one tag and no other visible text: <action>NAME</action>. "
                    "NAME must be exactly one of: "
                    + ", ".join(game.action_spec.names)
                    + "."
                )
            ),
        ]
        attempts: list[str] = []
        for _ in range(self._parse_retries):
            try:
                response = await self.get_model_response(state, retry_messages)
                retried_completion = await parse_response_message(response)
                tokens = await parse_response_tokens(response, self.max_seq_len)
            except Exception as exc:  # noqa: BLE001
                return (
                    None,
                    completion,
                    None,
                    {"parse_retry_error": f"{type(exc).__name__}: {exc}"},
                )
            retry_text = self._last_assistant_text(retried_completion)
            attempts.append(retry_text)
            reparsed = self.parser.parse_action(
                retry_text,
                game.action_spec,
                noop=game.noop_action_name,
            )
            if not reparsed[2]:
                return (
                    response,
                    retried_completion,
                    tokens,
                    {
                        "parse_recovered": True,
                        "parse_retry_original_output": original_text,
                        "parse_retry_attempts": attempts,
                    },
                )
            retry_messages = [
                *retry_messages,
                *retried_completion,
                vf.UserMessage(
                    content=(
                        "Still invalid. Output only <action>NAME</action> using "
                        "an exact valid action name."
                    )
                ),
            ]
        return (
            None,
            completion,
            None,
            {"parse_retry_attempts": attempts, "parse_recovered": False},
        )

    async def get_prompt_messages(self, state: dict[str, Any]) -> list[Any]:
        """Stateless prompt: each turn the LLM sees only [system, current_user_obs]."""
        if self._state_uses_memory(state):
            if len(state["trajectory"]) == 0:
                return state["prompt"]
            game: BaseGlyphEnv = state["game"]
            system_msg = state["prompt"][0]
            next_user = self._render_action_user(game, state, turn=game.turn)
            return [system_msg, vf.UserMessage(content=next_user)]
        if len(state["trajectory"]) == 0:
            return state["prompt"]
        prev = state["trajectory"][-1]
        messages = list(prev["prompt"]) + list(prev["completion"])
        new_user = await self.env_response(messages, state)
        system_msg = prev["prompt"][0]
        return [system_msg] + list(new_user)

    async def add_model_response(
        self,
        state: dict[str, Any],
        prompt_messages: list[Any],
        response: Response,
    ) -> None:
        if not self._state_uses_memory(state):
            await super().add_model_response(state, prompt_messages, response)
            state["num_turns"] = state.get("num_turns", 0) + 1
            state["num_action_turns"] = state.get("num_action_turns", 0) + 1
            traj = state.get("trajectory", [])
            if traj and traj[-1].get("is_truncated"):
                state["action_completion_truncations"] = (
                    state.get("action_completion_truncations", 0) + 1
                )
            return

        # Memory mode: each environment turn produces TWO trajectory steps so
        # every step's completion is purely assistant tokens (prime-rl's
        # pretokenize_rollout_trajectory rejects mixed-role completions).
        #
        #   action step:  prompt=[system, user_obs_t]
        #                 completion=[assistant_action_t]
        #   memory step:  prompt=[system, user_obs_t, assistant_action_t, memory_user_t]
        #                 completion=[assistant_memory_response_t]
        #
        # Both steps share the same per-turn env reward and (downstream) the
        # same advantage, so the trainer treats memory tokens as on-policy
        # too.
        action_completion = await parse_response_message(response)
        action_tokens = await parse_response_tokens(response, self.max_seq_len)
        game: BaseGlyphEnv = state["game"]
        parse_retry_details: dict[str, Any] = {}
        retry = await self._retry_invalid_completion(
            list(prompt_messages), action_completion, state, game
        )
        if retry is not None:
            retried_response, retried_completion, retried_tokens, parse_retry_details = retry
            if retried_response is not None:
                response = retried_response
                action_completion = retried_completion
                action_tokens = retried_tokens
                state["parse_recoveries"] += 1
        response_is_truncated = response.message.is_truncated or False
        action_is_truncated = response_is_truncated or (
            action_tokens is not None and bool(action_tokens.get("is_truncated"))
        )

        messages_for_action = list(prompt_messages) + list(action_completion)
        action_result = self._apply_action_response(messages_for_action, state)
        state["num_turns"] = state.get("num_turns", 0) + 1
        turn_reward = float(action_result["reward"])

        action_step = TrajectoryStep(
            prompt=prompt_messages,
            completion=action_completion,
            response=response,
            tokens=action_tokens,
            reward=turn_reward,
            advantage=None,
            is_truncated=action_is_truncated,
            trajectory_id=state["trajectory_id"],
            extras={
                "glyphbench_step_role": "action",
                "parse_failed": bool(action_result["parse_failed"]),
                "parse_failure_reason": action_result["parse_failure_reason"],
                "action_chosen": action_result["action_chosen"],
                "forfeit": bool(action_result["forfeit"]),
                "env_info": action_result["env_info"],
                **parse_retry_details,
            },
        )
        state["trajectory"].append(action_step)
        state["num_action_turns"] = state.get("num_action_turns", 0) + 1
        if action_is_truncated:
            state["action_completion_truncations"] = (
                state.get("action_completion_truncations", 0) + 1
            )

        if state["done"]:
            game: BaseGlyphEnv = state["game"]
            final_user = self._render_action_user(game, state, turn=game.turn)
            state["final_env_response"] = [vf.UserMessage(content=final_user)]
            return

        # Memory-update prompt: re-injects the action turn's reasoning
        # (which the chat template would otherwise strip from prior
        # assistant turns) + parse/truncation flags + env response + the
        # new obs. See memory.build_memory_update_user docstring for the
        # rationale on each section.
        common_memory_user_kwargs = {
            "action_reasoning": action_reasoning_text(action_completion),
            "action_chosen": str(action_result["action_chosen"]),
            "parse_failed": bool(action_result["parse_failed"]),
            "parse_failure_reason": action_result["parse_failure_reason"],
            "forfeit_mode": action_result.get("forfeit_mode"),
            "forfeit_action_name": action_result.get("forfeit_action_name"),
            "action_truncated": bool(action_is_truncated),
            "reward": turn_reward,
            "terminated": bool(action_result["terminated"]),
            "truncated": bool(action_result["truncated"]),
            "next_obs": state["current_obs"],
        }
        memory_user = build_memory_update_user(**common_memory_user_kwargs)
        memory_prompt_messages = messages_for_action + [memory_user]
        memory_response = await self.get_model_response(
            state,
            memory_prompt_messages,
            sampling_args=memory_sampling_args(
                state.get("sampling_args"),
                self._memory_update_max_tokens,
                stop_after_xml=self._stop_after_xml,
            ),
        )
        memory_completion = await parse_response_message(memory_response)
        memory_tokens = await parse_response_tokens(memory_response, self.max_seq_len)
        memory_response_text = self._last_assistant_text(memory_completion)
        extraction = extract_memory_update(memory_response_text)
        memory_parse_failed = bool(extraction.parse_failed)

        # Retain previous memory on parse failure; otherwise apply the new one.
        if not extraction.parse_failed:
            state["memory"] = extraction.memory
        # state["memory"] otherwise stays unchanged.

        memory_response_is_truncated = memory_response.message.is_truncated or False
        memory_is_truncated = memory_response_is_truncated or (
            memory_tokens is not None and bool(memory_tokens.get("is_truncated"))
        )

        memory_step = TrajectoryStep(
            prompt=memory_prompt_messages,
            completion=memory_completion,
            response=memory_response,
            tokens=memory_tokens,
            reward=turn_reward,
            advantage=None,
            is_truncated=bool(memory_is_truncated),
            trajectory_id=state["trajectory_id"],
            extras={
                "glyphbench_step_role": "memory",
                "memory_parse_failed": memory_parse_failed,
                "stored_memory": state["memory"],
                "harness": state.get("harness", "default"),
            },
        )
        state["trajectory"].append(memory_step)
        state["num_memory_turns"] = state.get("num_memory_turns", 0) + 1
        if memory_is_truncated:
            state["memory_completion_truncations"] = (
                state.get("memory_completion_truncations", 0) + 1
            )
        if memory_parse_failed:
            state["memory_parse_failures"] = (
                state.get("memory_parse_failures", 0) + 1
            )

    async def render_completion(self, state: dict[str, Any]) -> None:
        """Stitch full rollout from trajectory by walking each step and appending
        only the messages this step contributes that aren't already in the
        running transcript. Handles both stateless action-only steps (whose
        prompts share only the system message with the previous turn) and
        memory steps (whose prompts also include the just-appended action +
        a new memory_update_user).
        """
        if len(state["trajectory"]) == 0:
            state["completion"] = []
            return
        parts: list[Any] = []
        running = list(state["prompt"])
        for step in state["trajectory"]:
            step_prompt = list(step["prompt"])
            i = 0
            while (
                i < len(running)
                and i < len(step_prompt)
                and running[i] == step_prompt[i]
            ):
                i += 1
            new_prompt_tail = step_prompt[i:]
            parts.extend(new_prompt_tail)
            parts.extend(list(step["completion"]))
            running = step_prompt + list(step["completion"])
        if state.get("final_env_response"):
            parts.extend(list(state["final_env_response"]))
        state["completion"] = parts

    @vf.stop
    async def is_done(self, state: dict[str, Any]) -> bool:
        return bool(state.get("done", False))

    @vf.cleanup
    async def close_game(self, state: dict[str, Any]) -> None:
        game = state.pop("game", None)
        if game is not None:
            if self._save_gif and state.get("gif_frames"):
                env_slug = re.sub(
                    r"[^A-Za-z0-9_.-]+",
                    "_",
                    str(state.get("info", {}).get("env_id", "glyphbench")),
                ).strip("_")
                seed = state.get("info", {}).get("seed", "unknown")
                trajectory = re.sub(
                    r"[^A-Za-z0-9_.-]+",
                    "_",
                    str(state.get("trajectory_id", state.get("example_id", "rollout"))),
                ).strip("_")[:16]
                assert self._gif_output_dir is not None
                gif_path = self._gif_output_dir / (
                    f"{env_slug}-seed-{seed}-{trajectory}.gif"
                )
                try:
                    await asyncio.to_thread(
                        write_gif,
                        state["gif_frames"],
                        gif_path,
                        fps=self._gif_fps,
                    )
                    state["info"]["gif_path"] = str(gif_path.resolve())
                    state["info"]["gif_frames"] = len(state["gif_frames"])
                except Exception as exc:  # noqa: BLE001
                    state["info"]["gif_error"] = f"{type(exc).__name__}: {exc}"
            state.pop("gif_frames", None)
            with suppress(AttributeError, OSError, RuntimeError):
                game.close()

    def _record_gif_frame(
        self,
        state: dict[str, Any],
        game: BaseGlyphEnv,
        *,
        force: bool = False,
    ) -> None:
        if not self._save_gif or "craftax" not in game.env_id():
            return
        stride = max(1, int(state.get("gif_stride", 1)))
        if not force and game.turn > 0 and game.turn % stride != 0:
            return
        frames = state.get("gif_frames")
        if not isinstance(frames, list):
            return
        frame = render_native_craftax_frame(game)
        if frame is None:
            return
        if len(frames) >= self._gif_max_frames:
            if force:
                frames[-1] = frame
            return
        frames.append(frame)

"""BlenderGym multimodal verifiers environment.

Phase 5 fully integrates dataset / render / rubric / XMLParser / prompts.

The flow per rollout is:

1. ``init_state`` (verifiers default) sets up ``trajectory_id`` etc.
2. ``setup_state`` builds ``state["rollout"]`` (a
   :class:`~blendergym.schema.Rollout`) and carves out a per-rollout work dir
   (``<work_root>/<split>/example_<id>__<task_id>/<traj_id[:8]>``), populates
   ``inputs/`` symlinks to the dataset, picks a GPU from the pool, and
   base64-encodes the goal / init reference images.
3. ``get_prompt_messages``:
     - turn 0: system prompt + goal image + init image + ``start.py`` source.
     - turn N: previous prompt + previous completion + most-recent render.
4. ``add_model_response`` (override): ``super()`` first to push the step,
   then build a typed :class:`~blendergym.schema.TurnRecord` —
   either filling from the XML parse failure or from a Blender render — and
   append it to ``state["rollout"].turns``. ``response.txt`` is always written to
   the turn dir, including the parse-failure path.
5. Cleanup: ``BlenderGymRubric.@vf.cleanup`` writes ``meta.json`` /
   ``trajectory.json`` / ``trajectory.md`` (the rubric runs *after*
   ``score_rollout``, so ``state["rollout"].final_reward`` is already populated).
   ``keep_failed_only`` rmtree also lives on the rubric for the same reason.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import verifiers as vf

from .artifact_manager import ArtifactManager, ArtifactPolicy
from .dataset import build_dataset
from .prompts import REFINE_INSTRUCTION, SYSTEM_PROMPT, TASK_INSTRUCTION
from .rubric import BlenderGymRubric
from .schema import Rollout, Task, require_rollout
from .services.render.client import RenderClient
from .services.health import ensure_service_ready
from .trajectory_writer import completion_to_text

logger = logging.getLogger(__name__)


def _png_to_data_url(image_path: str | Path) -> str:
    """Read a PNG file and return ``data:image/png;base64,...`` URL."""
    raw = Path(image_path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _content_text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _content_image(data_url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url}}


class BlenderGymEnv(vf.MultiTurnEnv):
    """BlenderGym placement environment over MultiTurnEnv."""

    def __init__(
        self,
        data_root: str | Path = "data/blendergym",
        task_types: Sequence[str] = ("placement",),
        max_turns: int = 3,
        work_root: str | Path = "outputs/blendergym_v1/blendergym_work",
        keep_failed_only: bool = False,
        env_name: str = "blendergym",
        split: str = "train",
        eval_split: str = "eval",
        eval_holdout: int = 5,
        render_service_url: str = "http://localhost:8420",
        score_service_url: str = "http://localhost:8421",
        render_timeout_s: int = 600,
        # -- reward weights (overridable from TOML) --
        reward_weights: dict[str, float] | None = None,
        # -- artifact policy --
        save_blender_log: bool = True,
        save_response_txt: bool = True,
        save_meta_json: bool = True,
        save_trajectory_html: bool = True,
        remove_intermediate_turns: bool = False,
        max_rollouts_per_example: int = 0,
        **kwargs: Any,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")

        self.data_root = Path(data_root).expanduser().resolve()
        self.task_types = tuple(task_types)
        self.work_root = Path(work_root).expanduser().resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)

        self.render_client = RenderClient(render_service_url, render_timeout_s)
        ensure_service_ready(render_service_url, "render")

        policy = ArtifactPolicy(
            save_blender_log=save_blender_log,
            save_response_txt=save_response_txt,
            save_meta_json=save_meta_json,
            save_trajectory_html=save_trajectory_html,
            keep_failed_only=keep_failed_only,
            remove_intermediate_turns=remove_intermediate_turns,
            max_rollouts_per_example=max_rollouts_per_example,
        )
        self.artifact_manager = ArtifactManager(self.work_root, policy)
        self.env_name = env_name
        self.split = split
        self.eval_split = eval_split
        self.eval_holdout = eval_holdout
        self.render_timeout_s = render_timeout_s

        self.parser = vf.XMLParser(["code"], answer_field="code")

        rubric = BlenderGymRubric(
            score_service_url=score_service_url,
            parser=self.parser,
            artifact_manager=self.artifact_manager,
            reward_weights=reward_weights,
        )

        def _train_dataset_builder():
            return build_dataset(
                self.data_root,
                task_types=self.task_types,
                split=self.split,
                eval_holdout=self.eval_holdout,
            )

        def _eval_dataset_builder():
            return build_dataset(
                self.data_root,
                task_types=self.task_types,
                split=self.eval_split,
                eval_holdout=self.eval_holdout,
            )

        super().__init__(
            dataset=_train_dataset_builder,
            eval_dataset=_eval_dataset_builder,
            system_prompt=SYSTEM_PROMPT,
            max_turns=max_turns,
            rubric=rubric,
            parser=self.parser,
            **kwargs,
        )

    # ----------------------------------------------------------------- helpers

    def _build_initial_user_message(
        self,
        *,
        goal_data_url: str,
        init_data_url: str,
        start_code: str,
        task_id: str,
    ) -> dict[str, Any]:
        text = (
            f"Task: {task_id} (placement). {TASK_INSTRUCTION}\n\n"
            "GOAL image (target render):"
        )
        intro_init = (
            "INITIAL image (current scene rendered from the program below):"
        )
        # Wrap the initial program in <initial_code>...</initial_code> rather than
        # markdown fences so small VLMs don't mimic the fence and forget to emit
        # the required <code>...</code> answer block (observed during phase 5b).
        program_block = (
            "INITIAL program (start.py):\n"
            "<initial_code>\n"
            f"{start_code.rstrip()}\n"
            "</initial_code>"
        )
        content = [
            _content_text(text),
            _content_image(goal_data_url),
            _content_text(intro_init),
            _content_image(init_data_url),
            _content_text(program_block),
        ]
        return {"role": "user", "content": content}

    def _build_refine_user_message(self, *, render_data_url: str, turn_idx: int) -> dict[str, Any]:
        text = f"Render of your turn-{turn_idx} program:"
        return {
            "role": "user",
            "content": [
                _content_text(text),
                _content_image(render_data_url),
                _content_text(REFINE_INSTRUCTION),
            ],
        }

    # ------------------------------------------------------------------- hooks

    async def setup_state(self, state: vf.State) -> vf.State:
        info = state.get("info") or {}
        task = Task.from_info(info)
        env_name = state.get("env_name", self.env_name)
        example_id = state.get("example_id", info.get("example_id"))
        split = self.split
        metadata = {
            "env": env_name,
            "split": split,
            "example_id": example_id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "trajectory_id": state["trajectory_id"],
        }
        mgr = self.artifact_manager
        work_dir = mgr.make_rollout_dir(
            traj_id=state["trajectory_id"],
            task_id=task.task_id,
            split=split,
            example_id=example_id,
        )

        rollout = Rollout(
            task=task,
            trajectory_id=state["trajectory_id"],
            work_dir=work_dir,
            max_turns=self.max_turns,
            metadata=metadata,
            start_code_text=task.start_code_path.read_text(encoding="utf-8"),
            goal_image_data_url=_png_to_data_url(task.goal_image),
            init_image_data_url=_png_to_data_url(task.init_image),
        )
        mgr.populate_input_symlinks(rollout)
        state["rollout"] = rollout
        return state

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        rollout = require_rollout(state)
        trajectory = state.get("trajectory", [])
        if not trajectory:
            base = list(state["prompt"])  # already contains the system message
            base.append(
                self._build_initial_user_message(
                    goal_data_url=rollout.goal_image_data_url or "",
                    init_data_url=rollout.init_image_data_url or "",
                    start_code=rollout.start_code_text or "",
                    task_id=rollout.task.task_id,
                )
            )
            return base

        # Subsequent turn: rebuild from the previous step + show latest render.
        prev_step = trajectory[-1]
        prev_prompt = list(prev_step["prompt"])
        prev_completion = list(prev_step["completion"])
        messages: list[Any] = prev_prompt + prev_completion

        last_render = self.artifact_manager.last_render_path(rollout)
        turn_idx = rollout.render_count
        if last_render is not None:
            render_url = _png_to_data_url(last_render)
            messages.append(
                self._build_refine_user_message(
                    render_data_url=render_url,
                    turn_idx=turn_idx,
                )
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        _content_text(
                            "Your previous program could not be parsed or its "
                            "render failed. Try again — emit the full program "
                            "inside a single <code>...</code> block."
                        )
                    ],
                }
            )
        return messages

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        # All message construction lives in ``get_prompt_messages``; no implicit
        # env response between turns.
        return []

    async def add_model_response(
        self,
        state: vf.State,
        prompt_messages: vf.Messages,
        response: Any,
    ) -> None:
        # 1. Let the parent push the step so trajectory[-1] is the just-finished step.
        await super().add_model_response(state, prompt_messages, response)

        rollout = require_rollout(state)
        if not state.get("trajectory"):
            return
        completion = state["trajectory"][-1]["completion"]

        # Fix thinking-only responses (model exhausted tokens during <think>)
        for msg in completion:
            if msg.get("role") == "assistant" and msg.get("content") is None and not msg.get("tool_calls"):
                msg["content"] = ""

        mgr = self.artifact_manager
        turn_idx = rollout.render_count
        paths = mgr.begin_turn(rollout.work_dir, turn_idx)
        mgr.write_response(paths, completion_to_text(completion))

        record = mgr.init_record(turn_idx)
        code = self.parser.parse_answer(completion)

        if not code or not str(code).strip():
            record.fill_xml_parse_failure()
        else:
            result = await asyncio.to_thread(
                self.render_client.render,
                blend_file=rollout.task.blend_file,
                code=str(code),
                output_dir=str(paths.turn_dir),
            )
            if paths.log and result.stderr:
                paths.log.write_text(result.stderr, encoding="utf-8")
            mgr.fill_record(record, result)
            record.render_gpu_id = result.gpu_id

        rollout.turns.append(record)

    async def close(self) -> None:
        self.render_client.close()

    def __del__(self) -> None:
        try:
            self.render_client.close()
        except Exception:
            pass


def load_environment(**kwargs: Any) -> BlenderGymEnv:
    """Factory used by ``verifiers.load_environment("blendergym")``.

    Accepts any kwargs (including ``max_seq_len`` injected by prime-rl) and
    forwards them to :class:`BlenderGymEnv`.
    """
    return BlenderGymEnv(**kwargs)

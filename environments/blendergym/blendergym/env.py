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
from itertools import count
from pathlib import Path
from typing import Any

import verifiers as vf

from .dataset import build_dataset
from .prompts import REFINE_INSTRUCTION, SYSTEM_PROMPT, TASK_INSTRUCTION
from .render import RenderResult, run_blender
from .rubric import BlenderGymRubric
from .schema import Rollout, Task, TurnRecord, require_rollout
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
        blender_bin: str | Path = "_reference_codes/VIGA/utils/third_party/infinigen/blender/blender",
        gpu_id_pool: Sequence[int] = (0,),
        work_root: str | Path = "outputs/blendergym_v1/blendergym_work",
        keep_failed_only: bool = False,
        env_name: str = "blendergym",
        split: str = "train",
        eval_split: str = "eval",
        eval_holdout: int = 5,
        render_timeout_s: int = 120,
        cycles_resolution: int = 512,
        cycles_samples: int = 16,
        cycles_denoiser: str = "OPENIMAGEDENOISE",
        cycles_compute_device: str = "OPTIX",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        **kwargs: Any,
    ) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        if not gpu_id_pool:
            raise ValueError("gpu_id_pool must contain at least one GPU id")

        # Cycles knobs are forwarded to the Blender child via os.environ
        # (run_blender's ``_build_subprocess_env`` already inherits dict(os.environ),
        # so the public Python signature of run_blender stays unchanged).
        # ``os.environ`` is process-local; each prime-rl env worker is its own
        # Python process, so workers don't fight over these vars.
        os.environ["BLENDERGYM_RENDER_RESOLUTION"] = str(cycles_resolution)
        os.environ["BLENDERGYM_CYCLES_SAMPLES"] = str(cycles_samples)
        os.environ["BLENDERGYM_CYCLES_DENOISER"] = cycles_denoiser
        os.environ["BLENDERGYM_CYCLES_COMPUTE_DEVICE"] = cycles_compute_device

        self.data_root = Path(data_root).expanduser().resolve()
        self.task_types = tuple(task_types)
        self.blender_bin = Path(blender_bin).expanduser().resolve()
        self.gpu_id_pool = tuple(gpu_id_pool)
        self.work_root = Path(work_root).expanduser().resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.keep_failed_only = keep_failed_only
        # ``env_name`` mirrors the orchestrator's resolved env name so that
        # local metadata matches wandb sample tables (e.g. ``"blendergym"`` for
        # train, ``"blendergym-eval"`` for eval). Should be set per-env in the
        # toml ``args`` block — see ``configs/multimodal/rl_blendergym.toml``.
        self.env_name = env_name
        self.split = split
        self.eval_split = eval_split
        self.eval_holdout = eval_holdout
        self.render_timeout_s = render_timeout_s

        # Round-robin GPU assignment shared across rollouts in this worker.
        # itertools.count is process-local, which is what we want — multiple
        # env workers each cycle through the pool independently.
        self._gpu_counter = count()

        # Drop ``think`` from the XML parser: Qwen3 chat templates auto-strip
        # ``<think>...</think>``, so insisting the model emit them at the
        # parser layer just guarantees parse failures (see plan §"容易踩的坑").
        # The system prompt explicitly tells the model that ``<code>`` must be
        # the last thing in the reply; reasoning before that block is fine.
        self.parser = vf.XMLParser(["code"], answer_field="code")

        rubric = BlenderGymRubric(
            clip_model_name=clip_model_name,
            clip_pretrained=clip_pretrained,
            parser=self.parser,
            keep_failed_only=self.keep_failed_only,
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

    def _next_gpu(self) -> int:
        return self.gpu_id_pool[next(self._gpu_counter) % len(self.gpu_id_pool)]

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
        work_dir = self._make_work_dir(
            traj_id=state["trajectory_id"],
            task_id=task.task_id,
            split=split,
            example_id=example_id,
        )

        rollout = Rollout(
            task=task,
            trajectory_id=state["trajectory_id"],
            work_dir=work_dir,
            gpu_id=self._next_gpu(),
            max_turns=self.max_turns,
            metadata=metadata,
            start_code_text=task.start_code_path.read_text(encoding="utf-8"),
            goal_image_data_url=_png_to_data_url(task.goal_image),
            init_image_data_url=_png_to_data_url(task.init_image),
        )
        self._populate_inputs_symlinks(rollout)
        state["rollout"] = rollout
        return state

    def _make_work_dir(
        self,
        traj_id: str,
        task_id: str,
        split: str | None = None,
        example_id: object | None = None,
    ) -> Path:
        if split is not None and isinstance(example_id, int):
            work_dir = (
                self.work_root
                / split
                / f"example_{example_id:04d}__{task_id}"
                / traj_id[:8]
            )
        else:
            work_dir = self.work_root / f"{task_id}__{traj_id[:8]}"
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _populate_inputs_symlinks(self, rollout: Rollout) -> None:
        """Populate rollout ``inputs/`` symlinks without copying dataset assets."""
        inputs_dir = rollout.work_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        for src, link_name in (
            (rollout.task.goal_image, "goal.png"),
            (rollout.task.init_image, "init.png"),
            (rollout.task.start_code_path, "start.py"),
        ):
            link = inputs_dir / link_name
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(os.path.abspath(src), link)

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

        last_render = rollout.last_render_path
        turn_idx = rollout.render_count
        if last_render and last_render.is_file():
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
        last_step = state["trajectory"][-1]
        completion = last_step["completion"]

        # 2. Always persist the raw model response.txt — including XML parse
        # failure paths — so post-mortem can see what the model emitted.
        turn_idx = rollout.render_count
        turn_dir = rollout.work_dir / f"turn_{turn_idx}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / "response.txt").write_text(
            completion_to_text(completion), encoding="utf-8"
        )

        record = TurnRecord.for_turn(turn_idx)
        code = self.parser.parse_answer(completion)

        if not code or not str(code).strip():
            record.fill_xml_parse_failure()
        else:
            try:
                result: RenderResult = await asyncio.to_thread(
                    run_blender,
                    blend_file=rollout.task.blend_file,
                    code=str(code),
                    output_dir=turn_dir,
                    blender_bin=self.blender_bin,
                    gpu_id=rollout.gpu_id,
                    timeout=self.render_timeout_s,
                )
            except FileNotFoundError:
                # Operator-side misconfiguration (missing Blender / .blend) —
                # let it propagate so vf.Error path can mark the rollout as
                # errored.
                raise
            record.fill_from_render(result)

        rollout.turns.append(record)


def load_environment(**kwargs: Any) -> BlenderGymEnv:
    """Factory used by ``verifiers.load_environment("blendergym")``.

    Accepts any kwargs (including ``max_seq_len`` injected by prime-rl) and
    forwards them to :class:`BlenderGymEnv`.
    """
    return BlenderGymEnv(**kwargs)

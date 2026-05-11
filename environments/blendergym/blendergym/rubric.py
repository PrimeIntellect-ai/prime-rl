"""BlenderGym rubric.

The Phase 4 rubric is intentionally thin:

* one reward function — CLIP cosine similarity between the latest render and
  the goal reference image; the dominant learning signal for placement.
* three zero-weight metrics — ``xml_parse_success`` / ``render_success`` /
  ``code_non_empty``. They never affect the gradient but they're the most
  important diagnostic knobs in wandb when training stalls (they tell you
  whether the model is failing at format / Blender / semantics, in that
  order).

The CLIP backbone is **lazy-loaded** on first use, with the device taken from
``state["rollout"].gpu_id`` rather than the env-worker's ambient CUDA context —
see plan §"render.py 接口" for why those two are not the same.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import verifiers as vf
from PIL import Image

from .artifact_manager import ArtifactManager
from .schema import require_rollout

logger = logging.getLogger(__name__)


DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "openai"


def _resolve_device(gpu_id: int | None) -> str:
    """Pick the CUDA device for CLIP inference.

    Prefers the rollout-assigned ``gpu_id`` so that the CLIP forward
    pass lands on the same H20 chip we already pinned the Blender subprocess
    to via ``CUDA_VISIBLE_DEVICES`` (Phase 3); falls back to plain ``cuda`` and
    finally ``cpu``.
    """
    if gpu_id is not None and torch.cuda.is_available():
        return f"cuda:{int(gpu_id)}"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_clip_cosine_similarity(
    image_a: str | Path,
    image_b: str | Path,
    *,
    model,
    preprocess,
    device: str,
) -> float:
    """Cosine similarity of two CLIP image embeddings.

    Pure helper exposed so the test suite can validate the metric without
    mocking the rubric class. Returns a Python float in roughly ``[-1, 1]``
    (natural images typically fall in ``[0, 1]``).
    """
    img_a = Image.open(image_a).convert("RGB")  # PIL.Image
    img_b = Image.open(image_b).convert("RGB")  # PIL.Image
    a = preprocess(img_a).unsqueeze(0).to(device)  # (1, 3, H, W)
    b = preprocess(img_b).unsqueeze(0).to(device)  # (1, 3, H, W)
    with torch.no_grad():
        emb_a = model.encode_image(a)  # (1, D)
        emb_b = model.encode_image(b)  # (1, D)
        emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)  # (1, D) unit-norm
        emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)  # (1, D) unit-norm
        cos = (emb_a * emb_b).sum(dim=-1)  # (1,) scalar similarity
    return float(cos.item())


class BlenderGymRubric(vf.Rubric):
    """CLIP-similarity reward + format/render/code diagnostics."""

    def __init__(
        self,
        clip_model_name: str = DEFAULT_CLIP_MODEL,
        clip_pretrained: str = DEFAULT_CLIP_PRETRAINED,
        parser: vf.Parser | None = None,
        artifact_manager: ArtifactManager | None = None,
    ) -> None:
        if artifact_manager is None:
            raise TypeError("artifact_manager is required")
        super().__init__(parser=parser)
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.artifact_manager = artifact_manager

        self._clip_model = None
        self._clip_preprocess = None
        self._clip_device: str | None = None

        self.add_reward_func(self.clip_similarity, weight=1.0)
        self.add_metric(self.xml_parse_success)
        self.add_metric(self.render_success)
        self.add_metric(self.code_non_empty)

    def _ensure_clip(self, device: str) -> None:
        """Load (or migrate) the CLIP backbone to ``device``. No-op if already there."""
        if self._clip_model is not None and self._clip_device == device:
            return

        import open_clip

        # ``force_quick_gelu=True`` aligns the activation with the original OpenAI
        # checkpoint; without it open_clip>=3.0 emits a QuickGELU mismatch warning
        # and silently falls back to GELU, which is statistically near-identical
        # but means our reward isn't bit-equal to the published baseline.
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name,
            pretrained=self.clip_pretrained,
            device=device,
            force_quick_gelu=self.clip_pretrained == "openai",
        )
        model.eval()
        self._clip_model = model
        self._clip_preprocess = preprocess
        self._clip_device = device

    async def clip_similarity(
        self,
        state: vf.State,
        info: dict[str, Any],
    ) -> float:
        """Return the CLIP cosine similarity between the latest render and the goal.

        Returns 0.0 (instead of erroring) when there's nothing to compare —
        XML parse failure, render failure, or missing goal — so the rollout
        still produces a well-defined reward.

        Always writes ``rollout.final_reward`` (success and fallback paths) so
        the trajectory writer in ``cleanup`` can serialize the BlenderGym
        artifact metric. Verifiers ``state["reward"]`` is the trainer-facing
        weighted reward; it currently matches this value only because the CLIP
        reward weight is 1.0.
        """
        try:
            rollout = require_rollout(state)
        except RuntimeError:
            return 0.0

        last_render = self.artifact_manager.last_render_path(rollout)
        goal = rollout.task.goal_image
        if last_render is None or not goal.is_file():
            rollout.final_reward = 0.0
            return 0.0

        device = _resolve_device(rollout.gpu_id)
        self._ensure_clip(device)

        reward = compute_clip_cosine_similarity(
            last_render,
            goal,
            model=self._clip_model,
            preprocess=self._clip_preprocess,
            device=device,
        )
        rollout.final_reward = reward
        return reward

    async def xml_parse_success(self, state: vf.State) -> float:
        try:
            rollout = require_rollout(state)
        except RuntimeError:
            return 0.0
        return float(rollout.xml_parsed)

    async def render_success(self, state: vf.State) -> float:
        try:
            rollout = require_rollout(state)
        except RuntimeError:
            return 0.0
        return float(rollout.render_success)

    async def code_non_empty(self, completion, parser) -> float:  # type: ignore[override]
        if parser is None:
            return 0.0
        ans = parser.parse_answer(completion)
        return float(bool(ans and str(ans).strip()))

    @vf.cleanup
    async def write_artifacts_handler(self, state: vf.State) -> None:
        """Write trajectory artifacts and apply retention policy.

        Runs *after* ``score_rollout``, so ``rollout.final_reward`` and
        ``state["metrics"]`` are already populated.
        """
        try:
            rollout = require_rollout(state)
        except RuntimeError:
            return

        mgr = self.artifact_manager
        try:
            mgr.save_trajectory(rollout, metrics=state.get("metrics"))
        except Exception:
            logger.exception(
                "save_trajectory failed for work_dir=%s", rollout.work_dir
            )
        mgr.cleanup_rollout(rollout)
        mgr.prune_old_rollouts(rollout)

"""BlenderGym rubric.

Reward: CLIP cosine similarity between the latest render and the goal
reference image, computed via the Score Service.

Three zero-weight metrics (xml_parse_success / render_success /
code_non_empty) provide diagnostic signals in wandb when training stalls.
"""

from __future__ import annotations

import logging
from typing import Any

import verifiers as vf

from .artifact_manager import ArtifactManager
from .schema import require_rollout
from .services.score.client import ScoreClient

logger = logging.getLogger(__name__)


class BlenderGymRubric(vf.Rubric):
    """CLIP-similarity reward + format/render/code diagnostics."""

    def __init__(
        self,
        score_service_url: str = "http://localhost:8421",
        parser: vf.Parser | None = None,
        artifact_manager: ArtifactManager | None = None,
    ) -> None:
        if artifact_manager is None:
            raise TypeError("artifact_manager is required")
        super().__init__(parser=parser)
        self.score_client = ScoreClient(score_service_url)
        self.artifact_manager = artifact_manager

        self.add_reward_func(self.clip_similarity, weight=1.0)
        self.add_metric(self.xml_parse_success)
        self.add_metric(self.render_success)
        self.add_metric(self.code_non_empty)

    async def clip_similarity(
        self,
        state: vf.State,
        info: dict[str, Any],
    ) -> float:
        """Return the CLIP cosine similarity between the latest render and the goal.

        Returns 0.0 when there's nothing to compare (XML parse failure,
        render failure, or missing goal).
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

        reward = await self.score_client.score(str(last_render), str(goal))
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

    async def close(self) -> None:
        await self.score_client.close()

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

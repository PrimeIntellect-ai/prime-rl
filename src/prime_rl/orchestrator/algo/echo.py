from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from prime_rl.configs.algorithm import AdvantageConfig, EchoAdvantageConfig
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    import verifiers as vf
    from renderers.base import Renderer

    from prime_rl.utils.client import InferencePool


def _prompt_role_weights(tokens: dict[str, Any], role_weights: dict[str, float]) -> list[float]:
    """Per-token echo weights over one step's prompt tokens.

    Each token gets its message role's weight (0.0 for unselected roles), via
    the renderer's per-token attribution — message content bodies when the
    renderer provides ``is_content``, whole messages otherwise."""
    attribution = tokens.get("prompt_attribution")
    if attribution is None:
        raise ValueError(
            "echo selects env-provided tokens by message role, which needs the renderer's "
            "per-token attribution — MITO rollouts don't carry it; set orchestrator.renderer."
        )

    # Serialized steps carry the attribution as a dict of RenderedTokens
    # fields; in-process steps may carry the dataclass itself.
    def field(key: str) -> Any:
        return attribution.get(key) if isinstance(attribution, dict) else getattr(attribution, key, None)

    indices = field("message_indices")
    roles = field("message_roles")
    is_content = field("is_content") or []
    weights = []
    for k in range(len(tokens["prompt_ids"])):
        idx = indices[k]
        selected = idx >= 0 and roles[idx] in role_weights
        if selected and is_content:
            selected = bool(is_content[k])
        weights.append(role_weights[roles[idx]] if selected else 0.0)
    return weights


class EchoAlgorithm(GRPOAlgorithm):
    """GRPO on action tokens, plus weighted CE on env-provided tokens of
    later turns (tool output, user feedback), selected by message role —
    tool-response bodies at the vetted default. Selected tokens feed the
    ``ce`` loss component at their role's ``alpha`` and stay outside the rl
    mask and its denominator. An optional user filter narrows the selection
    per rollout (e.g. dropping tool-output warnings)."""

    def __init__(self, advantage: AdvantageConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(advantage, policy_pool, renderer)
        assert isinstance(advantage, EchoAdvantageConfig)
        self.role_weights = {
            role: role_config.alpha
            for role in ("system", "user", "assistant", "tool")
            if (role_config := getattr(advantage.roles, role)) is not None
        }
        self.filter_fn: Callable[..., list[list[bool]]] | None = None
        if advantage.filter is not None:
            self.filter_fn = partial(import_object(advantage.filter.import_path), **advantage.filter.kwargs)

    def observation_weights(self, output: vf.RolloutOutput) -> list[list[float]]:
        """Each step's prompt tokens get their message role's weight; a step's
        own completion tokens are actions, not observations (0.0). The user
        filter narrows the selection. ``interleave_rollout`` slices the spans
        that actually land as observations in the merged samples."""
        trajectory = output["trajectory"]
        filter_masks = self._filter_masks(output) if self.filter_fn is not None else None
        per_step = []
        for step_idx, step in enumerate(trajectory):
            tokens = step["tokens"]
            weights = _prompt_role_weights(tokens, self.role_weights)
            weights.extend([0.0] * len(tokens["completion_ids"]))
            if filter_masks is not None:
                mask = filter_masks[step_idx]
                weights = [weight if keep else 0.0 for weight, keep in zip(weights, mask)]
            per_step.append(weights)
        return per_step

    def _filter_masks(self, output: vf.RolloutOutput) -> list[list[bool]]:
        """Invoke the user echo filter and validate its shape: one keep-mask
        per trajectory step, each spanning that step's ``prompt_ids`` +
        ``completion_ids``."""
        assert self.filter_fn is not None
        trajectory = output["trajectory"]
        masks = self.filter_fn(output)
        if not isinstance(masks, list) or len(masks) != len(trajectory):
            got = len(masks) if isinstance(masks, list) else type(masks).__name__
            raise ValueError(
                f"echo filter must return one keep-mask per trajectory step: got {got}, expected {len(trajectory)}"
            )
        for step_idx, (step, mask) in enumerate(zip(trajectory, masks)):
            tokens = step["tokens"]
            expected = len(tokens["prompt_ids"]) + len(tokens["completion_ids"])
            if not isinstance(mask, list) or len(mask) != expected:
                got = len(mask) if isinstance(mask, list) else type(mask).__name__
                raise ValueError(
                    f"echo filter mask for step {step_idx} must span the step's prompt+completion "
                    f"tokens: got {got}, expected {expected}"
                )
        return masks

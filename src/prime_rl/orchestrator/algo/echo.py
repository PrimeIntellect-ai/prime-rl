from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from prime_rl.configs.algorithm import AdvantageConfig, EchoAdvantageConfig
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    import verifiers as vf
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import RolloutView
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

    def score_rollout(self, rollout: RolloutView) -> None:
        # Observation weighting is rollout-local; the group-relative GRPO
        # baseline is inherited unchanged as ``score_group``.
        self._weight_observations(rollout)

    def _weight_observations(self, rollout: RolloutView) -> None:
        """Write each sample's ``ce_weights`` stream for the env-provided
        observation spans interleaving recorded (``obs_spans``): each token
        gets its message role's weight, narrowed by the optional user filter.
        The selected tokens stay outside ``completion_mask``, so ce is the
        only component that trains them. Step attribution is looked up
        lazily — only steps whose prompt tokens actually landed as
        observations are computed; samples where nothing is selected ship no
        ce stream at all."""
        trajectory = rollout.raw["trajectory"]
        filter_masks = self._filter_masks(rollout.raw) if self.filter_fn is not None else None
        step_weights: dict[int, list[float]] = {}
        for sample in rollout.samples:
            if not sample.obs_spans:
                continue
            weights = [0.0] * len(sample.completion_ids)
            for start, step_idx, step_start, length in sample.obs_spans:
                if step_idx not in step_weights:
                    prompt_weights = _prompt_role_weights(trajectory[step_idx]["tokens"], self.role_weights)
                    if filter_masks is not None:
                        # Masks span the step's prompt+completion; obs spans
                        # only ever come from the prompt part.
                        prompt_weights = [w if keep else 0.0 for w, keep in zip(prompt_weights, filter_masks[step_idx])]
                    step_weights[step_idx] = prompt_weights
                weights[start : start + length] = step_weights[step_idx][step_start : step_start + length]
            if any(weights):
                sample.ce_weights = [0.0] * len(sample.prompt_ids) + weights

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

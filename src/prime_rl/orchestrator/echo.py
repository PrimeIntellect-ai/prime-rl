from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import verifiers as vf

from prime_rl.configs.losses import EchoLossConfig


@dataclass(frozen=True)
class EchoAnnotations:
    step_alpha: list[list[float | None]]

    def initial_sample_alpha(self, step_idx: int) -> list[float | None] | None:
        alpha = self.step_alpha[step_idx]
        return list(alpha) if any(a is not None for a in alpha) else None

    def extension_alpha(self, step_idx: int, prefix_len: int, prompt_len: int) -> list[float | None]:
        alpha = self.step_alpha[step_idx]
        return alpha[prefix_len:prompt_len] + alpha[prompt_len:]


def build_echo_annotations(
    rollout: vf.RolloutOutput,
    echo_config: EchoLossConfig | None,
    filter_fns: Sequence[Callable[..., list[list[bool]]]] = (),
) -> EchoAnnotations | None:
    if echo_config is None:
        return None

    trajectory = rollout["trajectory"]
    step_tokens = []
    for step in trajectory:
        tokens = step["tokens"]
        if tokens is None:
            return None
        step_tokens.append(tokens)

    filter_masks = _apply_echo_filters(rollout, filter_fns) if filter_fns and trajectory else None
    return EchoAnnotations(
        step_alpha=[
            _build_step_echo_alpha(
                prompt_attribution=tokens.get("prompt_attribution"),
                prompt_len=len(tokens["prompt_ids"]),
                completion_len=len(tokens["completion_ids"]),
                echo_config=echo_config,
                filter_mask=filter_masks[step_idx] if filter_masks is not None else None,
            )
            for step_idx, tokens in enumerate(step_tokens)
        ]
    )


def _build_step_echo_alpha(
    prompt_attribution: dict | None,
    prompt_len: int,
    completion_len: int,
    echo_config: EchoLossConfig | None,
    filter_mask: list[bool] | None = None,
) -> list[float | None]:
    expected_total_len = prompt_len + completion_len
    out: list[float | None] = [None] * expected_total_len
    if echo_config is not None:
        if echo_config.assistant is not None:
            out[prompt_len:expected_total_len] = [echo_config.assistant.alpha] * completion_len

        if prompt_attribution is not None:
            message_roles = prompt_attribution.get("message_roles")
            message_indices = prompt_attribution.get("message_indices")
            is_content = prompt_attribution.get("is_content")
            if message_roles is not None and is_content and message_indices:
                if len(is_content) == prompt_len and len(message_indices) == prompt_len:
                    role_alphas = {
                        "system": echo_config.system.alpha if echo_config.system is not None else None,
                        "user": echo_config.user.alpha if echo_config.user is not None else None,
                        "assistant": echo_config.assistant.alpha if echo_config.assistant is not None else None,
                    }
                    tool_config = echo_config.tool
                    tool_alpha = tool_config.alpha if tool_config is not None else None
                    enabled_tools = tool_config.tool_names if tool_config is not None else None
                    message_tool_names = prompt_attribution.get("message_tool_names") or []

                    for k, mi in enumerate(message_indices):
                        if mi < 0 or not is_content[k] or mi >= len(message_roles):
                            continue
                        role = message_roles[mi]
                        if role == "tool":
                            tool_name = message_tool_names[mi] if mi < len(message_tool_names) else None
                            if tool_alpha is not None and (enabled_tools is None or tool_name in enabled_tools):
                                out[k] = tool_alpha
                            continue

                        alpha = role_alphas.get(role)
                        if alpha is not None:
                            out[k] = alpha

    if filter_mask is not None:
        out = [alpha if keep else None for alpha, keep in zip(out, filter_mask, strict=True)]

    return out


def _apply_echo_filters(
    rollout: vf.RolloutOutput,
    filter_fns: Sequence[Callable[..., list[list[bool]]]],
) -> list[list[bool]]:
    """Apply each custom filter (validated) and intersect (AND) their per-step token masks."""
    masks = [apply_echo_filter(rollout, fn) for fn in filter_fns]
    combined = masks[0]
    for other in masks[1:]:
        combined = [[a and b for a, b in zip(cs, os, strict=True)] for cs, os in zip(combined, other, strict=True)]
    return combined


def apply_echo_filter(
    rollout: vf.RolloutOutput,
    filter_fn: Callable[..., list[list[bool]]],
) -> list[list[bool]]:
    trajectory = rollout["trajectory"]
    result = filter_fn(rollout)

    if not isinstance(result, list):
        raise TypeError(f"echo filter must return list[list[bool]], got {type(result).__name__}")
    if len(result) != len(trajectory):
        raise ValueError(
            f"echo filter returned {len(result)} per-step masks but the rollout has {len(trajectory)} trajectory steps"
        )

    for step_idx, (step, mask) in enumerate(zip(trajectory, result)):
        tokens = step["tokens"]
        prompt_len = len(tokens["prompt_ids"])
        completion_len = len(tokens["completion_ids"])
        expected = prompt_len + completion_len

        if not isinstance(mask, list):
            raise TypeError(f"echo filter step {step_idx}: mask must be a list, got {type(mask).__name__}")
        if len(mask) != expected:
            raise ValueError(
                f"echo filter step {step_idx}: mask length {len(mask)} "
                f"!= expected {expected} "
                f"(prompt_len={prompt_len}, completion_len={completion_len})"
            )
        for k, v in enumerate(mask):
            if type(v) is not bool:
                raise TypeError(
                    f"echo filter step {step_idx}: mask[{k}] must be a plain bool, got {type(v).__name__} ({v!r})"
                )

    return result

"""Convert v1 traces into trainer samples.

The base branch keeps the existing algorithm-class transport shape while moving
the env/orchestrator boundary to `vf.Trace`. Each trace branch becomes one
`TrainingSample`; sampled tokens are selected by `completion_mask`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import verifiers.v1 as vf

from prime_rl.transport import TrainingSample
from prime_rl.transport.types import EncodedTensor, RoutedExperts
from prime_rl.utils.logger import get_logger


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value)


def _encode_mm_kwargs(mm_items: dict[str, list[dict]]) -> dict[str, EncodedTensor] | None:
    bins: dict[str, list[np.ndarray]] = {}
    for items in mm_items.values():
        for item in items:
            for key, value in item.items():
                bins.setdefault(key, []).append(_to_numpy(value))
    encoded: dict[str, EncodedTensor] = {}
    for key, arrays in bins.items():
        arr = np.concatenate(arrays, axis=0)
        encoded[key] = EncodedTensor(dtype=str(arr.dtype), shape=list(arr.shape), data=arr.tobytes())
    return encoded or None


def _encode_routed_experts(arr: np.ndarray | None, num_tokens: int) -> RoutedExperts | None:
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    if arr.shape[0] > num_tokens:
        arr = arr[:num_tokens]
    elif arr.shape[0] < num_tokens:
        padding = np.zeros((num_tokens - arr.shape[0], *arr.shape[1:]), dtype=arr.dtype)
        arr = np.concatenate([arr, padding], axis=0)
    return RoutedExperts(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def trace_to_samples(
    trace: vf.Trace,
    *,
    env_name: str = "",
    mm_token_type_ids_mapping: dict[int, int] | None = None,
) -> list[TrainingSample]:
    """Build one training sample per trace branch.

    The old algo-branch transport splits samples into prompt/completion arrays.
    A trace branch is already a flat sequence, so the prompt side stays empty
    and the full branch lives in `completion_ids`; `completion_mask` carries
    the real sampled-token eligibility.
    """
    samples: list[TrainingSample] = []
    for branch in trace.branches:
        token_ids = branch.token_ids
        if not token_ids:
            continue
        sampled_mask = branch.sampled_mask
        if not any(sampled_mask):
            continue

        mm_kwargs: dict[str, EncodedTensor] | None = None
        mm_token_type_ids: list[int] | None = None
        mmd = branch.multi_modal_data
        if mmd is not None:
            mm_kwargs = _encode_mm_kwargs(mmd.mm_items)
            mapping = mm_token_type_ids_mapping or {}
            mm_token_type_ids = [mapping.get(token_id, 0) for token_id in token_ids]

        samples.append(
            TrainingSample(
                prompt_ids=[],
                prompt_mask=[],
                completion_ids=token_ids,
                completion_mask=[sampled and not trace.has_error for sampled in sampled_mask],
                completion_logprobs=branch.logprobs,
                completion_temperatures=[],
                ref_logprobs=None,
                env_name=env_name,
                mm_kwargs=mm_kwargs,
                mm_token_type_ids=mm_token_type_ids,
                routed_experts=_encode_routed_experts(branch.routed_experts, len(token_ids)),
            )
        )

    if not samples:
        get_logger().warning(
            f"No trainable samples (error={trace.has_error}, stop={trace.stop_condition}, num_turns={trace.num_turns})."
        )
    return samples

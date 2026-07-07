"""Turn a v1 `Trace` (the env server's native, typed output) into training data.

The orchestrator holds a real `vf.Trace` (validated in `envs.py`), so everything here is
attribute access — no dicts. The trace is a message graph (`trace.nodes`); each `trace.branches`
entry (a root→leaf path) is first-class and carries its own flat token sequence
(`branch.token_ids` / `branch.sampled_mask` / `branch.logprobs`), so a branch yields one
training sample directly. Token-length readers (`completion_len`, `total_tokens`, `num_turns`)
live on `vf.Trace` itself.

Training is renderer-only across every mode (RL/OPD student, SFT teacher), so every node
always carries its tokens — no backfill needed. For multimodal rollouts the branch also carries
the images it introduced (`branch.multi_modal_data`), rebuilt here into the flat `mm_kwargs` /
`mm_token_type_ids` the trainer forwards.
"""

from __future__ import annotations

import numpy as np
import verifiers.v1 as vf

from prime_rl.transport import TrainingSample
from prime_rl.transport.types import EncodedTensor, RoutedExperts
from prime_rl.utils.logger import get_logger


def _to_numpy(val) -> np.ndarray:
    """A renderer mm item value (torch tensor or numpy array) -> a contiguous numpy array."""
    if hasattr(val, "detach"):  # torch tensor
        val = val.detach().cpu().numpy()
    return np.ascontiguousarray(val)


def _encode_mm_kwargs(mm_items: dict[str, list[dict]]) -> dict[str, EncodedTensor] | None:
    """Concatenate the branch's per-image renderer items into the flat `mm_kwargs` the trainer
    forwards — one `EncodedTensor` per kwarg key (e.g. `pixel_values`, `image_grid_thw`), images
    cat'd along dim 0 in branch token order. Model-agnostic: the keys are whatever the processor
    emits. Returns None when there are no items."""
    bins: dict[str, list[np.ndarray]] = {}
    for items in mm_items.values():  # per modality
        for item in items:  # per image
            for key, val in item.items():
                bins.setdefault(key, []).append(_to_numpy(val))
    encoded: dict[str, EncodedTensor] = {}
    for key, arrs in bins.items():
        arr = np.concatenate(arrs, axis=0)
        encoded[key] = EncodedTensor(dtype=str(arr.dtype), shape=list(arr.shape), data=arr.tobytes())
    return encoded or None


def _encode_routed_experts(arr: np.ndarray | None, num_tokens: int) -> RoutedExperts | None:
    """The branch's router-replay array (`[tokens, layers, top_k]`) -> the transport
    `RoutedExperts` the trainer replays. Defensively realigns the token axis to `num_tokens`
    (the trainer asserts `routed_experts.shape[0] == len(token_ids)`): truncate if longer,
    zero-pad the tail if shorter. `Branch.routed_experts` already guarantees alignment, so this
    is a backstop."""
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    if arr.shape[0] > num_tokens:
        arr = arr[:num_tokens]
    elif arr.shape[0] < num_tokens:
        pad = np.zeros((num_tokens - arr.shape[0], *arr.shape[1:]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return RoutedExperts(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def _ttt_adapter_path(trace: vf.Trace, branch: vf.Branch) -> str | None:
    """Resolve the branch's TTT adapter checkpoint from the trace's stamps: version 0 (or an
    un-stamped rollout) sampled from the base model — no adapter; version k maps to the TTT
    service's checkpoint dir for this rollout (recorded per update in ``trace.info["ttt"]``).
    ``Branch.ttt_version`` enforces the one-version-per-branch invariant (raises on a mix).
    A stamped branch whose checkpoint path is missing from the info records is a hard error —
    replaying it against the base model would silently corrupt the importance ratio."""
    version = branch.ttt_version
    if not version:  # None (no TTT) or 0 (base model)
        return None
    for update in trace.info.get("ttt", {}).get("updates", []):
        if update.get("version") == version:
            path = update.get("ckpt_path")
            if path:
                return str(path)
            break
    raise ValueError(
        f"branch {branch.index} of rollout {trace.id} was sampled under TTT adapter "
        f"version {version}, but no checkpoint path is recorded in trace.info['ttt'] — "
        "cannot build an exact replay sample."
    )


def qa_recycle_samples(trace: vf.Trace, tokenizer, env_name: str = "") -> list[TrainingSample]:
    """Build ce-routed training samples from the rollout's recorded TTT Q&A pairs — the
    "recycle the Q&A compute into a permanent weight update" step: each pair (text, from
    ``trace.info["ttt"]``) is rendered standalone with the policy tokenizer's chat template
    — conditioned on the rollout's system prompt + tool schemas (the same frame the adapter
    training used), loss-masked — and routed entirely to the **ce** loss component (answer
    tokens; ``rl_weights`` all zero, no advantages), so it rides the same training batch as
    the RL samples without touching the policy-gradient math. QA samples carry no adapter
    ref — they train the live policy weights."""
    ttt_info = trace.info.get("ttt", {})
    system_prompt = ttt_info.get("system_prompt")
    tools = ttt_info.get("tools")
    head = [{"role": "system", "content": system_prompt}] if system_prompt else []
    template_kwargs: dict = {"tools": tools} if tools else {}
    samples: list[TrainingSample] = []
    for update in ttt_info.get("updates", []):
        for pair in update.get("qa_pairs") or []:
            question = str(pair.get("question", ""))
            answer = str(pair.get("answer", ""))
            if not answer.strip():
                continue
            conversation = [
                *head,
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            full = tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=False, **template_kwargs
            )
            full = list(full["input_ids"] if not isinstance(full, list) else full)
            prompt = tokenizer.apply_chat_template(
                conversation[:-1], tokenize=True, add_generation_prompt=True, **template_kwargs
            )
            prompt = list(prompt["input_ids"] if not isinstance(prompt, list) else prompt)
            prompt_len = len(prompt) if full[: len(prompt)] == prompt else 0
            answer_len = len(full) - prompt_len
            if answer_len < 1:
                continue
            mask = [False] * prompt_len + [True] * answer_len
            samples.append(
                TrainingSample(
                    token_ids=full,
                    mask=mask,
                    logprobs=[0.0] * len(full),  # ce is masked NLL — no importance ratio
                    temperatures=[],  # filled by TrainSink.process_group
                    env_name=env_name,
                    rl_weights=[0.0] * len(full),
                    ce_weights=[1.0 if m else 0.0 for m in mask],
                )
            )
    return samples


def trace_to_samples(
    trace: vf.Trace,
    *,
    env_name: str = "",
    mm_token_type_ids_mapping: dict[int, int] | None = None,
) -> list[TrainingSample]:
    """Convert a v1 `Trace` into `TrainingSample`s — one per branch.

    Each `trace.branches` entry is already a flat token sequence (`branch.token_ids` /
    `branch.sampled_mask` / `branch.logprobs`), so a sample carries it directly: `mask` marks
    the trainable (model-sampled) tokens, the context tokens between completions stay masked
    out. Errored rollouts are dropped upstream (`TrainSink.process_rollout`), so no error
    handling happens here. A branch carrying images also gets `mm_kwargs` (the concatenated
    pixel tensors) and `mm_token_type_ids` (the renderer's `mm_token_type_id_map` applied to
    the branch tokens). Branches with no sampled tokens (e.g. an openai client carrying none)
    yield nothing.

    A sampled node is trainable exactly ONCE across the trace: branches sharing a sampled
    prefix (TTT Q&A side-generations fork off the trajectory's leaf; subagent forks) re-carry
    those tokens as pure context (mask False) in every branch after the first, so shared
    tokens never receive N× gradient weight. Node order follows `trace.nodes` (creation
    order), so the "first" branch containing a node is deterministic.
    """
    samples: list[TrainingSample] = []
    trained_nodes: set[int] = set()  # id(node) of sampled nodes already granted their mask
    for branch in trace.branches:
        mask: list[bool] = []
        for node in branch.nodes:
            if node.sampled and any(node.mask):
                if id(node) in trained_nodes:
                    mask.extend([False] * len(node.mask))  # context here; trained elsewhere
                else:
                    trained_nodes.add(id(node))
                    mask.extend(node.mask)
            else:
                mask.extend(node.mask)
        if not any(mask):
            continue
        token_ids = branch.token_ids
        mm_kwargs: dict[str, EncodedTensor] | None = None
        mm_token_type_ids: list[int] | None = None
        mmd = branch.multi_modal_data
        if mmd is not None:
            mm_kwargs = _encode_mm_kwargs(mmd.mm_items)
            mapping = mm_token_type_ids_mapping or {}
            mm_token_type_ids = [mapping.get(t, 0) for t in token_ids]
        samples.append(
            TrainingSample(
                token_ids=token_ids,
                mask=mask,
                logprobs=branch.logprobs,
                temperatures=[],  # filled by TrainSink.process_group
                env_name=env_name,
                mm_kwargs=mm_kwargs,
                mm_token_type_ids=mm_token_type_ids,
                routed_experts=_encode_routed_experts(branch.routed_experts, len(token_ids)),
                ttt_adapter_path=_ttt_adapter_path(trace, branch),
            )
        )
    if not samples:
        get_logger().warning(
            f"No trainable samples (error={trace.has_error}, stop={trace.stop_condition}, num_turns={trace.num_turns})."
        )
    return samples

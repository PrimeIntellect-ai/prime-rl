"""Turn a v1 `Trace` (the env server's native, typed output) into training data.

The orchestrator holds a real `vf.Trace` (validated in `envs.py`), so everything
here is attribute access — no dicts. The trajectory is segmented into `branches`
(recomputed on the consumer; a branch is a maximal linear run of turns whose context
only grew), each turn carrying `tokens` (`prompt_ids` / `completion_ids` /
`completion_logprobs`, from the renderer client). Token-length readers
(`completion_len`, `total_tokens`, `has_response`) live on `vf.Trace` itself.
"""

from __future__ import annotations

import verifiers.v1 as vf
from verifiers.v1.clients.openai import message_to_wire

from prime_rl.transport import TrainingSample
from prime_rl.transport.types import EncodedTensor
from prime_rl.utils.chat_template import (
    common_prefix_len,
    deserialize_tool_calls,
    normalize_messages,
    render_messages,
    strip_message_content,
)
from prime_rl.utils.logger import get_logger


def backfill_rollout_tokens(trace: vf.Trace, tokenizer) -> None:
    """Populate per-turn ``tokens`` for turns the env returned without them — e.g. SFT
    against an external teacher whose chat client carries no token ids. Renders each
    turn's prompt + assistant response with the student chat template and splits on the
    longest common prefix; masks/logprobs are filled by ``trace_to_samples``. Mutates the
    trace in place. (Tools aren't re-supplied here, so this targets text turns.)
    """
    for turn in trace.trajectory:
        if turn.tokens is not None:
            continue
        prompt = strip_message_content(
            deserialize_tool_calls(normalize_messages([message_to_wire(m) for m in turn.prompt], "user"))
        )
        completion = strip_message_content(
            deserialize_tool_calls(normalize_messages([message_to_wire(turn.response.message)], "assistant"))
        )
        prompt_ids = render_messages(tokenizer, prompt, add_generation_prompt=True)
        full_ids = render_messages(tokenizer, prompt + completion)
        split_idx = common_prefix_len(prompt_ids, full_ids)
        turn.tokens = vf.TurnTokens(
            prompt_ids=full_ids[:split_idx],
            completion_ids=full_ids[split_idx:],
            completion_logprobs=[0.0] * (len(full_ids) - split_idx),
        )


def _decode_wire_tensor(wt: vf.WireTensor):
    import base64

    import numpy as np
    import torch

    arr = np.frombuffer(base64.b64decode(wt.data), dtype=np.dtype(wt.dtype)).reshape(wt.shape)
    return torch.from_numpy(arr.copy())


def _pack_mm_kwargs(mm_list: list[vf.MMData]) -> dict[str, EncodedTensor] | None:
    """Union each turn's *new* images into model-agnostic `mm_kwargs`: concat each
    HF-processor kwarg (e.g. `pixel_values`, `image_grid_thw`) in turn order. The model's
    `forward` signature is the schema, so image/video keys don't clash.

    The stitched ids carry each image's placeholder tokens once (in the turn it's
    introduced), so we contribute each image once too. A turn's `multi_modal_data` may be
    *cumulative* (the renderer re-rendered the whole prompt → every image so far, native v1)
    or *delta* (only the turn's new images, the v0 bridge); a turn is cumulative iff its
    hashes restate everything taken so far, so we take only the appended tail. Identical
    images in different turns (e.g. two squares of the same color) keep distinct slots —
    matched by position, not deduped by hash."""
    import torch

    per_kwarg: dict[str, list] = {}
    taken: dict[str, list] = {}  # modality -> hashes contributed so far, in order
    for mm in mm_list:
        for modality, items in mm.mm_items.items():
            hashes = list(mm.mm_hashes.get(modality) or [None] * len(items))
            acc = taken.setdefault(modality, [])
            if None not in hashes and hashes[: len(acc)] == acc:
                start = len(acc)  # cumulative turn: skip the restated prefix
                acc[:] = hashes
            else:
                start = 0  # delta turn: all images are new
                acc.extend(hashes)
            for item in items[start:]:
                for key, wt in item.items():
                    per_kwarg.setdefault(key, []).append(_decode_wire_tensor(wt))
    if not per_kwarg:
        return None
    out: dict[str, EncodedTensor] = {}
    for key, tensors in per_kwarg.items():
        arr = torch.cat(tensors, dim=0).contiguous().numpy()
        out[key] = EncodedTensor(dtype=str(arr.dtype), shape=list(arr.shape), data=arr.tobytes())
    return out


def trace_to_samples(
    trace: vf.Trace, *, env_name: str = "", mm_token_type_ids_mapping: dict[int, int] | None = None
) -> list[TrainingSample]:
    """Convert a v1 `Trace` into `TrainingSample`s — one per branch.

    Stitch each branch's turns into one token sequence: the branch's first-turn
    prompt, then for each turn the new context tokens since the running prefix (mask
    `False`, the model didn't generate them) followed by that turn's completion tokens
    (mask `True`, with logprobs). On a rollout error the whole completion is masked
    out. Branches whose turns carry no token ids (e.g. an openai client) yield nothing.
    """
    has_error = trace.has_error
    samples: list[TrainingSample] = []
    for branch in trace.branches:
        turns = branch.turns
        if not turns or any(turn.tokens is None for turn in turns):
            continue

        prompt_ids = list(turns[0].tokens.prompt_ids)
        completion_ids: list[int] = []
        completion_mask: list[bool] = []
        completion_logprobs: list[float] = []
        prefix_len = len(prompt_ids)  # running prompt+completion length of the branch

        for turn in turns:
            tokens = turn.tokens
            new_prompt = list(tokens.prompt_ids[prefix_len:])
            completion_ids.extend(new_prompt)
            completion_mask.extend([False] * len(new_prompt))
            completion_logprobs.extend([0.0] * len(new_prompt))

            turn_completion = list(tokens.completion_ids)
            completion_ids.extend(turn_completion)
            completion_mask.extend([not has_error] * len(turn_completion))
            logprobs = list(tokens.completion_logprobs)
            if len(logprobs) != len(turn_completion):
                logprobs = (logprobs + [0.0] * len(turn_completion))[: len(turn_completion)]
            completion_logprobs.extend(logprobs)

            prefix_len = len(tokens.prompt_ids) + len(turn_completion)

        if not completion_ids:
            continue
        # Multimodal: union each turn's image/video tensors into mm_kwargs, and tag every
        # token with its modality (image-placeholder vs text) via the renderer's map.
        mm = [t.tokens.multi_modal_data for t in turns if t.tokens and t.tokens.multi_modal_data]
        mm_kwargs = _pack_mm_kwargs(mm) if mm else None
        mm_token_type_ids = None
        if mm_kwargs is not None and mm_token_type_ids_mapping:
            mm_token_type_ids = [mm_token_type_ids_mapping.get(tid, 0) for tid in prompt_ids + completion_ids]
        samples.append(
            TrainingSample(
                prompt_ids=prompt_ids,
                prompt_mask=[False] * len(prompt_ids),
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                completion_logprobs=completion_logprobs,
                completion_temperatures=[],  # filled by TrainSink.process_group
                teacher_logprobs=None,
                advantage=None,
                env_name=env_name,
                mm_kwargs=mm_kwargs,
                mm_token_type_ids=mm_token_type_ids,
            )
        )
    if not samples:
        get_logger().warning(
            f"No trainable samples (error={has_error}, stop={trace.stop_condition}, num_turns={trace.num_turns})."
        )
    return samples

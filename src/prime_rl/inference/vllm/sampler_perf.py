"""Hot-path fixes for two O(batch x output_len) per-step CPU costs in vLLM 0.22.0.

Both costs collapse decode throughput ~5x for long generations with the Qwen3.5
recommended sampling defaults (presence_penalty=1.5) plus thinking_token_budget:

1. Penalties: ``vllm/v1/sample/ops/penalties.py`` rebuilds a padded
   ``[B, output_len]`` int64 tensor from Python lists via ``make_tensor_with_pad``
   on every decode step (~80ms/step at B=128, output 12k). Upstream's rework
   lives only in Model Runner V2, which rejects ``reasoning_config`` and
   ``enable_return_routed_experts``, so it is unusable here. We replace
   ``Sampler.apply_penalties`` with a vectorized builder that slices the
   already-materialized ``InputBatch.token_ids_cpu`` numpy buffer into a
   reusable pinned staging tensor (no per-token Python iteration). Any state
   we do not recognize (speculative-decode combined rows, foreign batches)
   falls back to the upstream implementation, so semantics are identical by
   construction.

   Under async scheduling, ``token_ids_cpu`` output positions are written as
   ``-1`` placeholders and never repaired (only the Python lists are). We
   vendor ``InputBatch.update_async_output_token_ids`` with a write-back so
   the numpy buffer stays authoritative; residual ``-1`` rows (kv-load
   discards, unrepaired rows) are masked to the pad bin exactly like
   upstream's ``masked_fill_``.

2. Thinking budget: ``ThinkingBudgetStateHolder._update_think_state`` rescans
   the entire generated output for the think-end token ids on every step until
   they appear (O(L) pure Python per request per step for the whole thinking
   phase). We wrap the method with an incremental watermark scan: only tokens
   generated since the last scan are searched. ``end_thinking`` uses a ``-2``
   sentinel for "scanned, not found": every downstream read in the original
   (`== -1` scan guards, ``> -1``, ``>= 0``) treats -2 exactly like -1, while
   the guard that triggers the full rescan only fires on -1. When the start
   tokens are absent we replicate the original's early return (scan-start ->
   scan-end -> return) without calling it.

Apply via :func:`apply_sampler_perf_patches` from the vLLM general plugin so
every engine/worker process gets patched. Kill switch:
``PRIME_RL_DISABLE_SAMPLER_PERF_PATCH=1``.
"""

import os
import weakref

import numpy as np
import torch

SUPPORTED_VLLM = "0.22.0"

# Weakref to the live InputBatch of this process (captured at construction).
_INPUT_BATCH_REF: weakref.ref | None = None


# ---------------------------------------------------------------------------
# Patch 1: penalties
# ---------------------------------------------------------------------------


class _PinnedStaging:
    """Double-buffered pinned staging for the [B, max_out_len] token tensor."""

    def __init__(self, max_rows: int, max_cols: int):
        numel = max_rows * max_cols
        self._bufs = [torch.empty(numel, dtype=torch.int64, pin_memory=torch.cuda.is_available()) for _ in range(2)]
        self._events = [torch.cuda.Event(), torch.cuda.Event()]
        self._recorded = [False, False]
        self._idx = 0

    def get(self, rows: int, cols: int) -> tuple[torch.Tensor, int]:
        i = self._idx
        self._idx ^= 1
        if self._recorded[i]:
            # The previous H2D copy from this buffer must complete before the
            # CPU overwrites it (non_blocking copies read pinned memory async).
            self._events[i].synchronize()
        return self._bufs[i][: rows * cols].view(rows, cols), i

    def record(self, i: int) -> None:
        self._events[i].record()
        self._recorded[i] = True


_staging: _PinnedStaging | None = None


def _capture_input_batch() -> None:
    from vllm.v1.worker.gpu_input_batch import InputBatch

    orig_init = InputBatch.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        global _INPUT_BATCH_REF, _staging
        _INPUT_BATCH_REF = weakref.ref(self)
        _staging = _PinnedStaging(self.max_num_reqs, self.max_model_len)

    InputBatch.__init__ = patched_init


def _patch_async_output_writeback() -> None:
    """Vendor InputBatch.update_async_output_token_ids (vLLM 0.22.0,
    gpu_input_batch.py) + write repaired ids back into token_ids_cpu."""
    from vllm.v1.worker.gpu_input_batch import InputBatch

    def update_async_output_token_ids(self) -> None:
        output_token_ids = self.sampling_metadata.output_token_ids
        if self.sampled_token_ids_cpu is None or not output_token_ids:
            return

        assert self.prev_req_id_to_index is not None
        sampled_token_ids = None
        for index, req_id in enumerate(self.req_ids):
            prev_index = self.prev_req_id_to_index.get(req_id)
            if prev_index is None:
                continue
            req_output_token_ids = output_token_ids[index]
            if not req_output_token_ids or req_output_token_ids[-1] != -1:
                continue
            if sampled_token_ids is None:
                assert self.async_copy_ready_event is not None
                self.async_copy_ready_event.synchronize()
                sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            new_ids: list[int] = sampled_token_ids[prev_index]
            if not new_ids:
                continue
            num_sampled_ids = len(new_ids) if new_ids[-1] != -1 else new_ids.index(-1)
            first_placeholder = len(req_output_token_ids)
            while first_placeholder > 0 and req_output_token_ids[first_placeholder - 1] == -1:
                first_placeholder -= 1
            num_placeholders = len(req_output_token_ids) - first_placeholder
            num_to_replace = min(num_sampled_ids, num_placeholders)
            del new_ids[num_to_replace:]
            req_output_token_ids[first_placeholder:] = new_ids
            # prime-rl addition: keep token_ids_cpu authoritative under async
            # scheduling so the fast penalties path can slice it.
            start = int(self.num_prompt_tokens[index]) + first_placeholder
            self.token_ids_cpu[index, start : start + len(new_ids)] = new_ids

    InputBatch.update_async_output_token_ids = update_async_output_token_ids


def build_output_tokens_fast(
    input_batch,
    staging: "_PinnedStaging",
    output_token_ids: list[list[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build the [B, max_out_len] padded output-token tensor without Python
    per-token iteration. Returns None when the rows are not the live batch
    rows (caller must fall back to the upstream implementation)."""
    n = len(output_token_ids)
    req_lists = input_batch.req_output_token_ids
    if len(req_lists) < n:
        return None
    for i in range(n):
        if output_token_ids[i] is not req_lists[i]:
            return None
    if n == 0:
        return torch.empty(0, 0, dtype=torch.int64, device=device)

    out_lens = np.fromiter(map(len, output_token_ids), np.int64, n)
    max_len = int(out_lens.max())
    if max_len == 0:
        return torch.empty(n, 0, dtype=torch.int64, device=device)

    buf, buf_idx = staging.get(n, max_len)
    dst = buf.numpy()
    dst.fill(vocab_size)
    token_ids_cpu = input_batch.token_ids_cpu
    num_prompt = input_batch.num_prompt_tokens
    for i in range(n):
        length = out_lens[i]
        if length:
            start = num_prompt[i]
            dst[i, :length] = token_ids_cpu[i, start : start + length]
    # Unrepaired placeholders / discarded rows: same semantics as upstream's
    # masked_fill_(output_tokens_t == -1, vocab_size).
    dst[dst == -1] = vocab_size
    tensor = buf.to(device, non_blocking=True)
    staging.record(buf_idx)
    return tensor


def _patch_fast_penalties() -> None:
    from vllm.model_executor.layers.utils import apply_penalties as gpu_apply_penalties
    from vllm.v1.sample.sampler import Sampler

    orig_apply_penalties = Sampler.apply_penalties

    def apply_penalties(logits, sampling_metadata, output_token_ids):
        if sampling_metadata.no_penalties:
            return logits
        input_batch = _INPUT_BATCH_REF() if _INPUT_BATCH_REF is not None else None
        tensor = None
        if input_batch is not None and _staging is not None and logits.shape[0] == len(output_token_ids):
            tensor = build_output_tokens_fast(input_batch, _staging, output_token_ids, logits.shape[1], logits.device)
        if tensor is None:
            # Unrecognized rows (e.g. spec-decode combined lists): upstream path.
            return orig_apply_penalties(logits, sampling_metadata, output_token_ids)
        assert sampling_metadata.prompt_token_ids is not None
        return gpu_apply_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            tensor,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
        )

    Sampler.apply_penalties = staticmethod(apply_penalties)


# ---------------------------------------------------------------------------
# Patch 2: thinking budget incremental scan
# ---------------------------------------------------------------------------


def find_last_in_window(lst: list[int], pattern: list[int], lo: int, hi: int) -> int:
    """Last start index of `pattern` in lst[lo:hi], or -1. C-speed for the
    common single-token pattern via list.index."""
    m = len(pattern)
    if m == 0 or hi - lo < m:
        return -1
    if m == 1:
        target = pattern[0]
        last = -1
        i = lo
        while True:
            try:
                i = lst.index(target, i, hi)
            except ValueError:
                return last
            last = i
            i += 1
    last = -1
    for i in range(lo, hi - m + 1):
        if lst[i : i + m] == pattern:
            last = i
    return last


def _patch_thinking_budget_scan() -> None:
    from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

    orig_update = ThinkingBudgetStateHolder._update_think_state

    def _update_think_state(self, state) -> None:
        if state.get("thinking_token_budget", -1) == -1 or not self.think_end_token_ids:
            return orig_update(self, state)

        out = state.get("output_tok_ids") or []
        # Watermark excludes trailing async -1 placeholders: those positions
        # are rewritten with real ids next step and must be rescanned then.
        hi = len(out)
        while hi > 0 and out[hi - 1] == -1:
            hi -= 1
        pos = state.get("_prime_scan_pos", 0)
        start_idx = state.get("start_thinking", -1)
        end_idx = state.get("end_thinking", -1)
        # Shrinkage (spec rejection / kv-load discard): rescan from scratch.
        # Caution: in the continue_thinking case, start_thinking is a
        # prompt-absolute index set at init — never treat it as shrunk.
        start_is_output_relative = not state.get("continue_thinking", False)
        if (
            pos > hi
            or (end_idx >= 0 and end_idx >= hi)
            or (start_is_output_relative and 0 <= start_idx and start_idx >= hi)
        ):
            pos = 0
            if start_is_output_relative and 0 <= start_idx and start_idx >= hi:
                start_idx = -1
                state["start_thinking"] = -1
            if end_idx >= 0 and end_idx >= hi:
                end_idx = -1
                state["end_thinking"] = -1

        if start_idx == -1:
            m = len(self.think_start_token_ids)
            lo = max(0, pos - (m - 1)) if m else 0
            idx = find_last_in_window(out, self.think_start_token_ids, lo, hi) if self.think_start_token_ids else -1
            if idx != -1:
                state["start_thinking"] = idx
                start_idx = idx
        if end_idx < 0:  # -1 (never scanned) or -2 (scanned, absent)
            m = len(self.think_end_token_ids)
            lo = max(0, pos - (m - 1))
            idx = find_last_in_window(out, self.think_end_token_ids, lo, hi)
            # -2 sentinel: skips the original's full rescan (`== -1` guard)
            # while behaving identically to -1 in every downstream comparison
            # (`> -1`, `>= 0`).
            state["end_thinking"] = idx if idx != -1 else -2
        state["_prime_scan_pos"] = hi

        if state.get("start_thinking", -1) == -1:
            # Replicate the original's early return (scan start -> scan end ->
            # `if start_thinking == -1: return`) without paying its scans.
            return None
        return orig_update(self, state)

    ThinkingBudgetStateHolder._update_think_state = _update_think_state


# ---------------------------------------------------------------------------


def apply_sampler_perf_patches() -> None:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    if os.environ.get("PRIME_RL_DISABLE_SAMPLER_PERF_PATCH", "0") == "1":
        logger.warning("Sampler perf patches disabled via PRIME_RL_DISABLE_SAMPLER_PERF_PATCH")
        return
    import vllm

    if getattr(vllm, "_prime_rl_sampler_perf_patched", False):
        return
    vllm._prime_rl_sampler_perf_patched = True

    if vllm.__version__ != SUPPORTED_VLLM:
        raise RuntimeError(
            f"sampler_perf patches are pinned to vLLM {SUPPORTED_VLLM}, found "
            f"{vllm.__version__}. Re-validate the vendored code paths "
            "(Sampler.apply_penalties, InputBatch.update_async_output_token_ids, "
            "ThinkingBudgetStateHolder._update_think_state) before bumping."
        )
    _capture_input_batch()
    _patch_async_output_writeback()
    _patch_fast_penalties()
    _patch_thinking_budget_scan()
    logger.info("Applied sampler perf patches (fast penalties tensor build + incremental thinking-budget scan)")

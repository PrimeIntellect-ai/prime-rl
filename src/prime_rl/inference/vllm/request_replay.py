"""Request-scoped replay for vLLM's V2 sampler and speculative verifier."""

import os
from contextvars import ContextVar
from copy import copy
from functools import wraps

import numpy as np
import torch

REPLAY_KEY = "prl_replay"
SAMPLER_ROWS_KEY = "prl_sampler_rows"
CAPACITY_ENV = "PRIME_SAMPLING_SUPPORT_CAPACITY"
_capture: ContextVar[list | None] = ContextVar("prl_capture", default=None)
_verified: ContextVar["_VerifiedReplay | None"] = ContextVar("prl_verified", default=None)


class _VerifiedReplay:
    """Replay flags of one verification call, consumed in request order across its chunks."""

    def __init__(self, replay: np.ndarray):
        self.replay = replay
        self.start = 0

    def take(self, num_reqs: int) -> np.ndarray:
        self.start += num_reqs
        return self.replay[self.start - num_reqs : self.start]


def wants_replay(params) -> bool:
    return params is not None and (params.extra_args or {}).get(REPLAY_KEY, False) is True


def sampler_rows(params) -> int:
    return 0 if params is None else (params.extra_args or {}).get(SAMPLER_ROWS_KEY, 0)


def patch_request_replay():
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.v1.engine.output_processor import RequestState
    from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    from prime_rl.inference.vllm.paired_capture import (
        VerifiedSupports,
        capture_supports,
        capture_verified_supports,
        pack_sampler_rows,
    )

    if getattr(Sampler, "_prime_request_replay", False):
        return

    validate = InputProcessor._validate_params

    @wraps(validate)
    def validate_params(self, params, supported_tasks):
        if isinstance(params, SamplingParams):
            rows = sampler_rows(params)
            if type(rows) is not int or not 0 <= rows <= self.model_config.get_vocab_size():
                raise VLLMValidationError(
                    "Sampler rows must be an integer between 0 and the vocabulary size",
                    parameter=SAMPLER_ROWS_KEY,
                    value=rows,
                )
            if rows and wants_replay(params):
                raise VLLMValidationError("A request captures its replayed support or sampler rows, not both")
            if self.model_config.return_sampling_mask:
                if rows and self.speculative_config is not None:
                    raise VLLMValidationError("Sampler rows do not support speculative decoding")
                if wants_replay(params) and params.temperature <= 0:
                    raise VLLMValidationError("Replay requests require temperature > 0")
                if wants_replay(params) and params.output_kind != RequestOutputKind.FINAL_ONLY:
                    raise VLLMValidationError("Replay requests do not support streaming")
                # Validate against a private view; never toggle shared engine
                # configuration while concurrent requests are being admitted.
                self = copy(self)
                self.model_config = copy(self.model_config)
                self.model_config.return_sampling_mask = False
            elif wants_replay(params) or rows:
                raise VLLMValidationError("Sampling-support capture requires enable_return_sampling_mask")
        return validate(self, params, supported_tasks)

    add_request = Sampler.add_request

    @wraps(add_request)
    def add(self, req_idx, sampling_params):
        if not hasattr(self, "_prime_capture"):
            self._prime_capture = np.zeros(self.sampling_states.max_num_reqs, dtype=bool)
            self._prime_sampler_rows = np.zeros(self.sampling_states.max_num_reqs, dtype=np.int64)
        self._prime_capture[req_idx] = wants_replay(sampling_params)
        self._prime_sampler_rows[req_idx] = sampler_rows(sampling_params)
        return add_request(self, req_idx, sampling_params)

    call = Sampler.__call__

    @wraps(call)
    def sample(self, logits, input_batch):
        if not hasattr(self, "_prime_capture"):
            return call(self, logits, input_batch)
        indices = input_batch.idx_mapping_np
        replay, num_rows = self._prime_capture[indices], self._prime_sampler_rows[indices]
        stash = []
        token = _capture.set(stash)
        try:
            output = call(self, logits, input_batch)
        finally:
            _capture.reset(token)
        if not stash or not (replay.any() or num_rows.any()):
            return output
        processed, cu_num_logits, num_sampled = stash
        keep = torch.from_numpy(replay).pin_memory().to(num_sampled.device, non_blocking=True)
        rows = None
        if num_rows.any():
            rows = pack_sampler_rows(processed, cu_num_logits, num_sampled, output.sampled_token_ids, num_rows)
        output.sampling_mask_tensors = capture_supports(
            processed, cu_num_logits, torch.where(keep, num_sampled, 0), int(os.environ[CAPACITY_ENV]), rows
        )
        return output

    verify = RejectionSampler.__call__

    @wraps(verify)
    def verify_and_capture(self, logits, input_batch, *args, **kwargs):
        replay = getattr(self.sampler, "_prime_capture", None)
        if replay is None:
            return verify(self, logits, input_batch, *args, **kwargs)
        token = _verified.set(_VerifiedReplay(replay[input_batch.idx_mapping_np]))
        try:
            return verify(self, logits, input_batch, *args, **kwargs)
        finally:
            _verified.reset(token)

    from_logits = SamplingMaskTensors.from_logits

    @classmethod
    def capture(cls, logits, cu_num_logits, num_sampled_tokens, max_num_kept, rows_per_request=1):
        verified = _verified.get()
        if verified is not None:
            replay = verified.take(num_sampled_tokens.shape[0])
            if not verified.replay.any():
                return None
            capacity = int(os.environ[CAPACITY_ENV])
            return capture_verified_supports(
                logits, cu_num_logits, num_sampled_tokens, replay, rows_per_request, capacity
            )
        stash = _capture.get()
        if stash is None:
            return from_logits(logits, cu_num_logits, num_sampled_tokens, max_num_kept, rows_per_request)
        stash.extend([logits, cu_num_logits, num_sampled_tokens])
        return None

    cat = SamplingMaskTensors.cat.__func__

    @classmethod
    def cat_chunks(cls, chunks):
        if all(chunk is None for chunk in chunks):
            return None
        if isinstance(chunks[0], VerifiedSupports):
            return VerifiedSupports.cat(chunks)
        return cat(cls, chunks)

    class DiscardChunks(list):
        def append(self, chunk):
            pass

    from_new_request = RequestState.from_new_request.__func__

    @classmethod
    @wraps(from_new_request)
    def new_request(cls, tokenizer, request, *args, **kwargs):
        state = from_new_request(cls, tokenizer, request, *args, **kwargs)
        params = request.sampling_params
        if not (wants_replay(params) or sampler_rows(params)):
            state.sampling_mask_chunks = DiscardChunks()
        return state

    InputProcessor._validate_params = validate_params
    Sampler.add_request = add
    Sampler.__call__ = sample
    RejectionSampler.__call__ = verify_and_capture
    SamplingMaskTensors.from_logits = capture
    SamplingMaskTensors.cat = cat_chunks
    RequestState.from_new_request = new_request
    Sampler._prime_request_replay = True

"""Request-scoped replay for vLLM's V2 non-speculative sampler."""

import os
from contextvars import ContextVar
from copy import copy
from functools import wraps

import numpy as np
import torch

REPLAY_KEY = "prl_replay"
CAPACITY_ENV = "PRIME_SAMPLING_SUPPORT_CAPACITY"
_capture: ContextVar[list | None] = ContextVar("prl_capture", default=None)


def wants_replay(params) -> bool:
    return params is not None and (params.extra_args or {}).get(REPLAY_KEY, False) is True


def patch_request_replay():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.v1.engine.output_processor import RequestState
    from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    from prime_rl.inference.vllm.paired_capture import capture_supports, pack_sampler_rows

    if getattr(Sampler, "_prime_request_replay", False):
        return

    validate = InputProcessor._validate_params

    @wraps(validate)
    def validate_params(self, params, supported_tasks):
        if isinstance(params, SamplingParams):
            if self.model_config.return_sampling_mask:
                if self.speculative_config is not None:
                    raise ValueError("Request-scoped sampling replay does not support speculative decoding")
                if wants_replay(params) and params.temperature <= 0:
                    raise ValueError("Replay requests require temperature > 0")
                # Supports are captured as CSR rows of any width, so vLLM's top_k bound does not apply.
                # Validate against a private view; never toggle shared engine
                # configuration while concurrent requests are being admitted.
                self = copy(self)
                self.model_config = copy(self.model_config)
                self.model_config.return_sampling_mask = False
            elif wants_replay(params):
                raise ValueError("Replay requests require enable_return_sampling_mask")
        return validate(self, params, supported_tasks)

    add_request = Sampler.add_request

    @wraps(add_request)
    def add(self, req_idx, sampling_params):
        if not hasattr(self, "_prime_capture"):
            self._prime_capture = np.zeros(self.sampling_states.max_num_reqs, dtype=bool)
            self._prime_sampler_rows = np.zeros(self.sampling_states.max_num_reqs, dtype=np.int64)
            self._prime_draws = np.zeros(self.sampling_states.max_num_reqs, dtype=bool)
        extra_args = sampling_params.extra_args or {}
        self._prime_capture[req_idx] = wants_replay(sampling_params)
        self._prime_sampler_rows[req_idx] = extra_args.get("prl_sampler_rows", 0)
        self._prime_draws[req_idx] = bool(extra_args.get("prl_sampler_draws"))
        return add_request(self, req_idx, sampling_params)

    call = Sampler.__call__

    @wraps(call)
    def sample(self, logits, input_batch):
        if not hasattr(self, "_prime_capture"):
            return call(self, logits, input_batch)
        indices = input_batch.idx_mapping_np
        replay, sampler_rows = self._prime_capture[indices], self._prime_sampler_rows[indices]
        stash = []
        token = _capture.set(stash)
        try:
            output = call(self, logits, input_batch)
        finally:
            _capture.reset(token)
        if not stash or not (replay.any() or sampler_rows.any()):
            return output
        processed, cu_num_logits, num_sampled = stash
        keep = torch.from_numpy(replay).pin_memory().to(num_sampled.device, non_blocking=True)
        rows = None
        if sampler_rows.any():
            rows = pack_sampler_rows(
                processed,
                cu_num_logits,
                num_sampled,
                output.sampled_token_ids,
                sampler_rows,
                self._prime_draws[indices],
            )
        output.sampling_mask_tensors = capture_supports(
            processed, cu_num_logits, torch.where(keep, num_sampled, 0), int(os.environ[CAPACITY_ENV]), rows
        )
        return output

    from_logits = SamplingMaskTensors.from_logits

    @classmethod
    def capture(cls, logits, cu_num_logits, num_sampled_tokens, *args, **kwargs):
        stash = _capture.get()
        if stash is None:
            return from_logits(logits, cu_num_logits, num_sampled_tokens, *args, **kwargs)
        stash.extend([logits, cu_num_logits, num_sampled_tokens])
        return None

    class DiscardChunks(list):
        def append(self, chunk):
            pass

    from_new_request = RequestState.from_new_request.__func__

    @classmethod
    @wraps(from_new_request)
    def new_request(cls, tokenizer, request, *args, **kwargs):
        state = from_new_request(cls, tokenizer, request, *args, **kwargs)
        params = request.sampling_params
        if not (wants_replay(params) or (params is not None and (params.extra_args or {}).get("prl_sampler_rows"))):
            state.sampling_mask_chunks = DiscardChunks()
        return state

    InputProcessor._validate_params = validate_params
    Sampler.add_request = add
    Sampler.__call__ = sample
    SamplingMaskTensors.from_logits = capture
    RequestState.from_new_request = new_request
    Sampler._prime_request_replay = True

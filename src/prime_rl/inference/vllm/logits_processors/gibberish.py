"""Gibberish detection logits processor.

Detects rare tokens generated at high entropy and forces EOS to abort the rollout.
A token is considered gibberish when both:
  - id(token) > token_id_threshold (rare BPE token)
  - logprob(token) < -log(vocab_size) - logprob_offset (generated at high entropy)
"""

import math
import os
from typing import TYPE_CHECKING

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor


def _force_eos(logits: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    logits.fill_(float("-inf"))
    logits[eos_token_id] = 0.0
    return logits


class GibberishDetector:
    """Per-request callable that detects gibberish tokens.

    Saves the logits from each step. At the next step, checks whether the
    token that was actually sampled (now visible in output_tok_ids) was a
    rare token generated at high entropy. If so, forces EOS.
    """

    def __init__(
        self,
        eos_token_id: int,
        token_id_threshold: int,
        logprob_threshold: float,
    ):
        self.eos_token_id = eos_token_id
        self.token_id_threshold = token_id_threshold
        self.logprob_threshold = logprob_threshold
        self.prev_logits: torch.Tensor | None = None
        self.aborted = False

    def __call__(self, output_tok_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if self.aborted:
            return _force_eos(logits, self.eos_token_id)

        if self.prev_logits is not None and output_tok_ids:
            last_token = output_tok_ids[-1]
            if last_token > self.token_id_threshold:
                log_normalizer = torch.logsumexp(self.prev_logits, dim=-1)
                logprob = (self.prev_logits[last_token] - log_normalizer).item()
                if logprob < self.logprob_threshold:
                    self.aborted = True
                    return _force_eos(logits, self.eos_token_id)

        self.prev_logits = logits.detach().clone()
        return logits


class GibberishDetectionLogitsProcessor(AdapterLogitsProcessor):
    """vLLM v1 adapter that wraps GibberishDetector for each request."""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool):
        self.enabled = os.environ.get("PRIME_GIBBERISH_DETECTION_ENABLED") == "1"
        if not self.enabled:
            return
        super().__init__(vllm_config, device, is_pin_memory)

        eos = vllm_config.model_config.hf_text_config.eos_token_id
        self.eos_token_id = eos[0] if isinstance(eos, list) else eos
        self.token_id_threshold = int(os.environ.get("PRIME_GIBBERISH_DETECTION_TOKEN_ID_THRESHOLD", "100000"))
        logprob_offset = float(os.environ.get("PRIME_GIBBERISH_DETECTION_LOGPROB_OFFSET", "2.0"))
        vocab_size = vllm_config.model_config.hf_text_config.vocab_size
        self.logprob_threshold = -math.log(vocab_size) - logprob_offset

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams) -> "RequestLogitsProcessor | None":
        if not self.enabled:
            return None
        return GibberishDetector(
            eos_token_id=self.eos_token_id,
            token_id_threshold=self.token_id_threshold,
            logprob_threshold=self.logprob_threshold,
        )

    def update_state(self, batch_update):
        if self.enabled:
            super().update_state(batch_update)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return super().apply(logits)
        return logits

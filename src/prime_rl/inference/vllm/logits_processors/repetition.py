"""Repetition detection logits processor.

Detects pathological repetition loops where the model generates with very
high confidence for many consecutive steps. Forces EOS to halt generation
after `window` consecutive tokens where max probability exceeds the threshold.
"""

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


class RepetitionDetector:
    """Per-request callable that detects repetition loops.

    At each step, checks whether the max probability of the current logits
    distribution exceeds a threshold. If this happens for `window` consecutive
    steps, forces EOS. No previous-step state is needed beyond the counter.
    """

    def __init__(
        self,
        eos_token_id: int,
        window: int,
        prob_threshold: float,
    ):
        self.eos_token_id = eos_token_id
        self.window = window
        self.prob_threshold = prob_threshold
        self.consecutive_count = 0
        self.aborted = False

    def __call__(self, output_tok_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if self.aborted:
            return _force_eos(logits, self.eos_token_id)

        max_prob = torch.softmax(logits, dim=-1).max().item()
        if max_prob > self.prob_threshold:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        if self.consecutive_count >= self.window:
            self.aborted = True
            return _force_eos(logits, self.eos_token_id)

        return logits


class RepetitionDetectionLogitsProcessor(AdapterLogitsProcessor):
    """vLLM v1 adapter that wraps RepetitionDetector for each request."""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool):
        self.enabled = os.environ.get("PRIME_REPETITION_DETECTION_ENABLED") == "1"
        if not self.enabled:
            return
        super().__init__(vllm_config, device, is_pin_memory)

        eos = vllm_config.model_config.hf_text_config.eos_token_id
        self.eos_token_id = eos[0] if isinstance(eos, list) else eos
        self.window = int(os.environ.get("PRIME_REPETITION_DETECTION_WINDOW", "3000"))
        self.prob_threshold = float(os.environ.get("PRIME_REPETITION_DETECTION_PROB_THRESHOLD", "0.99"))

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams) -> "RequestLogitsProcessor | None":
        if not self.enabled:
            return None
        return RepetitionDetector(
            eos_token_id=self.eos_token_id,
            window=self.window,
            prob_threshold=self.prob_threshold,
        )

    def update_state(self, batch_update):
        if self.enabled:
            super().update_state(batch_update)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return super().apply(logits)
        return logits

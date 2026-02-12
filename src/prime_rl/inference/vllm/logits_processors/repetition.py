"""Repetition detection logits processor.

Early truncation heuristic for pathological repetition loops during RL
training (Section 3.2, https://arxiv.org/abs/2506.13585). Once a model
enters a repetitive cycle, the probability of each token tends to spike.
String-matching is ineffective against varied repetition patterns, so we
use a probability-based proxy instead: generation is halted when `window`
(default 3000) consecutive tokens each have max probability above
`prob_threshold` (default 0.99).
"""

import os
from typing import TYPE_CHECKING

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from prime_rl.inference.vllm.logits_processors.utils import force_eos

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor


class RepetitionDetector:
    """Per-request callable that detects repetition loops via probability.

    Uses max-probability streaks as a proxy for repetition rather than
    inspecting the token history directly (output_tok_ids is unused).
    See module docstring for rationale.
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
            return force_eos(logits, self.eos_token_id)

        max_prob = torch.softmax(logits, dim=-1).max().item()
        if max_prob > self.prob_threshold:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        if self.consecutive_count >= self.window:
            self.aborted = True
            return force_eos(logits, self.eos_token_id)

        return logits


class RepetitionDetectionLogitsProcessor(AdapterLogitsProcessor):
    """vLLM v1 adapter that wraps RepetitionDetector for each request."""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool):
        self.enabled = os.environ.get("PRIME_REPETITION_DETECTION_ENABLED") == "1"
        super().__init__(vllm_config, device, is_pin_memory)
        if not self.enabled:
            return

        eos = vllm_config.model_config.hf_text_config.eos_token_id
        self.eos_token_id = eos[0] if isinstance(eos, list) else eos
        self.window = int(os.environ.get("PRIME_REPETITION_DETECTION_WINDOW", "3000"))
        self.prob_threshold = float(os.environ.get("PRIME_REPETITION_DETECTION_PROB_THRESHOLD", "0.99"))

    def is_argmax_invariant(self) -> bool:
        return not self.enabled

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

# SPDX-License-Identifier: Apache-2.0
"""A layer that samples the next tokens from the model's outputs."""

from typing import Optional

import torch
from vllm.model_executor.layers.sampler import (
    Sampler,
    SampleResultArgsType,
    SamplerOutput,
    _apply_min_p,
    _apply_min_tokens_penalty,
    _apply_top_k_top_p,
    _build_sampler_output,
)
from vllm.model_executor.layers.utils import apply_penalties
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sequence import Logprob

GUMBEL_BATCH_SIZE = 2**16


def generate_neg_gumbel_noise(n: int | tuple[int, int], generator: torch.Generator, device: torch.device):
    if isinstance(n, int):
        ret = torch.empty(n, device=device)
        for i in range(0, n, GUMBEL_BATCH_SIZE):
            end = min(i + GUMBEL_BATCH_SIZE, n)
            ret[i:end].exponential_(generator=generator).log_()
    else:
        ret = torch.empty(n[0], n[1], device=device)
        for i in range(0, n[0]):
            for j in range(0, n[1], GUMBEL_BATCH_SIZE):
                end_j = min(j + GUMBEL_BATCH_SIZE, n[1])
                ret[i, j:end_j].exponential_(generator=generator).log_()
    return ret


class Toploc2Sampler(Sampler):
    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """
        Args:
            logits: (num_tokens, vocab_size).
            sampling_metadata: Metadata for sampling.
        """
        assert logits is not None
        _, vocab_size = logits.shape

        # Prepare sampling tensors with pinned memory to avoid blocking.
        if not sampling_metadata.reuse_sampling_tensors:
            self._init_sampling_tensors(logits, sampling_metadata)
        elif self._do_penalties:
            # In this case, the sampling tensors logic depends on
            # "output_tokens" of a sequence. As a result, we cannot
            # reuse sampling tensors, since "output_tokens" changes
            # between decode runs.
            self._init_sampling_tensors(logits, sampling_metadata)

        assert self._sampling_tensors is not None
        sampling_tensors = self._sampling_tensors
        do_penalties = self._do_penalties
        do_top_p_top_k = self._do_top_p_top_k
        do_min_p = self._do_min_p

        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = apply_penalties(
                logits,
                sampling_tensors.prompt_tokens,
                sampling_tensors.output_tokens,
                sampling_tensors.presence_penalties,
                sampling_tensors.frequency_penalties,
                sampling_tensors.repetition_penalties,
            )

        # Use float32 to apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits = logits.to(torch.float)
        logits.div_(sampling_tensors.temperatures.unsqueeze(dim=1))

        if do_top_p_top_k:
            logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps, sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # # We use float32 for probabilities and log probabilities.
        # # Compute the probabilities.
        # probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        chosen_noises = []

        def _sample(logits, sampling_metadata: SamplingMetadata, sampling_tensors: SamplingTensors):
            assert len(sampling_metadata.seq_groups) == logits.shape[0]
            neg_gumbel_noise = torch.stack(
                [generate_neg_gumbel_noise(logits.shape[-1], sg.generator, logits.device) for sg in sampling_metadata.seq_groups]
            )
            assert neg_gumbel_noise.shape == logits.shape
            _race_result = logits - neg_gumbel_noise
            token_ids = torch.argmax(_race_result, dim=-1)
            chosen_noises.append(torch.gather(neg_gumbel_noise, 1, token_ids.unsqueeze(1)))
            return [([token_ids[i].item()], [0]) for i in range(len(sampling_metadata.seq_groups))]

        # Sample the next tokens.
        maybe_deferred_sample_results = _sample(
            logits,
            sampling_metadata,
            sampling_tensors,
        )
        # Get the logprobs query results.
        prompt_logprobs = None
        sample_logprobs = None
        if not sampling_metadata.skip_sampler_cpu_output:
            # Pythonize logprobs now (GPU -> CPU); do not defer.
            assert not isinstance(maybe_deferred_sample_results, SampleResultArgsType)
            prompt_logprobs = [None] * len(sampling_metadata.seq_groups)
            # TODO: How did the original code know the rank?
            sample_logprobs = [
                [
                    {
                        sample_result[0][0]: Logprob(logprob=logprobs[i, sample_result[0][0]].item(), rank=1, decoded_token=None),
                        0: Logprob(logprob=logits[i, sample_result[0][0]].item(), rank=2, decoded_token=None),
                        1: Logprob(logprob=chosen_noises[0][i].item(), rank=3, decoded_token=None),
                    }
                ]
                for i, sample_result in enumerate(maybe_deferred_sample_results)
            ]

        return _build_sampler_output(
            maybe_deferred_sample_results,
            sampling_metadata,
            prompt_logprobs=prompt_logprobs,
            sample_logprobs=sample_logprobs,
            on_device_tensors=None,
            skip_sampler_cpu_output=False,
        )

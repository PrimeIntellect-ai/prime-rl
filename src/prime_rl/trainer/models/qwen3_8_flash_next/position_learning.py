import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import ProcessGroup

from prime_rl.trainer.models.qwen3_8_flash_next.hyper_connection import ExpandedRMSNorm
from prime_rl.trainer.models.qwen3_8_flash_next.ngram_embedding import NGramEmbedding
from prime_rl.utils.cp import gather_for_cp


class PositionLearningEnhancement(nn.Module):
    """Inject hashed N-gram features into the expanded residual streams."""

    def __init__(
        self,
        *,
        hidden_size: int,
        stream_count: int,
        embedding_dim: int,
        ngram_size: int,
        heads_per_ngram: int,
        ngram_vocab_size: int,
        token_vocab_size: int,
        eos_token_id: int,
        vocab_size_divisor: int,
        ngram_layer_index: int,
        conv_kernel_size: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        expanded_size = stream_count * hidden_size
        self.hidden_size = hidden_size
        self.stream_count = stream_count
        self.conv_dilation = ngram_size

        self.ple_embedding = NGramEmbedding(
            embedding_dim=embedding_dim,
            ngram_size=ngram_size,
            heads_per_ngram=heads_per_ngram,
            ngram_vocab_size=ngram_vocab_size,
            token_vocab_size=token_vocab_size,
            eos_token_id=eos_token_id,
            vocab_size_divisor=vocab_size_divisor,
            ngram_layer_index=ngram_layer_index,
        )
        self.key_proj = nn.Linear(embedding_dim, expanded_size, bias=False)
        self.value_proj = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.norm_key = ExpandedRMSNorm(hidden_size, stream_count, norm_eps)
        self.norm_query = ExpandedRMSNorm(hidden_size, stream_count, norm_eps)
        self.norm_conv = ExpandedRMSNorm(hidden_size, stream_count, norm_eps)
        self.conv1d = nn.Conv1d(
            expanded_size,
            expanded_size,
            conv_kernel_size,
            groups=expanded_size,
            dilation=ngram_size,
            bias=False,
        )
        nn.init.zeros_(self.conv1d.weight)

        self.context_parallel_group: ProcessGroup | None = None
        self.context_parallel_rank = 0

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.conv1d.weight)

    def set_context_parallel_attributes(self, process_group: ProcessGroup, rank: int, world_size: int) -> None:
        self.context_parallel_group = process_group
        self.context_parallel_rank = rank
        self.ple_embedding.set_context_parallel_attributes(process_group, rank, world_size)

    def forward(
        self,
        expanded_states: torch.Tensor,
        input_ids: torch.LongTensor,
        cu_seqlens: torch.LongTensor,
    ) -> torch.Tensor:
        embeddings = self.ple_embedding(input_ids, cu_seqlens).to(expanded_states.dtype)
        keys = self.norm_key(self.key_proj(embeddings))
        queries = self.norm_query(expanded_states)
        keys = keys.unflatten(-1, (self.stream_count, self.hidden_size))
        queries = queries.unflatten(-1, (self.stream_count, self.hidden_size))

        gate = (keys * queries).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        values = self.value_proj(embeddings).unsqueeze(-2)
        gated_values = gate * values
        convolution_input = self.norm_conv(gated_values.flatten(-2))

        local_token_count = convolution_input.shape[1]
        if self.context_parallel_group is not None:
            convolution_input = gather_for_cp(convolution_input, self.context_parallel_group)

        packed_input = convolution_input.flatten(0, 1)
        positions = torch.arange(packed_input.shape[0], device=packed_input.device)
        sequence_indices = torch.searchsorted(cu_seqlens[1:], positions, right=True)
        sequence_starts = cu_seqlens[:-1][sequence_indices].long()
        convolution_output = torch.zeros_like(packed_input, dtype=torch.float32)
        weights = self.conv1d.weight[:, 0].float()
        kernel_size = weights.shape[-1]
        for kernel_index in range(kernel_size):
            delay = (kernel_size - kernel_index - 1) * self.conv_dilation
            source_positions = positions - delay
            source = packed_input[source_positions.clamp_min(0)]
            source = torch.where(
                (source_positions >= sequence_starts).unsqueeze(-1),
                source,
                torch.zeros_like(source),
            )
            convolution_output.add_(source.float() * weights[:, kernel_index])

        convolution_output = F.silu(convolution_output.to(packed_input.dtype)).reshape_as(convolution_input)
        if self.context_parallel_group is not None:
            local_start = self.context_parallel_rank * local_token_count
            convolution_output = convolution_output[:, local_start : local_start + local_token_count]
        return gated_values.flatten(-2) + convolution_output


__all__ = ["PositionLearningEnhancement"]

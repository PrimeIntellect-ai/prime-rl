import torch
from fla.modules import FusedRMSNormGated
from fla.modules.conv import causal_conv1d
from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda
from torch import nn

from prime_rl.utils.cp import CPContext

# Dynamo lowers all-gather to concatenation, then fails to copy the result into
# FLA's stacked output buffer. Keep CP convolution eager until this is fixed:
# https://github.com/pytorch/pytorch/issues/155632
causal_conv1d_with_context_parallelism = torch.compiler.disable(causal_conv1d)
# CP setup searchsorts tensor-valued sequence boundaries; keep it outside the compiled graph.
build_cp_context_eager = torch.compiler.disable(build_cp_context)


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention with GLM-5.3's safe (lower-bounded) forget gate.

    The checkpoint keeps one depthwise convolution per projection; they run as a single
    convolution over the concatenated q/k/v channels, which is the same computation.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        gate_lower_bound: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.conv_kernel_size = conv_kernel_size
        self.gate_lower_bound = gate_lower_bound

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.q_conv1d = self._depthwise_conv()
        self.k_conv1d = self._depthwise_conv()
        self.v_conv1d = self._depthwise_conv()

        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.A_log = nn.Parameter(torch.zeros(num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.projection_size))
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.o_norm = FusedRMSNormGated(head_dim, eps=norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False)

        self.cp_context = CPContext()

    def muon_head_partitions(self, heads_per_block: int) -> dict[nn.Parameter, None]:
        # Linear attention is not split by head (like Qwen3.5's gated delta net)
        return {p: None for p in self.parameters()}

    def _depthwise_conv(self) -> nn.Conv1d:
        return nn.Conv1d(
            self.projection_size,
            self.projection_size,
            kernel_size=self.conv_kernel_size,
            groups=self.projection_size,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.LongTensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        # FLA memoizes cu_seqlens metadata by tensor identity. A per-layer copy makes every layer
        # miss that cache in the forward and in its activation-checkpoint recompute alike, so both
        # run the same ops and checkpoint replay (matched by op invocation order) stays aligned.
        cu_seqlens = cu_seqlens.clone()

        context = None
        if self.cp_context.cp_enabled:
            context = build_cp_context_eager(
                cu_seqlens=cu_seqlens.to(device=hidden_states.device, dtype=torch.int32),
                group=self.cp_context.cp_group,
                conv1d_kernel_size=self.conv_kernel_size,
            )

        mixed_qkv = torch.cat(
            (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)), dim=-1
        )
        conv_weight = torch.cat((self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight)).squeeze(1)
        convolution = {"x": mixed_qkv, "weight": conv_weight, "bias": None, "activation": "silu"}
        if context is None:
            mixed_qkv, _ = causal_conv1d(**convolution, cu_seqlens=cu_seqlens)
        else:
            mixed_qkv, _ = causal_conv1d_with_context_parallelism(**convolution, cp_context=context)

        head_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        query, key, value = (part.reshape(head_shape) for part in mixed_qkv.chunk(3, dim=-1))
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)).reshape(head_shape)

        delta_rule = {
            "q": query,
            "k": key,
            "v": value,
            "g": forget_gate,
            "beta": self.b_proj(hidden_states),
            "A_log": self.A_log,
            "dt_bias": self.dt_bias,
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "use_beta_sigmoid_in_kernel": True,
            "safe_gate": True,
            "lower_bound": self.gate_lower_bound,
            "cu_seqlens": context.cu_seqlens if context is not None else cu_seqlens,
        }
        if context is None:
            core_output, _ = chunk_kda(**delta_rule, initial_state=None, output_final_state=False)
        else:
            core_output, _ = chunk_kda(**delta_rule, cp_context=context)

        output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        core_output = self.o_norm(
            core_output.reshape(-1, self.head_dim),
            output_gate.reshape(-1, self.head_dim),
        )
        return self.o_proj(core_output.reshape(batch_size, sequence_length, self.projection_size))


__all__ = ["KimiDeltaAttention"]

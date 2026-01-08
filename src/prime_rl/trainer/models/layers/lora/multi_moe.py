import math

import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.layers.lora.base import MultiLoRAModule, get_lora_num_tokens, get_multilora_scaling
from prime_rl.trainer.models.layers.moe import GroupedExperts


def _run_lora_grouped_mm(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Apply LoRA via grouped matrix multiplication.

    Args:
        x: Input tensor [total_tokens, in_features]
        lora_A: Low-rank A matrices [num_experts, rank, in_features]
        lora_B: Low-rank B matrices [num_experts, out_features, rank]
        offsets: Cumulative token counts per expert [num_experts]

    Returns:
        LoRA output [total_tokens, out_features]
    """
    _a_out = torch._grouped_mm(x.bfloat16(), lora_A.bfloat16().transpose(-2, -1), offs=offsets)
    lora_out = torch._grouped_mm(_a_out, lora_B.bfloat16().transpose(-2, -1), offs=offsets)
    return lora_out


def _run_lora_for_loop(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Apply LoRA via for-loop over experts (fallback for non-Hopper GPUs).

    Args:
        x: Input tensor [total_tokens, in_features]
        lora_A: Low-rank A matrices [num_experts, rank, in_features]
        lora_B: Low-rank B matrices [num_experts, out_features, rank]
        num_tokens_per_expert: Token counts per expert [num_experts]

    Returns:
        LoRA output [total_tokens, out_features]
    """
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()
    lora_out_splits = []

    start = 0
    for expert_idx, num_tokens in enumerate(num_tokens_per_expert_list):
        if num_tokens == 0:
            continue
        end = start + num_tokens

        # Apply LoRA for this expert: B @ A @ x
        _a_out = torch.matmul(x[start:end], lora_A[expert_idx].transpose(-2, -1))
        lora_out = torch.matmul(_a_out, lora_B[expert_idx].transpose(-2, -1))
        lora_out_splits.append(lora_out)

        start = end

    return torch.cat(lora_out_splits, dim=0)


class MultiLoRAGroupedExperts(MultiLoRAModule):
    """
    GroupedExperts + multi-LoRA with grouped GEMM.
    Applies LoRA to all three expert projections (w1, w2, w3) for multi-tenant MoE training.
    Compatible with vLLM's MoE LoRA format when broadcasting weights.
    """

    def __init__(
        self,
        base_layer: GroupedExperts,
        rank: int,
        n_adapters: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__(base_layer)
        if rank <= 0 or n_adapters <= 0:
            raise ValueError("rank and n_adapters must be > 0")

        self.num_experts = base_layer.num_experts
        self.dim = base_layer.w1.shape[2]
        self.hidden_dim = base_layer.w1.shape[1]

        use_grouped_mm = False
        if torch.cuda.is_available():
            cc_major, _ = torch.cuda.get_device_capability()
            if cc_major == 9:  # Hopper GPUs (H100, H200)
                use_grouped_mm = True

        # grouped_mm requires 8-byte alignment
        if rank % 8 != 0 or self.dim % 8 != 0 or self.hidden_dim % 8 != 0:
            use_grouped_mm = False

        self.rank = rank
        self.n_adapters = n_adapters
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.use_grouped_mm = use_grouped_mm

        self._lora_num_tokens = get_lora_num_tokens()
        self._scaling_factors = get_multilora_scaling()

        # Initialize LoRA parameters for w1 (gate_proj: dim -> moe_dim)
        self.w1_lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        rank,
                        self.dim,
                        device=base_layer.w1.device,
                        dtype=base_layer.w1.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.w1_lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.hidden_dim,
                        rank,
                        device=base_layer.w1.device,
                        dtype=base_layer.w1.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        # Initialize LoRA parameters for w2 (down_proj: moe_dim -> dim)
        self.w2_lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        rank,
                        self.hidden_dim,
                        device=base_layer.w2.device,
                        dtype=base_layer.w2.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.w2_lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.dim,
                        rank,
                        device=base_layer.w2.device,
                        dtype=base_layer.w2.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        # Initialize LoRA parameters for w3 (up_proj: moe_dim -> dim)
        self.w3_lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        rank,
                        self.dim,
                        device=base_layer.w3.device,
                        dtype=base_layer.w3.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.w3_lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.hidden_dim,
                        rank,
                        device=base_layer.w3.device,
                        dtype=base_layer.w3.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self, index: int | None = None) -> None:
        """Reset LoRA parameters using Kaiming uniform for A, zeros for B.

        Args:
            index: If provided, reset only the parameters for that adapter index.
                   If None, reset all adapter parameters.
        """
        if index is None:
            for i in range(self.n_adapters):
                self.reset_parameters(i)
        else:
            # Reset w1 LoRA
            nn.init.kaiming_uniform_(self.w1_lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.w1_lora_B[index])

            # Reset w2 LoRA
            nn.init.kaiming_uniform_(self.w2_lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.w2_lora_B[index])

            # Reset w3 LoRA
            nn.init.kaiming_uniform_(self.w3_lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.w3_lora_B[index])

    def named_parameters_for_adapter(self, idx: int) -> list[tuple[str, nn.Parameter]]:
        """Get named parameters for a specific adapter index.

        Returns the full stacked parameters (leaf tensors) for optimizer use.
        Each parameter has shape [num_experts, ...] with all experts stacked.

        Args:
            idx: The adapter index to get parameters for

        Returns:
            List of (name, parameter) tuples for the specified adapter
        """
        return [
            ("w1_lora_A", self.w1_lora_A[idx]),  # Shape: [num_experts, rank, dim]
            ("w1_lora_B", self.w1_lora_B[idx]),  # Shape: [num_experts, moe_dim, rank]
            ("w2_lora_A", self.w2_lora_A[idx]),  # Shape: [num_experts, rank, moe_dim]
            ("w2_lora_B", self.w2_lora_B[idx]),  # Shape: [num_experts, dim, rank]
            ("w3_lora_A", self.w3_lora_A[idx]),  # Shape: [num_experts, rank, dim]
            ("w3_lora_B", self.w3_lora_B[idx]),  # Shape: [num_experts, moe_dim, rank]
        ]

    def get_lora_param_counts(self) -> tuple[int, int]:
        """Get the number of LoRA adapter parameters and adapted base parameters.

        Returns:
            A tuple of (adapter_params, adapted_params) where:
            - adapter_params: Number of parameters in ONE LoRA adapter (all w1/w2/w3 lora_A + lora_B)
            - adapted_params: Number of base layer parameters being adapted by LoRA (w1, w2, w3)
        """
        adapter_params = (
            self.w1_lora_A[0].numel()
            + self.w1_lora_B[0].numel()
            + self.w2_lora_A[0].numel()
            + self.w2_lora_B[0].numel()
            + self.w3_lora_A[0].numel()
            + self.w3_lora_B[0].numel()
        )
        adapted_params = self.base_layer.w1.numel() + self.base_layer.w2.numel() + self.base_layer.w3.numel()
        return adapter_params, adapted_params

    def state_dict_for_adapter(self, idx: int) -> dict[str, torch.Tensor]:
        """Get state dict for a specific adapter index in vLLM-compatible format.

        Returns per-expert parameter slices for vLLM compatibility.
        For 8 experts, returns 48 tensors (8 experts × 3 projections × 2 matrices):
        - {expert_id}.gate_proj.lora_A.weight
        - {expert_id}.gate_proj.lora_B.weight
        - {expert_id}.down_proj.lora_A.weight
        - {expert_id}.down_proj.lora_B.weight
        - {expert_id}.up_proj.lora_A.weight
        - {expert_id}.up_proj.lora_B.weight

        Args:
            idx: The adapter index to get state dict for

        Returns:
            Dict mapping vLLM-compatible names to parameter tensors
        """
        state_dict = {}

        detached_w1_lora_a = self.w1_lora_A[idx].detach()
        detached_w1_lora_b = self.w1_lora_B[idx].detach()
        detached_w2_lora_a = self.w2_lora_A[idx].detach()
        detached_w2_lora_b = self.w2_lora_B[idx].detach()
        detached_w3_lora_a = self.w3_lora_A[idx].detach()
        detached_w3_lora_b = self.w3_lora_B[idx].detach()

        # The clone is necessary to avoid views that cause giant memory spikes
        # TODO: There's probably a better way to do this
        for expert_id in range(self.num_experts):
            state_dict[f"{expert_id}.gate_proj.lora_A.weight"] = detached_w1_lora_a[expert_id].clone()
            state_dict[f"{expert_id}.gate_proj.lora_B.weight"] = detached_w1_lora_b[expert_id].clone()
            state_dict[f"{expert_id}.down_proj.lora_A.weight"] = detached_w2_lora_a[expert_id].clone()
            state_dict[f"{expert_id}.down_proj.lora_B.weight"] = detached_w2_lora_b[expert_id].clone()
            state_dict[f"{expert_id}.up_proj.lora_A.weight"] = detached_w3_lora_a[expert_id].clone()
            state_dict[f"{expert_id}.up_proj.lora_B.weight"] = detached_w3_lora_b[expert_id].clone()

        return state_dict

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        # TODO: We assume theres only one adapter active in a sequence for now
        # Being able to route multi-adapter sequences efficiently requires two things that are tricky
        # 1. We need the tensor to be interleaved [(e0, a0), (e0, a1), (e1, a0), (e1, a1), ...]
        # This causes issues when we want to create a stacked param for the optimizer
        # 2. The topkrouter needs to set the offsets by binning its hist for each adapter
        # The sort currently occurs there, so it needs to be done there too
        adapter_idx = self._lora_num_tokens.argmax().item()
        w1_lora_a = self.w1_lora_A[adapter_idx]  # [num_experts, rank, dim]
        w1_lora_b = self.w1_lora_B[adapter_idx]  # [num_experts, hidden_dim, rank]
        w2_lora_a = self.w2_lora_A[adapter_idx]  # [num_experts, rank, hidden_dim]
        w2_lora_b = self.w2_lora_B[adapter_idx]  # [num_experts, dim, rank]
        w3_lora_a = self.w3_lora_A[adapter_idx]  # [num_experts, rank, dim]
        w3_lora_b = self.w3_lora_B[adapter_idx]  # [num_experts, hidden_dim, rank]

        # Get per-adapter scaling factor
        scaling = self._scaling_factors[adapter_idx].item()

        # Access base weights directly
        base_w1 = self.base_layer.w1  # [num_experts, hidden_dim, dim]
        base_w2 = self.base_layer.w2  # [num_experts, dim, hidden_dim]
        base_w3 = self.base_layer.w3  # [num_experts, hidden_dim, dim]

        # Compute offsets for grouped_mm
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        lora_x = self.lora_dropout(x)

        if self.use_grouped_mm:
            # Gate
            h1_base = torch._grouped_mm(x.bfloat16(), base_w1.bfloat16().transpose(-2, -1), offs=offsets)
            w1_lora_out = _run_lora_grouped_mm(lora_x, w1_lora_a, w1_lora_b, offsets)
            h1 = h1_base + scaling * w1_lora_out.bfloat16()

            # Up
            h3_base = torch._grouped_mm(x.bfloat16(), base_w3.bfloat16().transpose(-2, -1), offs=offsets)
            w3_lora_out = _run_lora_grouped_mm(lora_x, w3_lora_a, w3_lora_b, offsets)
            h3 = h3_base + scaling * w3_lora_out.bfloat16()

            # SwiGLU activation
            h = F.silu(h1) * h3

            # Down
            lora_h = self.lora_dropout(h)
            h2_base = torch._grouped_mm(h, base_w2.bfloat16().transpose(-2, -1), offs=offsets)
            w2_lora_out = _run_lora_grouped_mm(lora_h, w2_lora_a, w2_lora_b, offsets)
            out = h2_base + scaling * w2_lora_out.bfloat16()

            return out.type_as(x)
        else:
            # For-loop fallback for older GPUs
            return self._forward_for_loop(
                x,
                num_tokens_per_expert,
                w1_lora_a,
                w1_lora_b,
                w2_lora_a,
                w2_lora_b,
                w3_lora_a,
                w3_lora_b,
                base_w1,
                base_w2,
                base_w3,
                scaling,
            )

    def _forward_for_loop(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        w1_lora_a: torch.Tensor,
        w1_lora_b: torch.Tensor,
        w2_lora_a: torch.Tensor,
        w2_lora_b: torch.Tensor,
        w3_lora_a: torch.Tensor,
        w3_lora_b: torch.Tensor,
        base_w1: torch.Tensor,
        base_w2: torch.Tensor,
        base_w3: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        """For-loop implementation of forward pass (fallback for non-Hopper GPUs)."""
        num_tokens_per_expert_list = num_tokens_per_expert.tolist()
        out_splits = []

        start = 0
        for expert_idx, num_tokens in enumerate(num_tokens_per_expert_list):
            if num_tokens == 0:
                continue
            end = start + num_tokens
            x_expert = x[start:end]
            # Apply dropout for LoRA path (consistent with grouped_mm path)
            x_expert_lora = self.lora_dropout(x_expert)

            # Compute w1 + w1_lora for this expert
            h1_base = torch.matmul(x_expert, base_w1[expert_idx].transpose(-2, -1))
            w1_lora_tmp = torch.matmul(x_expert_lora, w1_lora_a[expert_idx].transpose(-2, -1))
            w1_lora_out = torch.matmul(w1_lora_tmp, w1_lora_b[expert_idx].transpose(-2, -1))
            h1 = h1_base + scaling * w1_lora_out

            # Compute w3 + w3_lora for this expert
            h3_base = torch.matmul(x_expert, base_w3[expert_idx].transpose(-2, -1))
            w3_lora_tmp = torch.matmul(x_expert_lora, w3_lora_a[expert_idx].transpose(-2, -1))
            w3_lora_out = torch.matmul(w3_lora_tmp, w3_lora_b[expert_idx].transpose(-2, -1))
            h3 = h3_base + scaling * w3_lora_out

            # SwiGLU activation
            h = F.silu(h1) * h3

            # Compute w2 + w2_lora for this expert
            # Apply dropout for LoRA path (consistent with grouped_mm path)
            h_lora = self.lora_dropout(h)
            h2_base = torch.matmul(h, base_w2[expert_idx].transpose(-2, -1))
            w2_lora_tmp = torch.matmul(h_lora, w2_lora_a[expert_idx].transpose(-2, -1))
            w2_lora_out = torch.matmul(w2_lora_tmp, w2_lora_b[expert_idx].transpose(-2, -1))
            out = h2_base + scaling * w2_lora_out

            out_splits.append(out)
            start = end

        return torch.cat(out_splits, dim=0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"n_adapters={self.n_adapters}, num_experts={self.num_experts}, "
            f"alpha={self.alpha}, dropout={self.lora_dropout}, "
            f"use_grouped_mm={self.use_grouped_mm})"
        )

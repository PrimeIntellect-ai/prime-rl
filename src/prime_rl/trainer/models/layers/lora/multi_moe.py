import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.activations import ClampedSwiglu
from prime_rl.trainer.models.layers.lora.base import MultiLoRAModule, get_lora_num_tokens, get_multilora_scaling
from prime_rl.trainer.models.layers.moe import (
    GroupedExperts,
    broadcast_expert_bias,
)


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


class MultiLoRAGroupedExperts(MultiLoRAModule):
    """
    Gated GroupedExperts + multi-LoRA with grouped GEMM.
    Applies LoRA to all three expert projections (gate_proj, down_proj, up_proj) for multi-tenant MoE training.
    Compatible with vLLM's MoE LoRA format when broadcasting weights.
    """

    def __init__(
        self,
        base_layer: GroupedExperts,
        rank: int,
        n_adapters: int,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__(base_layer)
        if rank <= 0 or n_adapters <= 0:
            raise ValueError("rank and n_adapters must be > 0")
        if base_layer.gate_proj is None:
            raise ValueError("MultiLoRAGroupedExperts requires gated experts")

        self.num_experts = base_layer.num_experts
        self.dim = base_layer.gate_proj.shape[2]
        self.hidden_dim = base_layer.gate_proj.shape[1]

        if rank % 8 != 0 or self.dim % 8 != 0 or self.hidden_dim % 8 != 0:
            raise ValueError("grouped_mm requires rank and expert dimensions divisible by 8")

        self.rank = rank
        self.n_adapters = n_adapters
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._lora_num_tokens = get_lora_num_tokens()
        self._scaling_factors = get_multilora_scaling()

        # Initialize LoRA parameters for gate_proj (gate_proj: dim -> moe_dim)
        self.gate_proj_lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        rank,
                        self.dim,
                        device=base_layer.gate_proj.device,
                        dtype=base_layer.gate_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.gate_proj_lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.hidden_dim,
                        rank,
                        device=base_layer.gate_proj.device,
                        dtype=base_layer.gate_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        # Initialize LoRA parameters for down_proj (down_proj: moe_dim -> dim)
        self.down_proj_lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        rank,
                        self.hidden_dim,
                        device=base_layer.down_proj.device,
                        dtype=base_layer.down_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.down_proj_lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.dim,
                        rank,
                        device=base_layer.down_proj.device,
                        dtype=base_layer.down_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        # Initialize LoRA parameters for up_proj (up_proj: moe_dim -> dim)
        self.up_proj_lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        rank,
                        self.dim,
                        device=base_layer.up_proj.device,
                        dtype=base_layer.up_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.up_proj_lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.hidden_dim,
                        rank,
                        device=base_layer.up_proj.device,
                        dtype=base_layer.up_proj.dtype,
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
            # Reset gate_proj LoRA
            nn.init.kaiming_uniform_(self.gate_proj_lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.gate_proj_lora_B[index])

            # Reset down_proj LoRA
            nn.init.kaiming_uniform_(self.down_proj_lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.down_proj_lora_B[index])

            # Reset up_proj LoRA
            nn.init.kaiming_uniform_(self.up_proj_lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.up_proj_lora_B[index])

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
            ("gate_proj_lora_A", self.gate_proj_lora_A[idx]),  # Shape: [num_experts, rank, dim]
            ("gate_proj_lora_B", self.gate_proj_lora_B[idx]),  # Shape: [num_experts, moe_dim, rank]
            ("down_proj_lora_A", self.down_proj_lora_A[idx]),  # Shape: [num_experts, rank, moe_dim]
            ("down_proj_lora_B", self.down_proj_lora_B[idx]),  # Shape: [num_experts, dim, rank]
            ("up_proj_lora_A", self.up_proj_lora_A[idx]),  # Shape: [num_experts, rank, dim]
            ("up_proj_lora_B", self.up_proj_lora_B[idx]),  # Shape: [num_experts, moe_dim, rank]
        ]

    def get_lora_param_counts(self) -> tuple[int, int]:
        """Get the number of LoRA adapter parameters and adapted base parameters.

        Returns:
            A tuple of (adapter_params, adapted_params) where:
            - adapter_params: Number of parameters in ONE LoRA adapter (all gate_proj/down_proj/up_proj lora_A + lora_B)
            - adapted_params: Number of base layer parameters being adapted by LoRA (gate_proj, down_proj, up_proj)
        """
        adapter_params = (
            self.gate_proj_lora_A[0].numel()
            + self.gate_proj_lora_B[0].numel()
            + self.down_proj_lora_A[0].numel()
            + self.down_proj_lora_B[0].numel()
            + self.up_proj_lora_A[0].numel()
            + self.up_proj_lora_B[0].numel()
        )
        adapted_params = (
            self.base_layer.gate_proj.numel() + self.base_layer.down_proj.numel() + self.base_layer.up_proj.numel()
        )
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

        detached_gate_proj_lora_a = self.gate_proj_lora_A[idx].detach()
        detached_gate_proj_lora_b = self.gate_proj_lora_B[idx].detach()
        detached_down_proj_lora_a = self.down_proj_lora_A[idx].detach()
        detached_down_proj_lora_b = self.down_proj_lora_B[idx].detach()
        detached_up_proj_lora_a = self.up_proj_lora_A[idx].detach()
        detached_up_proj_lora_b = self.up_proj_lora_B[idx].detach()

        # With EP, LoRA weights are DTensors sharded across expert-parallel ranks.
        # Gather them before per-expert indexing.
        if isinstance(detached_gate_proj_lora_a, DTensor):
            detached_gate_proj_lora_a = detached_gate_proj_lora_a.full_tensor()
            detached_gate_proj_lora_b = detached_gate_proj_lora_b.full_tensor()
            detached_down_proj_lora_a = detached_down_proj_lora_a.full_tensor()
            detached_down_proj_lora_b = detached_down_proj_lora_b.full_tensor()
            detached_up_proj_lora_a = detached_up_proj_lora_a.full_tensor()
            detached_up_proj_lora_b = detached_up_proj_lora_b.full_tensor()

        # The clone is necessary to avoid views that cause giant memory spikes
        # TODO: There's probably a better way to do this
        for expert_id in range(self.num_experts):
            state_dict[f"{expert_id}.gate_proj.lora_A.weight"] = detached_gate_proj_lora_a[expert_id].clone()
            state_dict[f"{expert_id}.gate_proj.lora_B.weight"] = detached_gate_proj_lora_b[expert_id].clone()
            state_dict[f"{expert_id}.down_proj.lora_A.weight"] = detached_down_proj_lora_a[expert_id].clone()
            state_dict[f"{expert_id}.down_proj.lora_B.weight"] = detached_down_proj_lora_b[expert_id].clone()
            state_dict[f"{expert_id}.up_proj.lora_A.weight"] = detached_up_proj_lora_a[expert_id].clone()
            state_dict[f"{expert_id}.up_proj.lora_B.weight"] = detached_up_proj_lora_b[expert_id].clone()

        return state_dict

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        # TODO: We assume theres only one adapter active in a sequence for now
        # Being able to route multi-adapter sequences efficiently requires two things that are tricky
        # 1. We need the tensor to be interleaved [(e0, a0), (e0, a1), (e1, a0), (e1, a1), ...]
        # This causes issues when we want to create a stacked param for the optimizer
        # 2. The topkrouter needs to set the offsets by binning its hist for each adapter
        # The sort currently occurs there, so it needs to be done there too
        adapter_idx = self._lora_num_tokens.argmax().item()
        gate_proj_lora_a = self.gate_proj_lora_A[adapter_idx]  # [num_experts, rank, dim]
        gate_proj_lora_b = self.gate_proj_lora_B[adapter_idx]  # [num_experts, hidden_dim, rank]
        down_proj_lora_a = self.down_proj_lora_A[adapter_idx]  # [num_experts, rank, hidden_dim]
        down_proj_lora_b = self.down_proj_lora_B[adapter_idx]  # [num_experts, dim, rank]
        up_proj_lora_a = self.up_proj_lora_A[adapter_idx]  # [num_experts, rank, dim]
        up_proj_lora_b = self.up_proj_lora_B[adapter_idx]  # [num_experts, hidden_dim, rank]

        # Get per-adapter scaling factor
        scaling = self._scaling_factors[adapter_idx].item()

        # Access base weights directly
        gate_proj = self.base_layer.gate_proj
        up_proj = self.base_layer.up_proj
        down_proj = self.base_layer.down_proj

        # EP handling: convert DTensors to local shards.
        # Standard EP also needs token permutation; DeepEP tokens are already dispatched.
        permuted_indices = None
        if isinstance(gate_proj, DTensor):
            gate_proj = gate_proj.to_local()
            up_proj = up_proj.to_local()
            down_proj = down_proj.to_local()
            gate_proj_lora_a = gate_proj_lora_a.to_local()
            gate_proj_lora_b = gate_proj_lora_b.to_local()
            down_proj_lora_a = down_proj_lora_a.to_local()
            down_proj_lora_b = down_proj_lora_b.to_local()
            up_proj_lora_a = up_proj_lora_a.to_local()
            up_proj_lora_b = up_proj_lora_b.to_local()

            if getattr(self.base_layer, "ep_comm_backend", "torch") != "deepep":
                from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

                from prime_rl.trainer.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M

                experts_per_ep_rank = gate_proj.shape[0]
                num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank

                with torch.no_grad():
                    permuted_indices, num_tokens_per_expert, _ = generate_permute_indices(
                        num_tokens_per_expert,
                        experts_per_ep_rank,
                        num_ep_ranks,
                        x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M,
                        TOKEN_GROUP_ALIGN_SIZE_M,
                    )

                x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
                input_shape = x.shape
                x = x[permuted_indices, :]

        # Compute offsets for grouped_mm
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        lora_x = self.lora_dropout(x)

        h1_base = torch._grouped_mm(x.bfloat16(), gate_proj.bfloat16().transpose(-2, -1), offs=offsets)
        gate_proj_lora_out = _run_lora_grouped_mm(lora_x, gate_proj_lora_a, gate_proj_lora_b, offsets)
        h1 = h1_base + scaling * gate_proj_lora_out.bfloat16()

        h3_base = torch._grouped_mm(x.bfloat16(), up_proj.bfloat16().transpose(-2, -1), offs=offsets)
        up_proj_lora_out = _run_lora_grouped_mm(lora_x, up_proj_lora_a, up_proj_lora_b, offsets)
        h3 = h3_base + scaling * up_proj_lora_out.bfloat16()

        h = self.base_layer.activation.apply(h1, h3)

        lora_h = self.lora_dropout(h)
        h2_base = torch._grouped_mm(h, down_proj.bfloat16().transpose(-2, -1), offs=offsets)
        down_proj_lora_out = _run_lora_grouped_mm(lora_h, down_proj_lora_a, down_proj_lora_b, offsets)
        out = (h2_base + scaling * down_proj_lora_out.bfloat16()).type_as(x)

        # EP handling: unpermute output back to dispatched token order
        if permuted_indices is not None:
            out_unpermuted = out.new_zeros(input_shape)
            out_unpermuted[permuted_indices, :] = out
            out = out_unpermuted[:-1]

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"n_adapters={self.n_adapters}, num_experts={self.num_experts}, "
            f"alpha={self.alpha}, dropout={self.lora_dropout})"
        )


class MultiLoRANonGatedGroupedExperts(MultiLoRAModule):
    """
    Non-gated GroupedExperts + multi-LoRA with grouped GEMM.
    Adapts the two projections (up_proj and down_proj).
    """

    def __init__(
        self,
        base_layer: GroupedExperts,
        rank: int,
        n_adapters: int,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__(base_layer)
        if rank <= 0 or n_adapters <= 0:
            raise ValueError("rank and n_adapters must be > 0")
        if base_layer.gate_proj is not None:
            raise ValueError("MultiLoRANonGatedGroupedExperts requires non-gated experts")

        self.num_experts = base_layer.num_experts
        # up_proj shape: [num_experts, intermediate_dim, input_dim]
        self.hidden_dim = base_layer.up_proj.shape[1]
        self.dim = base_layer.up_proj.shape[2]

        if rank % 8 != 0 or self.dim % 8 != 0 or self.hidden_dim % 8 != 0:
            raise ValueError("grouped_mm requires rank and expert dimensions divisible by 8")

        self.rank = rank
        self.n_adapters = n_adapters
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._lora_num_tokens = get_lora_num_tokens()
        self._scaling_factors = get_multilora_scaling()

        # up_proj (up: dim -> hidden_dim)
        self.up_proj_lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        rank,
                        self.dim,
                        device=base_layer.up_proj.device,
                        dtype=base_layer.up_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.up_proj_lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.hidden_dim,
                        rank,
                        device=base_layer.up_proj.device,
                        dtype=base_layer.up_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        # down_proj (down: hidden_dim -> dim)
        self.down_proj_lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        rank,
                        self.hidden_dim,
                        device=base_layer.down_proj.device,
                        dtype=base_layer.down_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.down_proj_lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.dim,
                        rank,
                        device=base_layer.down_proj.device,
                        dtype=base_layer.down_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self, index: int | None = None) -> None:
        if index is None:
            for i in range(self.n_adapters):
                self.reset_parameters(i)
        else:
            nn.init.kaiming_uniform_(self.up_proj_lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.up_proj_lora_B[index])
            nn.init.kaiming_uniform_(self.down_proj_lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.down_proj_lora_B[index])

    def named_parameters_for_adapter(self, idx: int) -> list[tuple[str, nn.Parameter]]:
        return [
            ("up_proj_lora_A", self.up_proj_lora_A[idx]),
            ("up_proj_lora_B", self.up_proj_lora_B[idx]),
            ("down_proj_lora_A", self.down_proj_lora_A[idx]),
            ("down_proj_lora_B", self.down_proj_lora_B[idx]),
        ]

    def get_lora_param_counts(self) -> tuple[int, int]:
        adapter_params = (
            self.up_proj_lora_A[0].numel()
            + self.up_proj_lora_B[0].numel()
            + self.down_proj_lora_A[0].numel()
            + self.down_proj_lora_B[0].numel()
        )
        adapted_params = self.base_layer.up_proj.numel() + self.base_layer.down_proj.numel()
        return adapter_params, adapted_params

    def state_dict_for_adapter(self, idx: int) -> dict[str, torch.Tensor]:
        state_dict = {}

        detached_up_proj_lora_a = self.up_proj_lora_A[idx].detach()
        detached_up_proj_lora_b = self.up_proj_lora_B[idx].detach()
        detached_down_proj_lora_a = self.down_proj_lora_A[idx].detach()
        detached_down_proj_lora_b = self.down_proj_lora_B[idx].detach()

        if isinstance(detached_up_proj_lora_a, DTensor):
            detached_up_proj_lora_a = detached_up_proj_lora_a.full_tensor()
            detached_up_proj_lora_b = detached_up_proj_lora_b.full_tensor()
            detached_down_proj_lora_a = detached_down_proj_lora_a.full_tensor()
            detached_down_proj_lora_b = detached_down_proj_lora_b.full_tensor()

        for expert_id in range(self.num_experts):
            state_dict[f"{expert_id}.up_proj.lora_A.weight"] = detached_up_proj_lora_a[expert_id].clone()
            state_dict[f"{expert_id}.up_proj.lora_B.weight"] = detached_up_proj_lora_b[expert_id].clone()
            state_dict[f"{expert_id}.down_proj.lora_A.weight"] = detached_down_proj_lora_a[expert_id].clone()
            state_dict[f"{expert_id}.down_proj.lora_B.weight"] = detached_down_proj_lora_b[expert_id].clone()

        return state_dict

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        adapter_idx = self._lora_num_tokens.argmax().item()
        up_proj_lora_a = self.up_proj_lora_A[adapter_idx]
        up_proj_lora_b = self.up_proj_lora_B[adapter_idx]
        down_proj_lora_a = self.down_proj_lora_A[adapter_idx]
        down_proj_lora_b = self.down_proj_lora_B[adapter_idx]

        scaling = self._scaling_factors[adapter_idx].item()

        up_proj = self.base_layer.up_proj
        down_proj = self.base_layer.down_proj

        permuted_indices = None
        if isinstance(up_proj, DTensor):
            up_proj = up_proj.to_local()
            down_proj = down_proj.to_local()
            up_proj_lora_a = up_proj_lora_a.to_local()
            up_proj_lora_b = up_proj_lora_b.to_local()
            down_proj_lora_a = down_proj_lora_a.to_local()
            down_proj_lora_b = down_proj_lora_b.to_local()

            if getattr(self.base_layer, "ep_comm_backend", "torch") != "deepep":
                from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

                from prime_rl.trainer.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M

                experts_per_ep_rank = up_proj.shape[0]
                num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank

                with torch.no_grad():
                    permuted_indices, num_tokens_per_expert, _ = generate_permute_indices(
                        num_tokens_per_expert,
                        experts_per_ep_rank,
                        num_ep_ranks,
                        x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M,
                        TOKEN_GROUP_ALIGN_SIZE_M,
                    )

                x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
                input_shape = x.shape
                x = x[permuted_indices, :]

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        lora_x = self.lora_dropout(x)

        h_base = torch._grouped_mm(x.bfloat16(), up_proj.bfloat16().transpose(-2, -1), offs=offsets)
        up_proj_lora_out = _run_lora_grouped_mm(lora_x, up_proj_lora_a, up_proj_lora_b, offsets)
        h = self.base_layer.activation.apply(None, h_base + scaling * up_proj_lora_out.bfloat16())

        lora_h = self.lora_dropout(h)
        out_base = torch._grouped_mm(h, down_proj.bfloat16().transpose(-2, -1), offs=offsets)
        down_proj_lora_out = _run_lora_grouped_mm(lora_h, down_proj_lora_a, down_proj_lora_b, offsets)
        out = (out_base + scaling * down_proj_lora_out.bfloat16()).type_as(x)

        if permuted_indices is not None:
            out_unpermuted = out.new_zeros(input_shape)
            out_unpermuted[permuted_indices, :] = out
            out = out_unpermuted[:-1]

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"n_adapters={self.n_adapters}, num_experts={self.num_experts}, "
            f"alpha={self.alpha}, dropout={self.lora_dropout})"
        )


class MultiLoRAGptOssGroupedExperts(MultiLoRAModule):
    """
    GPT-OSS GroupedExperts + multi-LoRA.

    Preserves GPT-OSS's combined gate/up adapter format while applying it to the
    canonical split gate_proj/up_proj runtime weights.
    """

    def __init__(
        self,
        base_layer: GroupedExperts,
        rank: int,
        n_adapters: int,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__(base_layer)
        if rank <= 0 or n_adapters <= 0:
            raise ValueError("rank and n_adapters must be > 0")
        if base_layer.gate_proj is None or base_layer.activation is not ClampedSwiglu:
            raise ValueError("MultiLoRAGptOssGroupedExperts requires gated GPT-OSS experts")
        if any(
            bias is None for bias in (base_layer.gate_proj_bias, base_layer.up_proj_bias, base_layer.down_proj_bias)
        ):
            raise ValueError("GPT-OSS experts require projection biases")

        self.num_experts = base_layer.num_experts
        self.hidden_size = base_layer.up_proj.shape[2]
        self.intermediate_size = base_layer.up_proj.shape[1]
        self.gate_up_out = 2 * self.intermediate_size

        if rank % 8 != 0 or self.hidden_size % 8 != 0 or self.intermediate_size % 8 != 0:
            raise ValueError("grouped_mm requires rank and expert dimensions divisible by 8")

        self.rank = rank
        self.n_adapters = n_adapters
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._lora_num_tokens = get_lora_num_tokens()
        self._scaling_factors = get_multilora_scaling()

        self.gate_up_lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        rank,
                        self.hidden_size,
                        device=base_layer.up_proj.device,
                        dtype=base_layer.up_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.gate_up_lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.gate_up_out,
                        rank,
                        device=base_layer.up_proj.device,
                        dtype=base_layer.up_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        self.down_lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        rank,
                        self.intermediate_size,
                        device=base_layer.down_proj.device,
                        dtype=base_layer.down_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.down_lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_experts,
                        self.hidden_size,
                        rank,
                        device=base_layer.down_proj.device,
                        dtype=base_layer.down_proj.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self, index: int | None = None) -> None:
        if index is None:
            for i in range(self.n_adapters):
                self.reset_parameters(i)
        else:
            nn.init.kaiming_uniform_(self.gate_up_lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.gate_up_lora_B[index])
            nn.init.kaiming_uniform_(self.down_lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.down_lora_B[index])

    def named_parameters_for_adapter(self, idx: int) -> list[tuple[str, nn.Parameter]]:
        return [
            ("gate_up_lora_A", self.gate_up_lora_A[idx]),
            ("gate_up_lora_B", self.gate_up_lora_B[idx]),
            ("down_lora_A", self.down_lora_A[idx]),
            ("down_lora_B", self.down_lora_B[idx]),
        ]

    def get_lora_param_counts(self) -> tuple[int, int]:
        adapter_params = (
            self.gate_up_lora_A[0].numel()
            + self.gate_up_lora_B[0].numel()
            + self.down_lora_A[0].numel()
            + self.down_lora_B[0].numel()
        )
        adapted_params = (
            self.base_layer.gate_proj.numel() + self.base_layer.up_proj.numel() + self.base_layer.down_proj.numel()
        )
        return adapter_params, adapted_params

    def state_dict_for_adapter(self, idx: int) -> dict[str, torch.Tensor]:
        """vLLM-compatible 3D MoE adapter format.

        For 3D MoE models (gpt-oss in vLLM has `is_3d_moe_weight = True`), vLLM expects:
        - `experts.base_layer.lora_{A,B}.weight` for the gate_up projection
        - `experts.lora_{A,B}.weight` for the down projection
        with experts stacked into the rank dim. See
        vllm/lora/model_manager.py::_stack_moe_lora_weights, which reshapes
            lora_A: (num_experts*rank, in)  -> (num_experts, rank, in)
            lora_B: (out, rank*num_experts) -> (out, rank, num_experts) -> (num_experts, out, rank)
        """
        detached_gu_a = self.gate_up_lora_A[idx].detach()
        detached_gu_b = self.gate_up_lora_B[idx].detach()
        detached_d_a = self.down_lora_A[idx].detach()
        detached_d_b = self.down_lora_B[idx].detach()

        if isinstance(detached_gu_a, DTensor):
            detached_gu_a = detached_gu_a.full_tensor()
            detached_gu_b = detached_gu_b.full_tensor()
            detached_d_a = detached_d_a.full_tensor()
            detached_d_b = detached_d_b.full_tensor()

        # lora_A: (num_experts, rank, in) -> (num_experts*rank, in)
        gu_a_flat = detached_gu_a.reshape(self.num_experts * self.rank, self.hidden_size).clone()
        d_a_flat = detached_d_a.reshape(self.num_experts * self.rank, self.intermediate_size).clone()
        # lora_B: (num_experts, out, rank) -> (out, rank, num_experts) -> (out, rank*num_experts)
        # vLLM's reshape treats the last dim of lora_B as (rank, num_experts) with experts fast-varying.
        gu_b_flat = detached_gu_b.permute(1, 2, 0).contiguous().reshape(self.gate_up_out, self.rank * self.num_experts)
        d_b_flat = detached_d_b.permute(1, 2, 0).contiguous().reshape(self.hidden_size, self.rank * self.num_experts)

        return {
            "base_layer.lora_A.weight": gu_a_flat,
            "base_layer.lora_B.weight": gu_b_flat,
            "lora_A.weight": d_a_flat,
            "lora_B.weight": d_b_flat,
        }

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        adapter_idx = self._lora_num_tokens.argmax().item()
        gu_a = self.gate_up_lora_A[adapter_idx]
        gu_b = self.gate_up_lora_B[adapter_idx]
        d_a = self.down_lora_A[adapter_idx]
        d_b = self.down_lora_B[adapter_idx]

        scaling = self._scaling_factors[adapter_idx].item()

        gate_proj = self.base_layer.gate_proj
        up_proj = self.base_layer.up_proj
        down_proj = self.base_layer.down_proj
        gate_proj_bias = self.base_layer.gate_proj_bias
        up_proj_bias = self.base_layer.up_proj_bias
        down_proj_bias = self.base_layer.down_proj_bias

        permuted_indices = None
        if isinstance(up_proj, DTensor):
            gate_proj = gate_proj.to_local()
            up_proj = up_proj.to_local()
            down_proj = down_proj.to_local()
            gate_proj_bias = gate_proj_bias.to_local()
            up_proj_bias = up_proj_bias.to_local()
            down_proj_bias = down_proj_bias.to_local()
            gu_a = gu_a.to_local()
            gu_b = gu_b.to_local()
            d_a = d_a.to_local()
            d_b = d_b.to_local()

            if getattr(self.base_layer, "ep_comm_backend", "torch") != "deepep":
                from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

                from prime_rl.trainer.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M

                experts_per_ep_rank = up_proj.shape[0]
                num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank

                with torch.no_grad():
                    permuted_indices, num_tokens_per_expert, _ = generate_permute_indices(
                        num_tokens_per_expert,
                        experts_per_ep_rank,
                        num_ep_ranks,
                        x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M,
                        TOKEN_GROUP_ALIGN_SIZE_M,
                    )

                x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
                input_shape = x.shape
                x = x[permuted_indices, :]

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        lora_x = self.lora_dropout(x)

        gate = torch._grouped_mm(x.bfloat16(), gate_proj.bfloat16().transpose(-2, -1), offs=offsets)
        gate = gate + broadcast_expert_bias(gate_proj_bias, num_tokens_per_expert, gate.shape[0]).bfloat16()
        up = torch._grouped_mm(x.bfloat16(), up_proj.bfloat16().transpose(-2, -1), offs=offsets)
        up = up + broadcast_expert_bias(up_proj_bias, num_tokens_per_expert, up.shape[0]).bfloat16()

        gate_up_lora = _run_lora_grouped_mm(lora_x, gu_a, gu_b, offsets)
        gate = gate + scaling * gate_up_lora[..., ::2].bfloat16()
        up = up + scaling * gate_up_lora[..., 1::2].bfloat16()

        h = self.base_layer.activation.apply(gate, up)
        lora_h = self.lora_dropout(h)

        out_base = torch._grouped_mm(h, down_proj.bfloat16().transpose(-2, -1), offs=offsets)
        out_base = out_base + broadcast_expert_bias(down_proj_bias, num_tokens_per_expert, out_base.shape[0]).bfloat16()
        out_lora = _run_lora_grouped_mm(lora_h, d_a, d_b, offsets)
        out = (out_base + scaling * out_lora.bfloat16()).type_as(x)

        if permuted_indices is not None:
            out_unpermuted = out.new_zeros(input_shape)
            out_unpermuted[permuted_indices, :] = out
            out = out_unpermuted[:-1]

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"n_adapters={self.n_adapters}, num_experts={self.num_experts}, "
            f"alpha={self.alpha}, dropout={self.lora_dropout})"
        )

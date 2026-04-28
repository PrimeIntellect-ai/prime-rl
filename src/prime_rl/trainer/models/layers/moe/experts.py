import torch
from torch import nn
from torchtitan.distributed.expert_parallel import expert_parallel

from prime_rl.configs.trainer import EPCommBackend, ExpertBackend
from prime_rl.trainer.models.layers.moe.kernels import (
    _disable_sonic_autotune,
    _run_experts_grouped_mm_impl,
    _run_experts_sonic_impl,
    _run_gpt_oss_experts_grouped_mm,
    _run_gpt_oss_experts_grouped_mm_impl,
    _run_nongated_experts_grouped_mm,
    _run_nongated_experts_grouped_mm_impl,
)


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.ep_comm_backend: EPCommBackend = "torch"
        self.expert_backend: ExpertBackend = "grouped_mm"
        self._forward_fn = expert_parallel(_run_experts_grouped_mm_impl)

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend
        if self.ep_comm_backend == "deepep":
            self._forward_fn = _run_experts_grouped_mm_impl

    def set_expert_backend(self, backend: ExpertBackend) -> None:
        self.expert_backend = backend
        if self.expert_backend == "sonic":
            _disable_sonic_autotune()
            self._forward_fn = _run_experts_sonic_impl

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
        *,
        top_scores: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
        expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.ep_comm_backend == "deepep":
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.expert_backend == "sonic":
            args = (w1, w2, w3, x, top_scores, token_indices, expert_indices)
        else:
            args = (w1, w2, w3, x, num_tokens_per_expert)

        return self._forward_fn(*args)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class NonGatedGroupedExperts(nn.Module):
    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_dim, input_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, input_dim, intermediate_dim))
        # Dummy w3 for @expert_parallel decorator compatibility (expects w1, w2, w3 signature)
        self.w3 = nn.Parameter(torch.empty(0))
        self.ep_comm_backend: EPCommBackend = "torch"
        self.expert_backend: ExpertBackend = "grouped_mm"

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend

    def set_expert_backend(self, backend: ExpertBackend) -> None:
        assert backend == "grouped_mm", "NonGatedGroupedExperts only supports model.expert_backend='grouped_mm'."
        self.expert_backend = backend

    def _forward_deepep(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        w1 = self.w1.to_local()
        w2 = self.w2.to_local()
        w3 = self.w3.to_local()
        return _run_nongated_experts_grouped_mm_impl(w1, w2, w3, x, num_tokens_per_expert)

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_comm_backend == "deepep":
            return self._forward_deepep(x, num_tokens_per_expert)

        return _run_nongated_experts_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)


class GptOssGroupedExperts(nn.Module):
    """GPT-OSS-style grouped experts.

    Mirrors HF's `GptOssExperts` parameter naming (gate_up_proj/down_proj plus per-expert
    biases, fused interleaved gate/up channels) so the unsloth BF16 checkpoint loads with
    no key conversion. Forward signature matches `GroupedExperts` (`x`, `num_tokens_per_expert`)
    so the surrounding MoE plumbing and LoRA wrapper follow the same convention.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        assert use_grouped_mm, "GptOssGroupedExperts only supports use_grouped_mm=True"
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, hidden_size, 2 * intermediate_size))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.down_proj_bias = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.ep_comm_backend: EPCommBackend = "torch"

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend

    def _forward_deepep(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        gate_up_proj = self.gate_up_proj.to_local()
        gate_up_proj_bias = self.gate_up_proj_bias.to_local()
        down_proj = self.down_proj.to_local()
        down_proj_bias = self.down_proj_bias.to_local()
        return _run_gpt_oss_experts_grouped_mm_impl(
            gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, x, num_tokens_per_expert
        )

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        if self.ep_comm_backend == "deepep":
            return self._forward_deepep(x, num_tokens_per_expert)

        return _run_gpt_oss_experts_grouped_mm(
            self.gate_up_proj, self.gate_up_proj_bias, self.down_proj, self.down_proj_bias, x, num_tokens_per_expert
        )

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_up_proj_bias)
        nn.init.trunc_normal_(self.down_proj, mean=0.0, std=init_std)
        nn.init.zeros_(self.down_proj_bias)

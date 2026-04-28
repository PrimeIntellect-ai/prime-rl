import torch
from torch import nn
from torchtitan.distributed.expert_parallel import expert_parallel

from prime_rl.configs.trainer import EPCommBackend, ExpertBackend
from prime_rl.trainer.models.layers.moe.kernels import (
    _disable_sonic_autotune,
    _run_experts_grouped_mm_impl,
    _run_experts_sonic_impl,
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

        args = (w1, w2, w3, x, num_tokens_per_expert)
        if self.expert_backend == "sonic":
            args += (top_scores, token_indices, expert_indices)

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

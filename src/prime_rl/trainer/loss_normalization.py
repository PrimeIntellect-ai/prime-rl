import torch
import torch.distributed as dist
from torch import Tensor

from prime_rl.configs.shared import LossNormalization
from prime_rl.trainer.parallel_dims import ParallelDims


class LossNormalizer:
    """Match loss counts to the context-parallel layout of the loss numerator."""

    def __init__(self, mode: LossNormalization, parallel_dims: ParallelDims, *, loss_replicated_across_cp: bool):
        self.mode = mode
        self.parallel_dims = parallel_dims
        self.loss_replicated_across_cp = loss_replicated_across_cp

    def local_count(self, mask: Tensor, seq_lens: Tensor | list[int]) -> int:
        if self.mode == "token":
            return int(mask.sum())
        lengths = seq_lens.tolist() if isinstance(seq_lens, Tensor) else seq_lens
        return sum(bool(part.any()) for part in mask.flatten().split(lengths))

    def weights(self, mask: Tensor, seq_lens: Tensor | list[int], *, require_nonempty: bool = False) -> Tensor | None:
        if self.mode == "token":
            return None
        lengths = seq_lens.tolist() if isinstance(seq_lens, Tensor) else seq_lens
        counts = torch.stack([part.sum() for part in mask.flatten().split(lengths)]).to(torch.float32)
        if require_nonempty and bool((counts == 0).any()):
            raise ValueError("Each packed sample must have at least one trainable token")
        weights = torch.repeat_interleave(
            counts.clamp_min(1).reciprocal(), torch.as_tensor(lengths, device=mask.device), output_size=mask.numel()
        )
        return weights.view_as(mask) * mask

    def global_counts(self, local_counts: Tensor, *, counts_replicated_across_cp: bool) -> Tensor:
        counts = local_counts.clone()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=self.parallel_dims.get_mesh("dp_cp").get_group())
        if counts_replicated_across_cp and not self.loss_replicated_across_cp:
            counts //= self.parallel_dims.cp
        elif self.loss_replicated_across_cp and not counts_replicated_across_cp:
            counts *= self.parallel_dims.cp
        return counts

    def gradient_scale(self, count: int, *, grad_accum_steps: int = 1) -> float:
        if count == 0:
            return 1.0
        return self.parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / count

import time
from collections.abc import Sequence

import torch
from torch import nn

from prime_rl.trainer.models.base import AttentionFlops, PreTrainedModelPrimeRL, count_active_params
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class FlopCounter:
    """Training FLOPs of packed documents.

    6 FLOPs per active parameter per token, plus each sequence-mixing layer's FLOPs per document (a
    token attends only within its own document). The model is walked once, at construction.

    `flops_per_token_6dn` is the plain 6 * active params estimate (no attention, no LoRA discount), for
    MFU comparable with the common 6ND convention.
    """

    def __init__(self, model: PreTrainedModelPrimeRL):
        active_params = model.num_active_params()
        self.flops_per_token_6dn = 6 * active_params
        lora_params = [param for name, param in model.named_parameters() if "lora_A" in name or "lora_B" in name]
        if lora_params:
            # Frozen base matmuls cost forward (2x) and dX (2x); adapters cost 6x.
            lora_active_params = count_active_params(model, lora_params)
            self.matmul_flops_per_token = 4 * (active_params - lora_active_params) + 6 * lora_active_params
        else:
            self.matmul_flops_per_token = 6 * active_params
        self.attention_modules = [module for module in model.modules() if isinstance(module, AttentionFlops)]

    def __call__(self, seq_lens: Sequence[int]) -> int:
        return sum(
            n * self.matmul_flops_per_token + sum(module.attention_flops(n) for module in self.attention_modules)
            for n in seq_lens
        )


class PerfCounter:
    """
    Computes throughput (tokens/s) and MFU.

    Two modes:
    - Sliding window over full-step wall time: `count_tokens(tokens)` each step,
      read smoothed values via `get_tokens_per_second()`.
      Time is measured between successive `count_tokens` calls (full step).
    - Single point on a caller-provided duration: `get_step_tokens_per_second(tokens, fwd_bwd_time)`
      / `get_step_mfu(flops, fwd_bwd_time)`, with `flops` summed from `flop_counter(seq_lens)`. No smoothing.

    Sliding window inspired by https://github.com/pytorch/torchtitan/blob/4b3f2e41a084bf79a8540068ed525539d1244edd/torchtitan/utils.py#L119
    """

    def __init__(self, model: nn.Module, window_size: int = 10):
        self.window_size = window_size
        self.tokens: list[int] = []
        self.times: list[float] = []

        self._world = get_world()
        self._logger = get_logger()

        if torch.cuda.is_available():
            self.gpu_peak_flops = self._get_peak_flops(torch.cuda.get_device_name(torch.device("cuda")))
        else:
            self.gpu_peak_flops = 0
        self.flop_counter = FlopCounter(model)

    def count_tokens(self, tokens: int) -> None:
        """Push a step into the sliding window. Time is recorded internally."""
        self.tokens.append(tokens)
        self.times.append(time.perf_counter())
        if len(self.tokens) > self.window_size:
            self.tokens.pop(0)
            self.times.pop(0)

    def get_tokens_per_second(self) -> float | None:
        if len(self.tokens) < 2:
            return None
        return sum(self.tokens[1:]) / (self.times[-1] - self.times[0])

    def get_step_tokens_per_second(self, tokens: int, fwd_bwd_time: float) -> float:
        """Single-step throughput from a caller-provided duration."""
        return tokens / fwd_bwd_time

    def get_step_mfu(self, flops: float, fwd_bwd_time: float) -> float:
        return 100 * flops / fwd_bwd_time / self.gpu_peak_flops / self._world.world_size

    def _get_peak_flops(self, device_name: str) -> float:
        """
        Peak BF16 FLOPs (without sparsity)

        From: https://github.com/pytorch/torchtitan/blob/05e47c38d99fdb1dd39aeba76f080e529a425c5c/torchtitan/tools/utils.py#L69
        """
        if "A100" in device_name:
            # https://www.nvidia.com/en-us/data-center/a100/
            return 312e12
        if "H100" in device_name or "H200" in device_name:
            # https://www.nvidia.com/en-us/data-center/h100/
            # https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/hpc-datasheet-sc23-h200
            if "NVL" in device_name:
                return 835e12
            elif "PCIe" in device_name:
                return 756e12
            else:  # For H100 SXM and other variants
                return 989e12
        if "GB200" in device_name or "GB300" in device_name:
            # Grace Blackwell Superchips, dense BF16 per GPU: 2,500 TFLOPS
            # https://www.nvidia.com/en-us/data-center/dgx-gb200/
            # https://www.nvidia.com/en-us/data-center/dgx-gb300/
            return 2.5e15
        if "B200" in device_name or "B300" in device_name:
            # https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
            # Checked after GB200/GB300 to avoid false match on "GB300"
            return 2.25e15  # This is half of the FLOPS reported in torchtitan
        # AMD Instinct GPUs
        if "MI300X" in device_name:
            # https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
            # Peak BF16: 1307.4 TFLOPS (matrix)
            return 1307.4e12
        if "MI325X" in device_name:
            # https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
            # Peak BF16: 1307.4 TFLOPS (matrix) - same compute dies as MI300X, more HBM3e
            return 1307.4e12
        else:
            self._logger.warning(f"Peak FLOPS undefined for `{device_name}`. Falling back to A100 (312 TFLOPS)")
            return 312e12


_PERF_COUNTER: PerfCounter | None = None


def get_perf_counter(model: nn.Module, window_size: int = 10) -> PerfCounter:
    global _PERF_COUNTER
    if _PERF_COUNTER is None:
        _PERF_COUNTER = PerfCounter(model, window_size)

    return _PERF_COUNTER

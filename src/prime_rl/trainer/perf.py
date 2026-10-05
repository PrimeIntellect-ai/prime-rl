import time
from fractions import Fraction

import torch
from torch import nn

from prime_rl.trainer.lora import has_lora_layers
from prime_rl.trainer.models.layers.lora import MultiLoRAModule
from prime_rl.trainer.models.layers.moe import MoE
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


def get_active_params(model: nn.Module) -> int:
    """Matmul parameters used per token, as torchtitan's `get_nparams_and_active_nparams`.

    Routed experts count by `top_k / num_experts`; embeddings count only when tied to the lm_head;
    the vision encoder and LoRA adapters do not count.
    """
    weights = {id(param): Fraction(1) for param in model.parameters()}
    for module in model.modules():
        if isinstance(module, MoE):
            for param in module.experts.parameters():
                weights[id(param)] = Fraction(module.router.top_k, module.experts.num_experts)
    lm_head_params = {id(param) for param in model.lm_head.parameters()}
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for param in module.parameters(recurse=False):
                if id(param) not in lm_head_params:
                    weights[id(param)] = Fraction(0)
    visual = getattr(model.model, "visual", None)
    if visual is not None:
        for param in visual.parameters():
            weights[id(param)] = Fraction(0)
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            weights[id(param)] = Fraction(0)
    return int(sum(param.numel() * weights[id(param)] for param in model.parameters()))


class PerfCounter:
    """
    Computes throughput (tokens/s) and MFU.

    Two modes:
    - Sliding window over full-step wall time: `count_tokens(tokens)` each step,
      read smoothed values via `get_tokens_per_second()` / `get_mfu()`.
      Time is measured between successive `count_tokens` calls (full step).
    - Single point on a caller-provided duration: `get_step_tokens_per_second(tokens, fwd_bwd_time)`
      / `get_step_mfu(tokens, fwd_bwd_time)`. No smoothing.

    Sliding window inspired by https://github.com/pytorch/torchtitan/blob/4b3f2e41a084bf79a8540068ed525539d1244edd/torchtitan/utils.py#L119
    """

    def __init__(self, model: nn.Module, seq_len: int, window_size: int = 10):
        self.window_size = window_size
        self.tokens: list[int] = []
        self.times: list[float] = []
        self.model = model

        self._world = get_world()
        self._logger = get_logger()

        if torch.cuda.is_available():
            self.gpu_peak_flops = self._get_peak_flops(torch.cuda.get_device_name(torch.device("cuda")))
        else:
            self.gpu_peak_flops = 0
        self.num_flop_per_token = self._get_num_flop_per_token(seq_len)

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

    def get_mfu(self) -> float | None:
        tokens_per_second = self.get_tokens_per_second()
        if tokens_per_second is None:
            return None
        return self._mfu_from_tps(tokens_per_second)

    def get_step_tokens_per_second(self, tokens: int, fwd_bwd_time: float) -> float:
        """Single-step throughput from a caller-provided duration."""
        return tokens / fwd_bwd_time

    def get_step_mfu(self, tokens: int, fwd_bwd_time: float) -> float:
        return self._mfu_from_tps(self.get_step_tokens_per_second(tokens, fwd_bwd_time))

    def _mfu_from_tps(self, tokens_per_second: float) -> float:
        return 100 * self.num_flop_per_token * tokens_per_second / self.gpu_peak_flops / self._world.world_size

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

    def _get_num_flop_per_token(self, seq_len: int) -> int:
        """Training FLOPs per token, as torchtitan's `get_nparams_and_flops`: 6 FLOPs per active matmul
        parameter plus each layer's own attention term (`attention_flops_per_token` hooks)."""
        attention_flops = sum(
            module.attention_flops_per_token(seq_len)
            for module in self.model.modules()
            if hasattr(module, "attention_flops_per_token")
        )
        active_params = get_active_params(self.model)
        if not has_lora_layers(self.model):
            return 6 * active_params + attention_flops
        # LoRA: frozen base matmuls still cost forward (2x) and dX (2x); fully trainable non-LoRA
        # params (modules_to_save) add their weight gradient (2x); adapters cost 6x.
        return (
            4 * active_params
            + 2 * self._count_fully_trainable_params_excluding_lora()
            + 6 * self._count_lora_adapter_params()
            + attention_flops
        )

    def _count_lora_adapter_params(self) -> int:
        """Count LoRA adapter parameters (sum of lora_A and lora_B across all MultiLoRAModules)."""
        params = 0
        for module in self.model.modules():
            if isinstance(module, MultiLoRAModule):
                adapter_params, _ = module.get_lora_param_counts()
                params += adapter_params
        return params

    def _count_fully_trainable_params_excluding_lora(self) -> int:
        """Count trainable parameters excluding LoRA adapter tensors.

        Approximates trainable matmul params in modules_to_save by subtracting LoRA adapter params
        from all trainable params.
        """
        total_trainable = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and ("lora_A" not in name and "lora_B" not in name):
                total_trainable += param.numel()
        return total_trainable


_PERF_COUNTER: PerfCounter | None = None


def get_perf_counter(model: nn.Module, seq_len: int, window_size: int = 10) -> PerfCounter:
    global _PERF_COUNTER
    if _PERF_COUNTER is None:
        _PERF_COUNTER = PerfCounter(model, seq_len, window_size)

    return _PERF_COUNTER

import time

import torch

from prime_rl.trainer.model import Model
from prime_rl.trainer.world import get_world


class PerfCounter:
    """
    A class to count throughput (tokens/s) with a rolling window to obtain
    precise throughput and MFU estimates.

    Inspired from https://github.com/pytorch/torchtitan/blob/4b3f2e41a084bf79a8540068ed525539d1244edd/torchtitan/utils.py#L119
    """

    def __init__(self, model: Model, seq_len: int, window_size: int):
        self.window_size = window_size
        self.tokens = []
        self.times = []
        self.model = model

        self.gpu_peak_flops = self._get_peak_flops(torch.cuda.get_device_name(torch.device("cuda")))
        self.num_params = self._get_num_params(model, exclude_embedding=True)
        self.num_flop_per_token = self._get_num_flop_per_token(self.num_params, model.config, seq_len=seq_len)

        self._world = get_world()

    def count_tokens(self, tokens: int):
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
        return 100 * self.num_flop_per_token * tokens_per_second / self.gpu_peak_flops / self._world.world_size

    def _get_peak_flops(self, device_name: str) -> float:
        """
        Peak BF16 MatMul FLOPs

        stolen from https://github.com/pytorch/torchtitan/blob/05e47c38d99fdb1dd39aeba76f080e529a425c5c/torchtitan/tools/utils.py#L69
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
        if "B200" in device_name:
            # data from https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
            return 2.25e15
        # Default to A100
        else:
            return 312e12

    # TODO: Add config type
    def _get_num_flop_per_token(self, num_params: int, model_config, seq_len: int) -> int:
        l, h, q, t = (  # noqa: E741
            model_config.num_hidden_layers,
            model_config.num_attention_heads,
            model_config.hidden_size // model_config.num_attention_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        flop_per_token = 6 * num_params + 12 * l * h * q * t

        return flop_per_token

    def _get_num_params(self, model: Model, exclude_embedding: bool = False) -> int:
        num_params = sum(p.numel() for p in model.parameters())
        if exclude_embedding:
            num_params -= model.lm_head.weight.numel()
        return num_params


_PERF_COUNTER: PerfCounter | None = None


def get_perf_counter(model: Model, seq_len: int, window_size: int = 10) -> PerfCounter:
    global _PERF_COUNTER
    if _PERF_COUNTER is None:
        _PERF_COUNTER = PerfCounter(model, seq_len, window_size)

    return _PERF_COUNTER

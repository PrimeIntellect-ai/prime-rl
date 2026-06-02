import torch
from torch import Tensor, nn
import torch.nn.functional as F
from prime_rl.trainer.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)
from typing import Callable

# Flash attention imports
try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn import flash_attn_func as fa2_func
except ImportError:
    flash_attn_varlen_func = None  # type: ignore
    fa2_func = None  # type: ignore

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
except ImportError:
    flash_attn_3_varlen_func = None  # type: ignore

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
except ImportError:
    flash_attn_4_varlen_func = None  # type: ignore


from prime_rl.utils.logger import get_logger


class DeepSeekAttentionCore:

    _flash_attn_version_mapper = {
        "flash_attention_2": 2,
        "flash_attention_3": 3,
        "fa4": 4,
    }

    _funcs = {
        2: flash_attn_varlen_func,
        3: flash_attn_3_varlen_func,
        4: flash_attn_4_varlen_func,
    }

    def __init__(self, config: DeepseekV3Config):

        self.func: Callable | None = None
        self._flash_attn_version: int = -1

        self.num_queries_per_kv = (
            config.num_attention_heads // config.num_key_value_heads
        )

        attn_impl = config._attn_implementation
        if attn_impl in ("eager", "sdpa"):
            self.attn_impl = "sdpa"
        elif attn_impl in self._flash_attn_version_mapper:
            # flash attention
            self.attn_impl = config._attn_implementation
            self._flash_attn_version = self._flash_attn_version_mapper[attn_impl]
            self.func = self._funcs[self._flash_attn_version]
            self._flash_attn_call = self.func
            if self._flash_attn_version == 4:
                self._flash_attn_call = torch._dynamo.disable(self.func)
        else:
            raise ValueError(
                f"Not supportted attention '{config._attn_implementation}'. "
            )

    def _compute_attention(
        self, q, k, v, cu_seqlens, max_seqlen, softmax_scale: float | None = None
    ):
        ### !! MUST BE PATCHED BY RING_ATTN

        args = [q, k, v, cu_seqlens, cu_seqlens]
        if self._flash_attn_version != 4:
            args.extend([max_seqlen, max_seqlen])

        kwargs: dict = {"causal": True}

        if softmax_scale:
            kwargs["softmax_scale"] = softmax_scale

        out = self._flash_attn_call(*args, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _attention_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.LongTensor | None,
        max_seqlen: int | None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Inputs:
        # q,k,v = (bs, sl, nkv, d)
        """

        if cu_seqlens is None:
            # self.attn_impl == 'sdpa'
            # q,k,v: (batch_size, seqlen, nheads, headdim)
            num_queries_per_kv = self.num_queries_per_kv
            if num_queries_per_kv > 1:
                k = k.repeat_interleave(num_queries_per_kv, dim=2)
                v = v.repeat_interleave(num_queries_per_kv, dim=2)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # q,k,v = (bs, nkv, sl, d)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=scale
            )
            attn_output = attn_output.transpose(1, 2)
        elif q.shape[0] > 1:
            # self.attn_impl == 'flash_attention'
            attn_output = fa2_func(q, k, v, causal=True, softmax_scale=scale)
        else:
            # Varlen Attention
            # inputs (bs==1, sl, nkv, d)
            attn_output = self._compute_attention(
                q[0], k[0], v[0], cu_seqlens, max_seqlen, softmax_scale=scale
            )
            attn_output = attn_output.unsqueeze(0)

        return attn_output.contiguous()

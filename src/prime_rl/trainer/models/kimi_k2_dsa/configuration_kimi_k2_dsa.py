import warnings

from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import _index_cache_skip_topk
from prime_rl.trainer.models.kimi_k2.configuration_kimi_k2 import KimiK2Config

__all__ = ["KimiK2DsaConfig", "_index_cache_skip_topk"]


class KimiK2DsaConfig(KimiK2Config):
    r"""
    Kimi K2 converted to DeepSeek Sparse Attention (DSA) — adds a lightning indexer and the
    dense/sparse attention toggle used for DSA conversion (see `docs/advanced.md`'s "DSA
    Conversion" section) on top of `KimiK2Config`'s MLA/MoE fields.

    Args:
        index_n_heads (`int`, defaults to `32`):
            Number of heads used by the sparse indexer.
        index_head_dim (`int`, defaults to `128`):
            Head dimension used by the sparse indexer.
        index_topk (`int`, defaults to `2048`):
            Number of top tokens selected by the sparse indexer. Must be a multiple of 64.
        use_index_cache (`bool`, defaults to `False`):
            Whether to reuse sparse attention top-k indices across DSA layers (IndexCache).
        index_topk_freq (`int`, defaults to `1`):
            Frequency for recomputing top-k indices when IndexCache is enabled.
        index_topk_pattern (`str`, *optional*):
            Optional per-layer pattern where `"F"` computes fresh indices and `"S"` reuses
            the cached indices from the previous full layer.
        indexer_types (`list[str]`, *optional*):
            Optional native per-layer `"full"`/`"shared"` IndexShare schedule.
        use_sparse_attn (`bool`, defaults to `True`):
            Whether to attend only over the indexer's top-k selection (DSA). Set to `False`
            to run ordinary dense causal attention over the full sequence instead — used to
            convert a dense Kimi-K2 checkpoint to DSA via continued pretraining.
        train_indexer (`bool`, defaults to `False`):
            Compute a differentiable indexer-vs-attention KL loss term each forward pass
            (see `compute_indexer_kl_loss`), for training the indexer during DSA conversion.
    """

    model_type = "kimi_k2_dsa"

    def __init__(
        self,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        use_index_cache=False,
        index_topk_freq=1,
        index_topk_pattern=None,
        indexer_types=None,
        use_sparse_attn=True,
        train_indexer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.use_index_cache = use_index_cache
        self.index_topk_freq = index_topk_freq
        self.index_topk_pattern = index_topk_pattern
        self.indexer_types = indexer_types
        self.use_sparse_attn = use_sparse_attn
        self.train_indexer = train_indexer

        if index_topk % 64 != 0:
            warnings.warn(f"index_topk should be a multiple of 64 (block_I), got {index_topk}")

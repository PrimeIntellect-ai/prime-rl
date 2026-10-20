from transformers.configuration_utils import PretrainedConfig


class PrimeMoeConfig(PretrainedConfig):
    r"""
    Configuration of the pretraining MoE (`PrimeMoeForCausalLM`): grouped-query attention (optional QK norm, gated
    attention, exclusive self attention), MoE layers with sigmoid-scored token-choice routed experts, a selection bias,
    shared experts and optional null experts or LatentMoE, after leading dense layers, and optional Gated Residual
    connections. Fields that also exist in the Hugging Face MoE configs keep their names there.

    Args:
        vocab_size (`int`): Vocabulary size.
        hidden_size (`int`): Width of the residual stream.
        intermediate_size (`int`): Hidden size of the dense MLP layers.
        num_hidden_layers (`int`): Number of decoder layers.
        num_attention_heads (`int`): Number of query heads.
        num_key_value_heads (`int`): Number of key/value heads (grouped-query attention).
        head_dim (`int`, *optional*): Dimension of each attention head. Defaults to `hidden_size // num_attention_heads`.
        partial_rotary_factor (`float`): Share of each head's dimensions that RoPE rotates.
        hidden_act (`str`): Activation of the gated MLPs.
        max_position_embeddings (`int`): Maximum sequence length.
        rms_norm_eps (`float`): Epsilon of the RMS norms.
        tie_word_embeddings (`bool`): Whether the LM head shares the input embedding.
        rope_theta (`float`): RoPE base period.
        rope_scaling (`dict`, *optional*): RoPE scaling configuration.
        attention_bias (`bool`): Whether the attention projections have biases.
        moe_intermediate_size (`int`): Hidden size of each routed and shared expert.
        num_experts_per_tok (`int`): Experts each token selects (`top_k`), null experts included.
        n_shared_experts (`int`): Number of shared experts; their hidden sizes add up.
        shared_expert_intermediate_size (`int`, *optional*): Hidden size of the shared experts together. Defaults to
            `moe_intermediate_size * n_shared_experts`.
        n_routed_experts (`int`): Number of routed experts.
        routed_scaling_factor (`float`): Scale of the routed experts' output.
        first_k_dense_replace (`int`): Number of leading dense layers.
        norm_topk_prob (`bool`): Whether the selected experts' scores are normalized to sum to 1.
        use_qk_norm (`bool`): Whether queries and keys are RMS-normalized.
        hc_count (`int`): Number of residual streams. Above 1, Gated Residual connections (Qwen3.8-Next) replace the
            pre-norms and the final norm.
        hc_lowrank (`int`, *optional*): Rank of the Gated Residual read gates; required when `hc_count` is above 1.
        hc_stream_width (`int`, *optional*): Width of each residual stream; a divisor of `hidden_size`, which each
            read then assembles from `hidden_size / hc_stream_width` chunks. Defaults to `hidden_size`.
        hc_qkv_reads (`bool`): Separate Gated Residual reads (gates) for the attention's query, key and value inputs.
        hc_moe_reads (`bool`): Separate reads for the MoE router and the experts (routed and shared).
        hc_write (`str`, *optional*): Gated Residual write variant of sequential blocks: `norm` (RMS-normalize the block
            output before the write), `output` (write scalars also from the normalized block output) or `channel` (a
            gate per stream and channel instead of one scalar per stream). None for the plain write.
        hc_parallel_block (`str`, *optional*): Parallel attention and MLP with Gated Residual: `shared_read` (both
            take the same read; the MLP the query read with `hc_qkv_reads`) or `separate_reads` (an own read for the
            MLP); both outputs are written to all streams with their own scalars. None for sequential blocks.
        hc_qkv_conv (`int`): Kernel size of causal depthwise convolutions after the attention read, one per query, key
            and value input; 0 for none.
        hc_stream_inputs (`str`): What the residual streams start from: `copy` (the token embedding in every stream),
            `tables` (one token embedding table per stream), `ngram` (the last stream hashed token bi- and trigrams,
            the others token tables), `bytes` (the last stream the token's preceding UTF-8 bytes, the others token
            tables) or several of `ngram` / `bytes` joined by `_` (the last streams, e.g. `ngram_bytes`,
            `ngram_ngram`, `ngram_ngram_ngram`, `bytes_bytes`; a repeated kind is an independent copy).
        hc_byte_add (`str`, *optional*): Add an own byte embedding (`hc_byte_window` bytes, as the byte stream) to the
            input of every token-table stream (`tables`), every n-gram stream (`ngrams`) or both (`all`); None for no
            addition.
        hc_ngram_vocab_size (`int`, *optional*): Rows per hash head of the n-gram stream.
        hc_byte_window (`int`, *optional*): Bytes per token of the byte stream.
        hc_byte_table (`str`, *optional*): Path of the tokenizer's token-to-bytes table for the byte stream.
        n_null_experts (`int`): Zero-output experts the router can select next to the routed experts (AdaMoE); a
            token that selects them uses fewer routed experts.
        attn_output_gate (`bool`): Elementwise sigmoid gate on the attention output, computed from the block input
            (gated attention, Qwen3-Next).
        exclusive_attention (`bool`): Exclusive self attention (XSA): remove from each head's output its component
            along the token's own value vector.
        latent_dim (`int`, *optional*): LatentMoE: the routed experts work in this width (`hidden_size -> latent_dim`
            projection and RMS norm before them, `latent_dim -> hidden_size` projection after the combine), with hidden
            size `moe_intermediate_size`. The router and the shared experts stay full width.
        latent_norm (`str`): Where LatentMoE's RMS norm sits: `"input"` on the projected expert input (Marin) or
            `"output"` on the combined expert output before the up-projection (Kimi K3's Stable LatentMoE).
        expert_activation (`str`, *optional*): Activation of the routed and shared experts: `"situ_glu"` (Kimi K3's
            soft-capped SwiGLU). Defaults to `hidden_act`.
        router_bias_on_logits (`bool`): Add the selection bias to the router logits instead of the sigmoid scores
            (the space quantile balancing, the default, works best in).
        pad_token_id (`int`, *optional*): Padding token id, the input embedding's `padding_idx`.
    """

    model_type = "prime_moe"

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=4096,
        intermediate_size=10944,
        num_hidden_layers=46,
        num_attention_heads=96,
        partial_rotary_factor=0.5,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        moe_intermediate_size=1408,
        num_experts_per_tok=8,
        n_shared_experts=1,
        n_routed_experts=128,
        routed_scaling_factor=1.0,
        shared_expert_intermediate_size=None,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        use_qk_norm=False,
        hc_count=1,
        hc_lowrank=None,
        hc_stream_width=None,
        hc_qkv_reads=False,
        hc_moe_reads=False,
        hc_qkv_conv=0,
        hc_write=None,
        hc_parallel_block=None,
        hc_stream_inputs="copy",
        hc_byte_add=None,
        hc_ngram_vocab_size=None,
        hc_byte_window=None,
        hc_byte_table=None,
        n_null_experts=0,
        attn_output_gate=False,
        exclusive_attention=False,
        latent_dim=None,
        latent_norm="input",
        expert_activation=None,
        router_bias_on_logits=True,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.partial_rotary_factor = partial_rotary_factor
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.pad_token_id = pad_token_id

        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.shared_expert_intermediate_size = (
            shared_expert_intermediate_size
            if shared_expert_intermediate_size is not None
            else moe_intermediate_size * n_shared_experts
        )
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.use_qk_norm = use_qk_norm

        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.hc_stream_width = hc_stream_width
        self.hc_qkv_reads = hc_qkv_reads
        self.hc_moe_reads = hc_moe_reads
        self.hc_qkv_conv = hc_qkv_conv
        self.hc_write = hc_write
        self.hc_parallel_block = hc_parallel_block
        self.hc_stream_inputs = hc_stream_inputs
        self.hc_byte_add = hc_byte_add
        self.hc_ngram_vocab_size = hc_ngram_vocab_size
        self.hc_byte_window = hc_byte_window
        self.hc_byte_table = hc_byte_table
        self.n_null_experts = n_null_experts
        self.attn_output_gate = attn_output_gate
        self.exclusive_attention = exclusive_attention
        self.latent_dim = latent_dim
        self.latent_norm = latent_norm
        self.expert_activation = expert_activation
        self.router_bias_on_logits = router_bias_on_logits

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

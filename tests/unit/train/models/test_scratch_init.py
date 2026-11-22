import pytest
import torch

from prime_rl.trainer.models import _CUSTOM_CAUSAL_LM_MODELS, AutoModelForCausalLMPrimeRL

# Narrow widths and few experts; the layer count stays the default so every layer type is built.
TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=64,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    moe_shared_expert_intermediate_size=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    head_dim=64,
    q_lora_rank=16,
    o_lora_rank=16,
    o_groups=2,
    index_n_heads=2,
    indexer_n_heads=2,
    num_experts=8,
    n_routed_experts=8,
    num_local_experts=8,
    num_experts_per_tok=2,
    linear_num_heads=4,
    linear_head_dim=16,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    mamba_num_heads=8,
    mamba_head_dim=8,
    ssm_state_size=16,
    n_groups=2,
    hc_lowrank=16,
    ple_embed_dim=64,
    ngram_vocab_size_base=1024,
    depth=1,
    num_heads=4,
    out_hidden_size=64,
)


# Layer kinds the default configs do not build.
LAYER_KINDS = {
    "deepseek_v4": dict(
        num_hidden_layers=4,
        num_hash_layers=1,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
    ),
    "glm5_next_text": dict(num_hidden_layers=4, layer_types=["linear_attention", "deepseek_sparse_attention"] * 2),
}


def tiny_config(config_cls):
    default = config_cls()

    def shrink(config):
        tiny = {key: value for key, value in TINY.items() if hasattr(config, key)}
        return tiny | LAYER_KINDS.get(config.model_type, {})

    sub_configs = {
        name: shrink(getattr(default, name)) for name in default.sub_configs if getattr(default, name, None) is not None
    }
    # Sparse attention backend the CPU can construct; models without sparse attention ignore it.
    return config_cls(**(sub_configs or shrink(default)), dsa_backend="tilelang")


@pytest.mark.parametrize(
    "config_cls", [config_cls for config_cls, _ in _CUSTOM_CAUSAL_LM_MODELS], ids=lambda cls: cls.model_type
)
def test_scratch_init_writes_every_weight(config_cls):
    """The from-scratch init (``initialize_from_scratch``) writes every parameter and buffer."""
    with torch.device("meta"):
        model = AutoModelForCausalLMPrimeRL.from_config(
            tiny_config(config_cls), attn_implementation="flash_attention_3"
        )
    model.to_empty(device="cpu")
    for tensor in [*model.parameters(), *model.buffers()]:
        tensor.data.fill_(float("nan") if tensor.is_floating_point() else -1)
    model.init_buffers_post_meta()
    model.initialize_weights()
    tensors = [*model.named_parameters(), *model.named_buffers()]
    unwritten = [
        name for name, tensor in tensors if (tensor.isnan() if tensor.is_floating_point() else tensor == -1).any()
    ]
    assert unwritten == []

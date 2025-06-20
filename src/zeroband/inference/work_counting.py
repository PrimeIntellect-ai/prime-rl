from functools import lru_cache

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from zeroband.utils.logger import get_logger

# Note: Only matmuls are counted


@lru_cache()
def _get_config(model_name_or_path: str) -> PretrainedConfig:
    return AutoConfig.from_pretrained(model_name_or_path)


def get_inference_input_output_flops_qwen3(
    config: Qwen3Config | Qwen3MoeConfig, num_input_tokens: int, num_output_tokens: int
) -> tuple[int, int]:
    """Get input and output flops for Qwen3 inference

    Args:
        config: Qwen3Config or Qwen3MoeConfig
        num_input_tokens: Number of input tokens
        num_output_tokens: Number of output tokens

    Returns:
        tuple[int, int]: Input and output flops
    """
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    head_dim = config.head_dim
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    num_hidden_layers = config.num_hidden_layers

    # Linears
    ## Attn
    q_flops = 2 * num_hidden_layers * hidden_size * num_attention_heads * head_dim
    k_flops = 2 * num_hidden_layers * hidden_size * num_key_value_heads * head_dim
    v_flops = 2 * num_hidden_layers * hidden_size * num_key_value_heads * head_dim
    o_flops = 2 * num_hidden_layers * hidden_size * num_attention_heads * head_dim

    ## MLP
    if isinstance(config, Qwen3MoeConfig):
        mlp_flops = 2 * num_hidden_layers * 3 * config.num_experts_per_tok * config.moe_intermediate_size * hidden_size
    else:
        mlp_flops = 2 * num_hidden_layers * 3 * intermediate_size * hidden_size
    ## LM Head
    lm_head_flops = 2 * vocab_size * hidden_size
    ## Total
    input_linear_flops = (q_flops + k_flops + v_flops + o_flops + mlp_flops + lm_head_flops) * num_input_tokens
    output_linear_flops = (q_flops + k_flops + v_flops + o_flops + mlp_flops + lm_head_flops) * num_output_tokens

    # SDPA
    ## 4lhqt from mm
    ## Each subsequent token sees 1 more ctx so the total is the sum of an arithmetic series
    input_ctx_sum = (num_input_tokens + 1) * num_input_tokens // 2
    output_ctx_sum = (num_output_tokens + num_input_tokens + num_input_tokens + 1) * num_output_tokens // 2
    input_sdpa = 4 * num_hidden_layers * head_dim * num_attention_heads * input_ctx_sum
    output_sdpa = 4 * num_hidden_layers * head_dim * num_attention_heads * output_ctx_sum

    return input_linear_flops + input_sdpa, output_linear_flops + output_sdpa


def get_inference_input_output_flops_deepseek_v3(
    config: DeepseekV3Config, num_input_tokens: int, num_output_tokens: int
) -> tuple[int, int]:
    """Get input and output flops for Deepseek V3 inference

    Args:
        config: DeepseekV3Config
        num_input_tokens: Number of input tokens
        num_output_tokens: Number of output tokens

    Returns:
        tuple[int, int]: Input and output flops
    """
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    head_dim = config.qk_head_dim  # Nope + Rope included
    num_attention_heads = config.num_attention_heads
    num_hidden_layers = config.num_hidden_layers

    # MoE
    num_dense_layers = config.first_k_dense_replace
    num_sparse_layers = config.num_hidden_layers - num_dense_layers
    shared_experts = config.n_shared_experts
    routed_experts = config.n_routed_experts
    experts_per_tok = config.num_experts_per_tok
    intermediate_size = config.intermediate_size
    moe_intermediate_size = config.moe_intermediate_size

    # Linears
    ## Attn
    q_flops = 2 * num_hidden_layers * (hidden_size * config.q_lora_rank + config.q_lora_rank * num_attention_heads * config.qk_head_dim)
    kv_flops = (
        2
        * num_hidden_layers
        * (
            hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim)
            + config.kv_lora_rank * num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim)
        )
    )
    o_flops = 2 * num_hidden_layers * (num_attention_heads * config.v_head_dim * hidden_size)
    ## MLP
    dense_mlp_flops = 2 * num_dense_layers * 3 * intermediate_size * hidden_size
    sparse_mlp_flops = num_sparse_layers * (
        2 * shared_experts * 3 * moe_intermediate_size * hidden_size  # Shared experts
        + 2 * experts_per_tok * 3 * moe_intermediate_size * hidden_size  # Routed experts
        + 2 * routed_experts * hidden_size  # Router
    )
    ## LM Head
    lm_head_flops = 2 * vocab_size * hidden_size
    ## Total
    input_linear_flops = (q_flops + kv_flops + o_flops + dense_mlp_flops + sparse_mlp_flops + lm_head_flops) * num_input_tokens
    output_linear_flops = (q_flops + kv_flops + o_flops + dense_mlp_flops + sparse_mlp_flops + lm_head_flops) * num_output_tokens

    # SDPA
    ## 4lhqt from mm
    ## Each subsequent token sees 1 more ctx so the total is the sum of an arithmetic series
    input_ctx_sum = (num_input_tokens + 1) * num_input_tokens // 2
    output_ctx_sum = (num_output_tokens + num_input_tokens + num_input_tokens + 1) * num_output_tokens // 2
    input_sdpa = 4 * num_hidden_layers * head_dim * num_attention_heads * input_ctx_sum
    output_sdpa = 4 * num_hidden_layers * head_dim * num_attention_heads * output_ctx_sum

    return input_linear_flops + input_sdpa, output_linear_flops + output_sdpa


@lru_cache()
def _get_num_params(model_name_or_path: str) -> int:
    # This import is needed because it uses Tensor.item() which will error if default device is meta
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: F401

    config = _get_config(model_name_or_path)
    default_device = torch.get_default_device()
    torch.set_default_device("meta")
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_device(default_device)
    if not config.tie_word_embeddings and hasattr(config, "hidden_size") and hasattr(config, "vocab_size"):
        offset = config.vocab_size * config.hidden_size
    else:
        offset = 0
    return sum(p.numel() for p in model.parameters()) - offset


def get_inference_input_output_flops(model_name_or_path: str, num_input_tokens: int, num_output_tokens: int) -> tuple[int, int]:
    config = _get_config(model_name_or_path)
    if isinstance(config, Qwen3Config) or isinstance(config, Qwen3MoeConfig):
        return get_inference_input_output_flops_qwen3(config, num_input_tokens, num_output_tokens)
    elif isinstance(config, DeepseekV3Config):
        return get_inference_input_output_flops_deepseek_v3(config, num_input_tokens, num_output_tokens)
    else:
        get_logger().warning(
            f"Model {type(config).__name__} flop calculation not specifically supported. Using fallback calculation based on parameter count."
        )
        num_params = _get_num_params(model_name_or_path)
        return 2 * num_params * num_input_tokens, 2 * num_params * num_output_tokens

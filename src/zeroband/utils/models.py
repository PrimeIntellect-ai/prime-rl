from typing import Literal, TypeAlias
from transformers import AutoTokenizer, LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, AutoConfig, AutoModelForCausalLM


ModelName: TypeAlias = Literal[
    # Dummy models
    "PrimeIntellect/llama-2m-fresh",
    "PrimeIntellect/llama-150m-fresh",
    "PrimeIntellect/llama-1b-fresh",
    # Llama 3
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Meta-Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3.2-3B-Instruct",
    # Qwen2.5
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/QwQ-32B",
    # DeepSeek R1 Qwen Distils
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # DeepSeek R1 Llama Distils
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    # Qwen3
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]

ModelType: TypeAlias = LlamaForCausalLM | Qwen2ForCausalLM | Qwen3ForCausalLM
AttnImpl: TypeAlias = Literal["flex_attention", "sdpa", "flash_attention_2"]


def get_model_and_tokenizer(model_name: ModelName, attn_impl: AttnImpl) -> tuple[ModelType, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config_model = AutoConfig.from_pretrained(model_name, attn_implementation=attn_impl)
    config_model.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, config=config_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer  # type: ignore

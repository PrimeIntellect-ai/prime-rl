from typing import Literal, TypeAlias

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM

ModelType: TypeAlias = LlamaForCausalLM | Qwen2ForCausalLM | Qwen3ForCausalLM
AttnImpl: TypeAlias = Literal["sdpa", "flash_attention_2"]


def get_model_and_tokenizer(model_name: str, attn_impl: AttnImpl) -> tuple[ModelType, AutoTokenizer]:
    config_model = AutoConfig.from_pretrained(model_name, attn_implementation=attn_impl)
    config_model.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, config=config_model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer  # type: ignore

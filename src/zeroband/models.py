from typing import Literal, TypeAlias
import torch
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

ModelName: TypeAlias = Literal["debugmodel", "150M", "1B", "Qwen32B", "Qwen1.5B", "Qwen7B"]

name_to_hf_model = {
    "debugmodel": "PrimeIntellect/llama-2m-fresh",
    "150M": "PrimeIntellect/llama-150m-fresh",
    "1B": "PrimeIntellect/llama-1b-fresh",
    "Qwen1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}

name_to_hf_tokenizer = {
    "debugmodel": "mistralai/Mistral-7B-v0.1",
    "150M": "mistralai/Mistral-7B-v0.1",
    "1B": "mistralai/Mistral-7B-v0.1",
    "Qwen1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}

name_to_class = {
    "debugmodel": (LlamaConfig, LlamaForCausalLM),
    "150M": (LlamaConfig, LlamaForCausalLM),
    "1B": (LlamaConfig, LlamaForCausalLM),
    "Qwen1.5B": (Qwen2Config, Qwen2ForCausalLM),
    "Qwen7B": (Qwen2Config, Qwen2ForCausalLM),
    "Qwen32B": (Qwen2Config, Qwen2ForCausalLM),
}


def get_model_and_tokenizer(model_name: ModelName) -> tuple[torch.nn.Module, AutoTokenizer]:
    config_class, model_class = name_to_class[model_name]
    tokenizer = AutoTokenizer.from_pretrained(name_to_hf_tokenizer[model_name])
    config_model = config_class.from_pretrained(name_to_hf_model[model_name], attn_implementation="flex_attention")
    model = model_class.from_pretrained(pretrained_model_name_or_path=name_to_hf_model[model_name], config=config_model)

    if isinstance(model, Qwen2ForCausalLM):
        tokenizer.pad_token_id = (
            tokenizer.eod_id
        )  # inspired from https://github.com/QwenLM/Qwen/blob/f7e3e7cb774b8667dde469250ac8e38ce850db1c/finetune.py#L328C1-L329C1
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer  # type: ignore

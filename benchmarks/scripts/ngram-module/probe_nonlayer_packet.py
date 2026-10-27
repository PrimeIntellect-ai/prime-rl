import sys

import torch

from prime_rl.trainer.models.layers.ngram import LAYOUT_BUFFERS
from prime_rl.trainer.models.qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextTextConfig,
)
from prime_rl.transports.weights.nccl import filter_state_dict_by_layers, get_max_layer_num, preprocess_layer_checkpoint
from prime_rl.utils.vlm import get_layer_prefix
from prime_rl.utils.weights import get_chunked_conversion_ops

golden = torch.load(sys.argv[1], weights_only=True)
for name, spec in golden["configs"].items():
    text = Qwen3_8FlashNextTextConfig(**spec["text_config"])
    config = Qwen3_8FlashNextConfig(text_config=text) if spec["composite"] else text
    config = type(config).from_dict(config.to_dict(), attn_implementation="flash_attention_2")
    with torch.device("meta"):
        model = Qwen3_8FlashNextForCausalLM(config)
    state_dict = model.state_dict()
    layer_prefix = get_layer_prefix(model.config)
    num_layers = get_max_layer_num(state_dict, layer_prefix)
    chunked_ops = get_chunked_conversion_ops(model, state_dict)
    _, non_layer = next(filter_state_dict_by_layers(state_dict, num_layers, layer_prefix))
    for key in chunked_ops:
        non_layer.pop(key, None)
    buffer_keys = [key for key in non_layer if key.rsplit(".", 1)[-1] in LAYOUT_BUFFERS]
    without_buffers = {key: value for key, value in non_layer.items() if key not in buffer_keys}
    converted_without = preprocess_layer_checkpoint(model, dict(without_buffers), -1)
    converted_with = preprocess_layer_checkpoint(model, dict(non_layer), -1)
    hf_buffer_keys = sorted(set(converted_with) - set(converted_without))
    other_keys_match = sorted(converted_without) == sorted(set(converted_with) - set(hf_buffer_keys))
    print(f"{name}: chunked {sorted(chunked_ops)}")
    print(f"  non-layer packet keys without tables: {sorted(converted_without)}")
    print(f"  table buffers in packet: {buffer_keys}")
    print(f"  buffers after conversion: {hf_buffer_keys}")
    print(f"  other keys identical with and without buffers: {other_keys_match}")

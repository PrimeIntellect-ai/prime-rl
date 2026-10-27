import sys

import torch

from prime_rl.trainer.models.qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextTextConfig,
)
from prime_rl.utils.vlm import get_language_model

golden = torch.load(sys.argv[1], weights_only=True)
buffers = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
for name, spec in golden["configs"].items():
    text = Qwen3_8FlashNextTextConfig(**spec["text_config"])
    config = Qwen3_8FlashNextConfig(text_config=text) if spec["composite"] else text
    config = type(config).from_dict(config.to_dict(), attn_implementation="flash_attention_2")
    model = Qwen3_8FlashNextForCausalLM(config)
    model.init_buffers_post_meta()
    ngram = get_language_model(model).ngram
    expected = {layout["decoder_index"]: layout for layout in golden["layouts"][name]}
    assert list(ngram.tables) == [str(d) for d in sorted(expected)], (name, list(ngram.tables))
    for decoder_index, layout in expected.items():
        table = ngram.tables[str(decoder_index)]
        assert ngram.table_indices[decoder_index] == layout["table_index"]
        assert table.weight.shape[0] == layout["num_rows"]
        assert all(torch.equal(getattr(table, b), layout[b]) for b in buffers)
    print(name, "tables", list(ngram.tables), "table_indices", ngram.table_indices)
print("model tables match main")

import sys

import torch

from prime_rl.trainer.models.qwen3_8_flash_next import Qwen3_8FlashNextTextConfig
from prime_rl.trainer.models.qwen3_8_flash_next.ngram_hasher import Qwen3_8FlashNextNGramHasher

golden = torch.load(sys.argv[1], weights_only=True)
buffers = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
checked = 0
for name, spec in golden["configs"].items():
    hasher = Qwen3_8FlashNextNGramHasher(Qwen3_8FlashNextTextConfig(**spec["text_config"]))
    for layout in golden["layouts"][name]:
        assert hasher.build_layout(layout["table_index"]) == tuple(layout[b].tolist() for b in buffers), (
            name,
            layout["table_index"],
        )
        assert hasher.num_table_rows(layout["table_index"]) == layout["num_rows"]
        checked += 1
real = golden["real_hash"]
fields = (
    "vocab_size",
    "eos_token_id",
    "ngram_vocab_size_base",
    "ngram_size",
    "heads_per_ngram",
    "make_ngram_vocab_size_divisible_by",
)
hasher = Qwen3_8FlashNextNGramHasher(Qwen3_8FlashNextTextConfig(**{f: real["params"][f] for f in fields}))
for layout in real["layouts"]:
    assert hasher.build_layout(layout["table_index"]) == tuple(layout[b].tolist() for b in buffers)
    assert hasher.num_table_rows(layout["table_index"]) == layout["num_rows"]
    print(
        "real table",
        layout["table_index"],
        "rows",
        hasher.num_table_rows(layout["table_index"]),
        "first sizes",
        layout["ngram_heads_vocab_sizes"][:3].tolist(),
    )
    checked += 1
print(f"all {checked} layouts match main")

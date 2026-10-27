import importlib.util
import os
import sys

import torch
import torch.distributed as dist

from prime_rl.trainer.models.layers.ngram import LAYOUT_BUFFERS
from prime_rl.trainer.models.qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextTextConfig,
)
from prime_rl.utils.cp import CPContext
from prime_rl.utils.vlm import get_language_model

spec = importlib.util.spec_from_file_location("ngram_before_split", sys.argv[2])
old_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_module)
old_forward = old_module.NGramHasher.forward

golden = torch.load(sys.argv[1], weights_only=True)
world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
if world_size > 1:
    dist.init_process_group("gloo")


def packed_row(vocab_size, eos_token_id, length, generator):
    tokens = torch.randint(0, vocab_size, (length,), generator=generator)
    eos_mask = torch.rand(length, generator=generator) < 0.05
    tokens[eos_mask] = eos_token_id
    cuts = torch.randperm(length - 1, generator=generator)[:5].sort().values + 1
    cu_seqlens = torch.cat((torch.tensor([0]), cuts, torch.tensor([length]))).to(torch.int32)
    return tokens[None], cu_seqlens


checks = 0
for name, config_spec in golden["configs"].items():
    text = Qwen3_8FlashNextTextConfig(**config_spec["text_config"])
    config = Qwen3_8FlashNextConfig(text_config=text) if config_spec["composite"] else text
    config = type(config).from_dict(config.to_dict(), attn_implementation="flash_attention_2")
    model = Qwen3_8FlashNextForCausalLM(config)
    model.init_buffers_post_meta()
    ngram = get_language_model(model).ngram
    hasher = ngram.hasher
    tables = [ngram.tables[str(layer_index)] for layer_index in ngram.hash_positions]
    multipliers, sizes, offsets = (
        torch.stack([getattr(table, buffer) for table in tables]) for buffer in LAYOUT_BUFFERS
    )
    for seed in range(5):
        generator = torch.Generator().manual_seed(seed)
        input_ids, cu_seqlens = packed_row(text.vocab_size, text.eos_token_id, 512, generator)
        hasher.cp_context = CPContext()
        full_new = hasher(input_ids, cu_seqlens, multipliers, sizes, offsets)
        full_old = old_forward(hasher, input_ids, cu_seqlens, multipliers, sizes, offsets)
        assert full_new.dtype == full_old.dtype and torch.equal(full_new, full_old), (name, seed, "full")
        checks += 1
        if world_size > 1:
            local_count = input_ids.shape[1] // world_size
            local_ids = input_ids[:, rank * local_count : (rank + 1) * local_count].contiguous()
            hasher.cp_context = CPContext(cp_group=dist.group.WORLD, cp_rank=rank, cp_world_size=world_size)
            cp_new = hasher(local_ids, cu_seqlens, multipliers, sizes, offsets)
            cp_old = old_forward(hasher, local_ids, cu_seqlens, multipliers, sizes, offsets)
            assert torch.equal(cp_new, cp_old), (name, seed, "cp vs old")
            assert torch.equal(cp_new, full_new[rank * local_count : (rank + 1) * local_count]), (
                name,
                seed,
                "cp vs full",
            )
            checks += 2
    print(f"rank {rank} {name}: shape {tuple(full_new.shape)} identical")
print(f"rank {rank}: {checks} checks passed, world size {world_size}")

# Local checks run during review

All on job 7048 (prime-nebius-puku-h200-gpu-020), from the live branch worktree.

## Hasher forward split (`c55988267`)

`check_hasher_split.py <golden_main.pt> <old ngram.py>` compares the split forward with the forward before it (`git show c55988267^:src/prime_rl/trainer/models/layers/ngram.py`) on random packed rows: 512 tokens, 6 documents, 5% EOS, 5 seeds per config. The golden configs are the ones captured on main in `golden/main_1810d5b9a.pt` (commit `f33d503d3`).

| Config               | max_ngram_size | Tables | Output shape  | Single process | CP=2 (gloo), both ranks         |
| -------------------- | -------------- | ------ | ------------- | -------------- | ------------------------------- |
| tiny_text            | 3              | 2      | (512, 2, 4)   | identical      | identical to old and to full row |
| tiny_composite_4gram | 4              | 2      | (512, 2, 6)   | identical      | identical to old and to full row |
| tiny_truncated       | 3              | 1      | (512, 1, 4)   | identical      | identical to old and to full row |

15 single-process checks and 45 per-rank CP checks passed. `check_model_tables.py` and `check_sympy_layouts.py` (all 7 layouts) also match main.

## NCCL non-layer packet (`4766d44a9`)

`probe_nonlayer_packet.py <golden_main.pt>` builds each golden config on the meta device and runs `preprocess_layer_checkpoint` on the NCCL non-layer packet with and without the table layout buffers. With the buffers the packet is detected as prime format, the buffers convert to their HF names (`...layers.{i}.ple.ple_embedding.{layer_multipliers,ngram_heads_vocab_sizes,ngram_heads_offsets}`), and every other key is identical to the conversion without them, for all three configs.

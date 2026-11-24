# Triton autotune GC A/B

Peak memory before and after collecting garbage after a Triton autotune with failed configs. Every run is SFT on random-init models truncated to 8 layers, with full activation checkpointing, no offloading, cold compile caches, and 8 steps on one 8-GPU node.

Setup:
- `stage_models.sh MODELS_DIR` downloads config and tokenizer files only. It lowers Qwen3.8's `ngram_vocab_size_base` to 1M, since the full n-gram table does not fit next to the optimizer on one node. The table sits outside the decoder layers.
- `python make_vlm_data.py DATA_DIR/vlm-synth` writes 64 synthetic image SFT samples for the Qwen3.8 runs.

Run each arm with `PYTHON=... MODELS_DIR=... DATA_DIR=... run.sh MODEL ARM SHA`, where MODEL is `glm5_next`, `glm5_next_8k` or `qwen3_8_flash_next`, ARM is `base` (the branch's merge base with main) or `fix` (the commit adding the listener with its `cache_hit` guard), and PYTHON is an interpreter with the dependencies locked at those commits. `run.sh` wipes the run's Triton and Inductor caches first, since a warm autotune cache benchmarks no configs. Per-step peak memory per rank lands in `OUT_DIR/runs/MODEL-ARM/peaks/rank<R>.jsonl`.

Summarize one model with `python summarize.py OUT_DIR/runs/MODEL-base OUT_DIR/runs/MODEL-fix`: peak allocated and reserved at step 1 and over steps 2 to 8, median step time over steps 4 to 8, and the largest relative loss and grad norm difference from the base arm.

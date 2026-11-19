# FLA compile A/B

Before/after measurement for compiling the GLM5-next, Qwen3.5 and Qwen3.8 flash next decoder layers through FLA custom ops. Every run uses random-init models truncated to 8 layers, full activation checkpointing, no offloading, cold compile caches and 8 steps on one 8-GPU node.

Setup:
- `stage_models.sh MODELS_DIR` downloads config and tokenizer files only. It lowers Qwen3.8's `ngram_vocab_size_base` to 1M, since the full n-gram table does not fit next to the optimizer on one node; the table sits outside the decoder layers.
- `python make_vlm_data.py DATA_DIR/vlm-synth` writes 64 synthetic image SFT samples for the Qwen VLM runs.

Run each arm with `PYTHON=... MODELS_DIR=... DATA_DIR=... run.sh MODEL ARM SHA`, where MODEL is one of `glm5_next`, `glm5_next_8k`, `qwen3_5`, `qwen3_8_flash_next`. Summarize a run with `python analyze.py OUT_DIR/runs/MODEL-ARM`: compiles and recompile reasons per frame, compile seconds, step-1 time, per-step loss, grad norm and step time. Per-step peak memory per rank is in `OUT_DIR/runs/MODEL-ARM/peaks/rank<R>.jsonl`.

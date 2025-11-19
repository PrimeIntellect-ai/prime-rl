uv run rl \
  --trainer @ examples/reverse_text/rl/train.toml \
  --orchestrator @ examples/reverse_text/rl/orch.toml \
  --model.name arcee-train/hf-converted-afmoe-nano-ctxext-v4 \
  --trainer-gpu-ids 1,2,3 \
  --trainer.model.ac \
  --trainer.model.compile \
  --trainer.model.attn "sdpa" \
  --trainer.model.impl "custom" \
  --trainer.model.load_using_meta \
  --no-trainer.model.moe_use_grouped_mm \
  --log.level debug \
  --wandb.project arcee_debug \
  --wandb.name run


# uv run rl \
#   --trainer @ examples/reverse_text/rl/train.toml \
#   --orchestrator @ examples/reverse_text/rl/orch.toml \
#   --model.name PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT \
#   --trainer-gpu-ids 1,2,3 \
#   --trainer.model.ac \
#   --trainer.model.compile \
#   --trainer.model.attn "sdpa" \
#   --log.level debug \
#   --wandb.project arcee_debug \
#   --wandb.name run


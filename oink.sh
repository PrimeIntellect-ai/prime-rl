uv run orchestrator @ orchestrator.toml \
  --train.sampling.max-completion-tokens 256 \
  --output_dir outputs/run_woof \
  --wandb.project gemma4-debug \
  --wandb.name original-impl \
  --num-train-workers 1 \
  --model.lora.name r8-1e-4-meow \
  --model.lora.rank 16 \
  --log.level debug

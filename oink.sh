uv run orchestrator @ examples/reverse_text/rl/orch.toml \
  --sampling.max-tokens 128 \
  --lora-name r8-1e-4-meow \
  --output_dir outputs \
  --wandb.project multi-tenant-debug \
  --wandb.name r8-1e-4-orch-meow \
  --num-train-workers 4 \
  --log.level debug


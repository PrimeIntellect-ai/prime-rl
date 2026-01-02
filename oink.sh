uv run orchestrator @ examples/reverse_text/rl/orch.toml \
    --model.name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --lora-name meow \
    --output-dir outputs/run_meow \
    --wandb.project multi-tenant-debug \
    --wandb.name meow \
    --log.level debug

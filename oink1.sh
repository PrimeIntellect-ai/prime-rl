uv run orchestrator @ examples/reverse_text/rl/orch.toml \
    --model.name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --lora-name woof \
    --output-dir outputs/run_woof \
    --wandb.project multi-tenant-debug \
    --wandb.name woof \
    --log.level debug

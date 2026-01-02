uv run orchestrator @ examples/reverse_text/rl/orch.toml \
    --model.name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --lora-name oink \
    --output-dir outputs/run_oink \
    --wandb.project multi-tenant-debug \
    --wandb.name oink \
    --log.level debug

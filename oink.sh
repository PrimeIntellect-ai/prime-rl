uv run orchestrator @ examples/reverse_text/rl/orch.toml \
    --output-dir outputs/run_meow \
    --optim.lr 1e-5 \
    --wandb.project multi-tenant-debug \
    --wandb.name r8-1e-4-meow \
    --log.level debug \
    --sampling.max-tokens 128 \
    --max-steps 20 \
    --max-async-level 2
    #--model.lora.name meow \
    #--model.lora.alpha 16 \


# uv run rl \
#   --trainer @ configs/reverse_text/rl/train.toml \
#   --orchestrator @ configs/reverse_text/rl/orch.toml \
#   --orchestrator.sampling.max-tokens 128 \
#   --trainer.model.experimental.lora.rank 8 \
#   --trainer.ckpt.weights.save-adapter-separately \
#   --trainer.weight_broadcast.adapter_only \
#   --orchestrator.lora-name r8-1e-4 \
#   --trainer_gpu_ids 1 \
#   --output_dir multi_outputs/trainer_3 \
#   --trainer.optim.lr 1e-4 \
#   --wandb.project multi-tenant-debug \
#   --wandb.name r8-1e-4 \
#   --log.level debug \
#   --ckpt

uv run orchestrator @ configs/reverse_text/rl/orch.toml \
  --sampling.max-tokens 128 \
  --lora-name r8-1e-4 \
  --output_dir meow_outs \
  --wandb.project multi-tenant-debug \
  --wandb.name debug \
  --log.level debug \

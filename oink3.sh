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


uv run orchestrator @ configs/reverse_text/rl/orch.toml \
  --sampling.max-tokens 128 \
  --lora-name r8-1e-4-oink \
  --output_dir outputs/run_oink \
  --wandb.project multi-tenant-debug \
  --wandb.name r8-1e-4-orch-oink \
  --log.level debug \

# export CUDA_VISIBLE_DEVICES=1
# uv run torchrun --nproc-per-node 1 src/prime_rl/trainer/rl/train.py \
#   @ configs/reverse_text/rl/train.toml \
#   --model.experimental.lora.rank 8 \
#   --model.seq_len 4096 \
#   --ckpt.weights.save-adapter-separately \
#   --weight_broadcast.adapter_only \
#   --output_dir multi_out_dev \
#   --optim.lr 1e-4 \
#   --wandb.project multi-tenant-debug \
#   --wandb.name r8-1e-4 \
#   --log.level debug \
#   --ckpt

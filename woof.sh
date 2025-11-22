uv run rl \
  --trainer @ configs/reverse_text/rl/train.toml \
  --orchestrator @ configs/reverse_text/rl/orch.toml \
  --orchestrator.sampling.max-tokens 128 \
  --trainer.model.experimental.lora.rank 8 \
  --trainer.ckpt.weights.save-adapter-separately \
  --trainer.weight_broadcast.adapter_only \
  --orchestrator.lora-name r8-1e-4 \
  --trainer_gpu_ids 1 \
  --output_dir outputs \
  --trainer.optim.lr 1e-4 \
  --wandb.project multi-tenant-debug \
  --wandb.name r8-1e-4 \
  --log.level debug \
  --ckpt.resume-step 20 \
  --max-steps 25



#uv run orchestrator @ configs/reverse_text/rl/orch.toml \
#  --sampling.max-tokens 128 \
#  --lora-name r8-1e-4 \
#  --output_dir meow_outs \
#  --wandb.project multi-tenant-debug \
#  --wandb.name debug \
#  --log.level debug \

export CUDA_VISIBLE_DEVICES=1
uv run torchrun --nproc-per-node 1 src/prime_rl/trainer/rl/train.py \
  @ configs/reverse_text/rl/train.toml \
  --model.experimental.lora.rank 8 \
  --model.seq_len 4096 \
  --ckpt.weights.save-adapter-separately \
  --weight_broadcast.adapter_only \
  --output_dir outputs \
  --optim.lr 1e-4 \
  --wandb.project multi-tenant-debug \
  --wandb.name r8-1e-4-trainer \
  --log.level debug \
  --max-steps 100 \
  --max-async-level 5 \
  --ckpt

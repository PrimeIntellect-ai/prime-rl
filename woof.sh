export CUDA_VISIBLE_DEVICES=4,5,6,7
uv run torchrun --nproc-per-node 4 src/prime_rl/trainer/rl/train.py \
  @ examples/reverse_text/rl/train.toml \
  --model.experimental.lora.rank 8 \
  --model.impl custom \
  --model.load_using_meta \
  --ckpt.weights.save-adapter-separately \
  --weight_broadcast.adapter_only \
  --output_dir outputs \
  --optim.lr 1e-4 \
  --wandb.project multi-tenant-debug \
  --wandb.name r8-1e-4-trainer \
  --log.level debug \
  --max-steps 100 \
  --max-async-level 5
  #--model.seq_len 4096 \


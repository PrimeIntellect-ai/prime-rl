export CUDA_VISIBLE_DEVICES=2,3
uv run torchrun --nproc-per-node 2 src/prime_rl/trainer/rl/train.py \
  @ examples/reverse_text/rl/train.toml \
  --output_dir outputs \
  --optim.lr 1e-4 \
  --wandb.project multi-tenant-debug \
  --wandb.name r8-1e-4-trainer \
  --log.level debug \
  --model.seq-len 65536 \
  --model.ac \
  --model.compile \
  --max-steps 100 \
  --max-async-level 5
  #--max-concurrent-runs 2 \
  #--model.lora.rank 8 \
  #--weight-broadcast.type nixl \


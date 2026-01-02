export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2,3

uv run torchrun --nproc-per-node 2 src/prime_rl/trainer/rl/train.py \
  @ examples/reverse_text/rl/train.toml \
  --model.name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --model.compile \
  --model.ac \
  --model.lora.rank 8 \
  --model.seq_len 16384 \
  --model.optimization-dtype bfloat16 \
  --max-concurrent-runs 2 \
  --output_dir outputs \
  --optim.lr 5e-5 \
  --wandb.project multi-tenant-debug \
  --wandb.name r8-1e-4-trainer \
  --log.level debug \
  --max-steps 100 \
  --max-async-level 5

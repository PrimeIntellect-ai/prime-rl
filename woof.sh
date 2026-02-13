export CUDA_VISIBLE_DEVICES=2,3
uv run torchrun --nproc-per-node 2 src/prime_rl/trainer/rl/train.py \
  @ configs/ci/integration/rl_multi_run/trainer.toml \
  --model.name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --max-concurrent-runs 2 \
  --model.lora.rank 8 \
  --model.impl custom \
  --output_dir outputs \
  --optim.lr 1e-5 \
  --wandb.project multi-tenant-debug \
  --wandb.name multi-run-trainer \
  --log.level debug \
  --model.seq-len 8192 \
  --model.ac \
  --model.attn flash_attention_4 \
  --model.compile \
  --bench --data.fake.batch-size 4 \
  --max-async-level 3

uv run torchrun --nproc_per_node=4 src/prime_rl/trainer/train.py @ configs/trainer/hendrycks_math/16b-a3b.toml \
    --bench \
    --data.fake.micro_batch_size 4 \
    --data.fake.batch_size 128 \
    --data.fake.seq_len 2048 \
    --model.ep_mode 1

#uv run rl \
#  --trainer @ configs/trainer/hendrycks_math/16b-a3b.toml \
#  --orchestrator @ configs/orchestrator/hendrycks_math/16b-a3b.toml \
#  --inference @ configs/inference/hendrycks_math/16b-a3b.toml \
#  --trainer-gpus 2 --inference-gpus 6

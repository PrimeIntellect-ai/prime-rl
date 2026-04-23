export PYTORCH_ALLOC_CONF=expandable_segments:True

uv run torchrun --nproc-per-node 4 \
    src/prime_rl/trainer/rl/train.py \
    --model.name google/gemma-4-26B-A4B-it --tokenizer.name google/gemma-4-26B-A4B-it --max-steps 2 \
    --wandb.project gemma4-debug --wandb.name trainer --max-steps 2 \
    --model.ac --model.compile \
    --model.fused-lm-head-token-chunk-size 1024 --model.attn sdpa \
    --max-concurrent-runs 2 --model.lora.rank 16 --model.seq-len 16384 --model.lora.target-modules '["o_proj","q_proj","k_proj","v_proj","gate_proj","up_proj","down_proj","experts"]' \
    --output_dir outputs --max-async-level 3 --dist_timeout_seconds 3600 --log.level debug | tee train.log

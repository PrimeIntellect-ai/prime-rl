# DeepSeek-V4.1-Flash

SFT and RL configs for [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). Only the text model is trained; the vision tower and the DSpark draft head in the checkpoint are dropped.

V4.1 is V4's architecture (shared-KV latent attention, mHC residual streams, sqrt-softplus MoE) with three additions the trainer implements in `prime_rl/trainer/models/deepseek_v41/`:

- **Shared long-range attention.** Only a few layers compress the token stream or run a Lightning Indexer; the layers after them reuse those entries and picks, and the later indexers only score inside the entry blocks a candidate layer ranked highest.
- **Delayed mHC gates.** Each sublayer's `pre` gate collapses the residual streams for the next sublayer.
- **Engram.** Two hashed n-gram tables (~384M rows x 256 each, ~98B parameters per table) add a gated lookup into the residual streams before layers 1 and 14. The trainer row-shards each table over the FSDP ranks and serves lookups with all-to-alls, outside FSDP. `model.freeze_engram_tables = true` freezes them (their projections still train).

Context parallelism and pipeline parallelism are not supported for this model yet.

## Convert the checkpoint

The trainer loads [`PrimeIntellect/DeepSeek-V4.1-Flash-bf16`](https://huggingface.co/PrimeIntellect/DeepSeek-V4.1-Flash-bf16), a bf16 copy of the published fp8 / fp4 checkpoint under the published names (1.34 TB). Its engram tables stay fp8 because vLLM serves them only in fp8; the trainer dequantizes them on load.

The trainer's automatic conversion to its own layout loads the whole checkpoint on one rank, which needs ~1.5 TB of host memory. Convert it once into a PrimeRL-format checkpoint instead, streaming layer by layer (one process per GPU):

```bash
SNAPSHOT=$(hf download PrimeIntellect/DeepSeek-V4.1-Flash-bf16)
OUT=/shared/models/DeepSeek-V4.1-Flash-bf16
for k in $(seq 0 7); do
  uv run python tools/convert_deepseek_v41_to_prime.py $SNAPSHOT $OUT --worker $k --num-workers 8 &
done; wait
uv run python tools/convert_deepseek_v41_to_prime.py $SNAPSHOT $OUT --finalize
```

The output directory also carries the tokenizer, which the engram n-gram hash is built from. Point `model.name` in [`sft/base.toml`](sft/base.toml) at it.

## SFT

Compose the base config with a data overlay:

```bash
uv run sft @ examples/advanced/deepseek-v4.1-flash/sft/base.toml @ examples/advanced/deepseek-v4.1-flash/sft/math-10k.toml
```

- EP 8 + FSDP, full activation checkpointing, SignSGD (no optimizer state), frozen engram tables, 8k-token sequences.
- For a fake-data dry run, append [`sft/fake.toml`](sft/fake.toml) instead of the data overlay.
- Chats are rendered with the `deepseek-v4.1` renderer from `renderers` (text conversations without tools).

Without context or pipeline parallelism every GPU holds a shard of the whole model state, so the run needs enough GPUs for ~550B non-engram parameters (fp32 masters and gradients) plus the frozen engram shards: on 80 GB GPUs, 24 is not enough. `num_train_nodes = 12` assumes 8-GPU nodes with more memory per GPU; size it to your hardware.

## RL

```bash
uv run rl @ examples/advanced/deepseek-v4.1-flash/rl/math.toml
```

- **Inference:** one node serves the bf16 checkpoint, which vLLM quantizes online to per-block FP8. It runs DP 8 with expert parallelism and keeps fp32 FP8 scales to match the trainer's Hopper FP8 kernels.
- **Trainer:** seven nodes run the SFT setup (EP 8 + FSDP, SignSGD, frozen engram tables).
- **Weight broadcast:** after every step the trainer sends its weights over NCCL. The engram tables can't be gathered whole, so the rank that holds each part of a table sends it in fp8 chunks, and vLLM writes the chunks straight into its host-memory copy of the table.
- **DeepGEMM:** V4.1's indexer needs DeepGEMM 2.8 or later on Hopper, because it reads its KV cache in 32-token blocks.

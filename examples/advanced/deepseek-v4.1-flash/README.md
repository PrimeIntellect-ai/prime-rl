# DeepSeek-V4.1-Flash

SFT configs for [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), tuned for 8-GPU H200 nodes. Only the text model is trained; the vision tower and the DSpark draft head in the checkpoint are dropped.

## Convert the checkpoint

The published checkpoint is fp8 / fp4 (~500 GB) and becomes ~1.4 TB in bf16, including two ~98B-parameter engram tables. Convert it once into a PrimeRL-format bf16 checkpoint, streaming layer by layer (one process per GPU, ~20 minutes on one node):

```bash
SNAPSHOT=$(hf download deepseek-ai/DeepSeek-V4.1-Flash)
OUT=/home/huggingface/prime/DeepSeek-V4.1-Flash-bf16
for k in $(seq 0 7); do
  uv run python tools/convert_deepseek_v41_to_prime.py $SNAPSHOT $OUT --worker $k --num-workers 8 &
done; wait
uv run python tools/convert_deepseek_v41_to_prime.py $SNAPSHOT $OUT --finalize
```

`base.toml` points `model.name` at that directory, which also carries the tokenizer the engram n-gram hash is built from.

## SFT

Compose the base config with a data overlay:

```bash
uv run sft @ examples/advanced/deepseek-v4.1-flash/sft/h200/base.toml @ examples/advanced/deepseek-v4.1-flash/sft/h200/math-10k.toml
```

- 8 nodes at 131k context: CP 8, EP 8, full activation checkpointing with activation offloading, optimizer state offloaded to CPU
- Context parallelism shards the queries; the single-head KV latents, compressed entries and index keys are all-gathered, so every rank's attention and indexer work stays local

For a fake-data dry run, append [`fake.toml`](sft/h200/fake.toml) instead.

### 12 nodes at 16k

```bash
uv run sft @ examples/advanced/deepseek-v4.1-flash/sft/h200/base.toml @ examples/advanced/deepseek-v4.1-flash/sft/h200/math-10k.toml @ examples/advanced/deepseek-v4.1-flash/sft/h200/16k-12-node.toml
```

On 12 nodes the sharded model state fits in GPU memory without CP or any offloading, and each GPU runs one 16k sequence per step (batch 96) with compiled transformer blocks. The experts are spread over all 96 GPUs (`ep = 96`) and dispatched with DeepEP: about 5.6 s per step (~280k tokens/s), 118 GiB peak memory.

On Hopper, prime-kernels' fused indexer top-k, mHC projection and MoE expert kernels are picked up when installed (`model.moe.compute.backend = "prime_kernels"` for the experts); otherwise the model runs its own implementations.

The engram tables are row-sharded across every data-parallel rank and served by all-to-all lookups, outside FSDP. Their gradients are dense fp32 shards, so plan for ~2 × 4 bytes × 197B / (number of GPUs) of engram state per GPU.

## Inference

vLLM serves the published fp8 / fp4 checkpoint on one 8xH200 node:

```bash
uv run inference @ examples/advanced/deepseek-v4.1-flash/infer/single-node.toml
```

On Hopper, V4.1 needs 64-token KV pages. The indexer caches of the 2:1 compressed layers then hold 32 rows per page, which DeepGEMM's paged indexer kernels only support on Blackwell; prime-rl's vLLM plugin routes that case to a torch implementation (`prime_rl/inference/vllm/deepseek_v41_hopper_indexer.py`).


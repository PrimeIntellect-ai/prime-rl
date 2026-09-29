# LMCache for multi-turn rollouts

This example selects `kv_cache_offload.type = "lmcache"` to connect inference to an
[LMCache multiprocess server](https://docs.lmcache.ai/getting_started/quickstart.html).
Prime-RL builds the connector configuration and enables prefix caching automatically.
SLURM launches one daemon per inference node; local runs manage the daemon separately.
The training command reuses the two-GPU Wordle example (one trainer, one inference GPU).

## Setup

Complete the [Prime-RL setup](../../../README.md#setup) first. Run the commands below
from the repository root. Install the optional `lmcache` extra (also included by
`uv sync --all-extras`). For a local run, start the daemon with the capacity, ports,
and chunk size matching [`inference.toml`](inference.toml):

```bash
uv run --extra lmcache lmcache server \
  --host 127.0.0.1 --port 5555 --http-host 127.0.0.1 --http-port 8080 \
  --l1-size-gb 2 --eviction-policy LRU --chunk-size 256
```

The server reserves 2 GiB of host memory. Match the LMCache native wheel to your
PyTorch/CUDA installation; see the [compatibility guide](https://docs.lmcache.ai/getting_started/compatibility.html).
Package installation alone does not validate a runtime combination.

## Check external cache reuse and policy salts

In another terminal, start a dedicated inference engine on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 \
  uv run --extra lmcache inference @ examples/extra/lmcache/inference.toml \
  --router None --server.host 127.0.0.1 \
  --vllm.model Qwen/Qwen3-0.6B --vllm.max-model-len 8192 \
  --vllm.gpu-memory-utilization 0.1 --vllm.enforce-eager
```

Once the engine is ready:

```bash
uv run --no-project python examples/extra/lmcache/check_cache.py \
  http://127.0.0.1:8000 Qwen/Qwen3-0.6B
```

The check tokenizes a prompt and sends it through `/inference/v1/generate`, the
route used for training rollouts, twice under each of two fresh salts. It requires
zero cached tokens on each salt's first request and a cache hit on its second request.
Before every request it resets **only vLLM's local cache**, waits for a successful
reset, and leaves LMCache intact. This distinguishes external reuse from local GPU
prefix hits. Missing usage details or an unsuccessful reset fails the check.

Run this only against an idle, single-engine development server: the check resets
that engine's prefix cache. Request timings are diagnostic, not a throughput benchmark.
The check verifies salt isolation with fixed weights; it does not validate weight
updates, overlapping policy versions, or training quality.

The smoke check passed with vLLM 0.29.0, LMCache 0.5.5, PyTorch 2.13.0+cu130,
Python 3.12, and one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU, using the
command's eager mode and Qwen3-0.6B in BF16. Both cold requests reported 0/2561
cached tokens; both warm requests reported 2560/2561. The Wordle command below
has been config-validated; a full RL run and throughput comparison remain unvalidated.

## Multi-turn RL

Stop the smoke-test inference engine and restart the LMCache server with an empty
cache before training. Then run:

```bash
uv run --extra lmcache rl @ configs/basic/wordle/rl.toml \
  --inference @ examples/extra/lmcache/inference.toml \
  --max-steps 5 --run.name wordle-lmcache
```

Prime-RL supplies `cache_salt` from each rollout group's starting policy version.
The external connector must preserve it during lookup, store, and retrieve; do not
replace it with a constant salt to increase cache hits. Restart the cache server
between independent training runs, since policy version numbers can repeat.

## SLURM

Use the same native config with either SLURM entrypoint. For example, generate an
inference job for two nodes without submitting it:

```bash
uv run --extra lmcache inference @ examples/extra/lmcache/inference.toml \
  --slurm --deployment.type multi_node --deployment.num-nodes 2 --dry-run
```

For RL, add `--slurm` to the Wordle command above (and `--dry-run` to inspect it
without submission). Both single-node and multi-node RL templates launch the
daemon automatically; do not pre-start a daemon on the same ports. SLURM installs
the extra through its existing `uv sync --all-extras` step. Each inference node's
engines share `cpu.num_bytes` of host cache. Pools are independent across nodes.

The launcher waits for readiness and monitors daemon exits; logs live under
`logs/attempt_<n>/lmcache/node_<rank>.log`. `port` and `http_port` must be unused on
each node. The daemon stops with the job, so same-weight cache reuse across engine
restarts requires leaving the separately managed local daemon running instead.

Disaggregated P/D automatically composes LMCache with NIXL via `MultiConnector`.
Connector composition and generated SLURM scripts are tested; a live multi-node
SLURM/P/D run remains unvalidated. Router replay with KV offload is rejected by
the existing RL validator. Disk offload is rejected for this backend. LoRA and
multi-replica reuse require separate runtime validation.

## Benchmark procedure

Use the same Wordle base config, model, GPU allocation, rollout concurrency, sampling,
and number of steps for three separate runs:

| Run | Inference configuration |
| --- | --- |
| GPU prefix cache | Omit this example's inference config; set `--inference.vllm.enable-prefix-caching` |
| Native CPU offload | Omit this example's inference config; set `--inference.kv-cache-offload.type native --inference.kv-cache-offload.cpu.num-bytes 2147483648` |
| LMCache MP | Use the command above with the 2 GiB LMCache server |

Keep GPU KV capacity equal and restart inference between runs. Compare completed
rollouts per second and end-to-end training-step time after startup/warmup, alongside
reward, preemptions, and prefilling work. Collect the standalone server's
[LMCache metrics](https://docs.lmcache.ai/mp/observability/metrics.html) separately:
Prime-RL's inference collector filters for `vllm:` metrics. Record package versions,
GPU type, context lengths, concurrency, and cache capacity with the results.

Wordle is a small functional workload and may not benefit from offloading. Evaluate
longer multi-turn contexts with GPU cache pressure before claiming a speedup.

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

## Multi-turn RL

Start a fresh LMCache daemon before a local training run:

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

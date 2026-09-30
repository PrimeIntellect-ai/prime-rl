# LMCache offload

[`inference.toml`](inference.toml) selects native LMCache MP offload with a 2 GiB
CPU pool per inference node. It requires the `lmcache` extra.

SLURM starts the daemon automatically. To use it with the two-GPU Wordle example:

```bash
uv run --extra lmcache rl @ configs/basic/wordle/rl.toml \
  --inference @ examples/extra/lmcache/inference.toml \
  --max-steps 5 --run.name wordle-lmcache --slurm
```

For a local run, omit `--slurm` and first start a daemon matching the config:

```bash
uv run --extra lmcache lmcache server \
  --host 127.0.0.1 --port 5555 --http-host 127.0.0.1 --http-port 8080 \
  --l1-size-gb 2 --eviction-policy LRU --chunk-size 256
```

Restart the local daemon between independent training runs, whose policy versions
can overlap. See [KV cache offload](../../../docs/inference.md#kv-cache-offload)
for configuration and deployment details.

# GLM-5.3 NVFP4 P/D on B300

From the repository root:

```bash
uv sync --all-extras --all-packages
uv run inference @ examples/advanced/glm-5.3/infer/pd-nvfp4-b300.toml --dry-run
uv run inference @ examples/advanced/glm-5.3/infer/pd-nvfp4-b300.toml
```

The dry run writes resolved prefill/decode configs and the Slurm script without
allocating nodes. Remove `--dry-run` to submit two exclusive eight-GPU nodes,
with no job time limit. The router listens on port 8000 on the first allocated
node; use model name `glm53`. Each role is an independent DPEP8 group, TP1.

| Setting | Prefill | Decode |
|---|---|---|
| All-to-all backend | FlashInfer NVLink one-sided | DeepEP low latency |
| Batched tokens | 32,768 | 320 |
| Maximum sequences per GPU | 256 | 320 |
| GPU memory utilization | 0.75 | 0.90 |
| Graphs | None | Full + piecewise, breakable |
| Mooncake store RAM per node | 2,560 GiB | 0 |
| HiSparse host RAM per GPU | Disabled | 300 GiB |

Both roles use FP8 KV, a 32,768-token context limit, index cache with frequency
4, natural expert routing, and no EPLB, DBO or MTP. Routing is sticky least
loaded with retries disabled. Each rank binds to its configured NUMA node and
UCX NIC. These mappings and the host-memory capacities target this B300 cluster;
adjust them for a different machine topology.

The model is the prequantized `RadixArk/GLM-5.3-NVFP4` checkpoint at revision
`11af4cba759e6559eda70358a5778bd1bddddd78`, using its existing local download.
To load from HF instead, override `--vllm.model RadixArk/GLM-5.3-NVFP4`; keep
the revision pin. This config does not quantize a BF16 trainer checkpoint online.

Use the pinned HiSparse/NVFP4 wheel with the standard Python frontend and native
`NixlConnector`. No experimental connector module or worker class is required.
Mooncake master/storage processes belong to the Slurm job, so cancellation also
releases their memory. Both roles connect to the same store pool with separate
role cache-key prefixes; shared capacity does not imply cross-role key reuse.
Each connector also uses a 512 MiB RDMA staging buffer.

This is inference-only: it does not start an eval, trainer, persistent supervisor,
or profiler. For RL, use these settings under `[inference]` and add the trainer,
orchestrator, weight-broadcast and root deployment settings. The RL launcher uses
the same role-resolution and Mooncake provisioning paths.

# GLM-5.2 + HiSparse on llm-d (P/D disaggregation, wideEP)

This example trains `GLM-5.2` with RL on agentic SWE tasks using P/D disaggregation, FP8 inference, a Mooncake distributed CPU KV store on the prefill side, and **HiSparse host-resident decode** — fronted by the upstream [**llm-d**](https://llm-d.ai) router. It is the `glm5_llmd` recipe adapted for GLM-5.2's DSA sparse attention.

## How it differs from `glm5_llmd`

GLM-5.2 uses DSA sparse MLA: decode only ever attends to the indexer's top-k (2048) tokens per step, so the full KV does not need to be GPU-resident. The decode instances run [HiSparse](https://github.com/vllm-project/vllm/pull/46326): the full MLA KV lives in a pinned host pool (`hisparse_config.host_pool_gib`, per DP rank) and decode is served from small per-request GPU hot buffers. This lifts the decode concurrency ceiling from single digits to whatever `max_num_seqs` admits (measured ~14-20x decode throughput at c=96 on 2-node P/D).

Decode instances are **wideEP** (DP×EP with TP1 ranks, 8 engines/node — the same topology as `glm5_llmd`); each DP rank owns its own host pool, so aggregate decode KV capacity scales with DP.

## Knobs to understand before launching

- **`host_pool_gib` (160/rank here)**: per-rank pinned host pool = each rank's total KV capacity. 8 ranks/node → **1.28 TiB pinned per decode node**; a startup check fails fast if it does not fit in free RAM. Size as `(usable node RAM − co-tenants) / ranks-per-node`.
- **`kv_cache_offload.roles = ["prefill"]`**: the Mooncake store is scoped to prefill nodes — decode nodes contribute no 1 TB segment (their RAM is spoken for by the host pools) and drop the store connector. Prefill prefix reuse across nodes/replicas is unaffected. If you want the store on decode too (it works — validated), budget RAM for both.
- **decode `max_num_seqs` (96)**: bounds the eagerly-allocated hot buffers (the vLLM default of 1024 OOMs at model load) **and is the admission control**. HiSparse removes the GPU-KV ceiling that used to act as accidental backpressure: an over-admitted decode instance accepts everything and starves everyone. Size it to sustained decode compute at your target context length, or rely on `max_inflight_rollouts`.
- **decode `gpu_memory_utilization` (0.9)**: safe under HiSparse — the GPU KV budget only holds the indexer cache.
- **`non_cached_tokens = 1`**: all prefills route to prefill nodes. Decode-local prefill concurrent with NIXL arrivals can hit a device-side fault in the mixed-batch path (open follow-up in the HiSparse PR); keep the shortcut disabled until that lands.
- **Renderer**: `glm-5.1` — GLM-5.2 shares the GLM-5 template surface, so no dedicated renderer is needed; verify template fidelity once before long runs.
- **`device_name`**: RDMA NICs for Mooncake — set by hand from `nvidia-smi topo -m` on your nodes.

## Validation status

The inference side of this recipe is validated piecewise: HiSparse decode at this exact wideEP topology (DP32+EP, TP1, `host_pool_gib=160`, `max_num_seqs=96`) ran an 8-node prod-mirror canary clean (1232/1232, zero drift); Mooncake-store + HiSparse round-trips are validated on-rig (store hits load directly into the pinned host pool, byte-identical decode); greedy parity and AIME24 A/B are within noise vs plain decode. The combined full-scale rehearsal (llm-d router + prefill store + HiSparse decode together) should be smoke-tested once on your cluster before long runs — watch for `register_buffer failed` in decode logs and lease-expiry frees in prefill logs.

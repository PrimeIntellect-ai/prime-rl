# SFT recompiles from rows padded to a multiple of cp

Since #3948, SFT pads each packed row only to a multiple of `model.cp`, so rows have variable lengths (2413 to 16384 tokens
on the run below). Under `torch.compile`, a rank recompiles its block whenever a row length breaks a guard of the compiled
graph, e.g. the routing sort's `8*args[0].size()[1] >= 32767` (`sort_stable`) or the FSDP view-stride guards. The
recompiling rank stalls for 6-15 s; its expert-parallel peers wait in the all-to-all and every rank then waits in the
inter-node FSDP collectives.

`data.pad_to_seq_len = true` pads every row to `seq_len * micro_batch_size` again (the pre-#3948 behaviour), so rows have
one shape.

## Reproduce

```bash
uv run sft @ benchmarks/scripts/sft_pad_to_seq_len/qwen3_235b_fp8.toml --output-dir outputs/pad-cp
uv run sft @ benchmarks/scripts/sft_pad_to_seq_len/qwen3_235b_fp8.toml --output-dir outputs/pad-seq-len --data.pad_to_seq_len true
```

Add `TORCH_LOGS = "recompiles"` under `[env_vars]` to see the guard failures in every rank's trainer log.

## Measured (8x8 H200, 2026-10-19, this branch, both runs at the same time on separate node sets)

| `data.pad_to_seq_len` | Steps 6-30 mean | Slow (>20 s) steps | MFU | Loss @ 30 | Recompiles (all ranks) |
|---|---|---|---|---|---|
| `false` (main) | 20.2 s | 15 of 25: ~25 s, the rest ~12 s | 21.5% | 0.2417 | 24 |
| `true` | 12.7 s | 0 of 25: steady 12.5-13.2 s | 36.7% | 0.2417 | 0 |

The slow steps are the same steps on every run, on different node sets, with the NCCL flight recorder off, and with the
public `deep_gemm` wheel. A profile of a slow step shows one rank in dynamo/inductor compile for 6.5 s mid-step while its 7
expert-parallel peers spend ~20 s in the all-to-all.

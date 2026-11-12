# nanochat gated-residual autoresearch: final configs

Single-node 8×H200 pretraining configs for nanochat (karpathy/nanochat @ ff46300) in prime-rl.
- Batch: 2^19 tokens (256 × 2048).
- Weight decay: 0.2·D_REF/D.
- Warmdown: 0.5.
- AdamW embedding/unembedding LRs scaled by (d/768)^-0.5.
- Steps: FLOPs-matched, steps = round(C / (FLOPs/token · 2^19)), where FLOPs/token includes each residual connection's parameters and extra FLOPs.

| config | what | val bpb (ours) |
|---|---|---|
| `d12-plain-1e18.toml` | plain baseline, proxy (1e18 FLOPs) | 0.8999 (3 seeds) |
| `d12-gated-s4r8-muon-1e18.toml` | **best variant**: gated residual, 4 streams, rank-8 gates trained with Muon | 0.8956 (3 seeds) |
| `d12-gated-default-1e18.toml` | default gated residual (4 streams, rank 64, AdamW gates) | 0.9006 |
| `d16-plain-3e18.toml` | plain, 3e18 FLOPs | 0.8497 (4 seeds) |
| `d16-gated-s4r8-muon-3e18.toml` | best variant, 3e18 FLOPs | 0.8452 (3 seeds) |
| `d12-plain-1.08Btok.toml` | plain d12 at the nanochat #420 horizon (8 × params) | 0.9068, CORE 0.1294 |

**Data and model configs.** Each TOML reads its model config from `/volume/models/cfg/<name>`; copy the matching folder from `models/` there. The volume also needs:
- `datasets/fineweb100b-packed/{train,val}` (pre-packed 2049-token rows)
- `models/tokenizer-rustbpe32k`
- `eval_bundle/` (nanochat's CORE bundle)

**Tools.**
- `tools/launch.py` computes params and FLOPs on the meta device, derives the hyperparameters, writes the TOML and submits it.
- `tools/test_variants.py` and `tools/test_core.py` are CPU unit tests: variant forward/backward and accounting, and the CORE port against nanochat's `core_eval`.

**Full #420 ladder.** `ladder/` holds every run of the full ladder for plain and the best gated variant:
- isoflop sweeps at 1e18, 3e18 and 6e18;
- the miniseries d10–d20.

See `ladder/README.md` for the configs and the result.

# Full nanochat #420 ladder: plain vs gated (s4 r8, Muon gates)

Every run of Karpathy's #420 scaling ladder (karpathy/nanochat discussion #420), on two architectures:
- `plain`: nanochat ff46300.
- `gated-s4r8-muon`: residual `{"type":"gated","streams":4,"low_rank":8,"read_scale":2.0}` with `connection_matrix_optimizer = "muon"`.

Each run is a single 8×H200 node with batch 2^19 tokens and seed 0 (init_seed 42).

**Recipe:**
- Weight decay: 0.2·D_REF/D, with D_REF = 10.5 × d12 scaling params.
- Warmdown: 0.5.
- AdamW embedding/unembedding LRs × (d_model/768)^−0.5.
- Eval: in-trainer val bpb on the first 10240 rows of `fineweb100b-packed/val`, plus full CORE at the end.

| set | depths | horizon |
|---|---|---|
| `iso1e18` | d8–d16 | steps = round(1e18 / (FLOPs/token · 2^19)) |
| `iso3e18` | d10–d20 | 3e18 FLOPs |
| `iso6e18` | d14–d22 | 6e18 FLOPs |
| `mini` | d10–d20 | tokens = 8 × all params (#420 v1 convention: wte + lm_head + matrices + scalars) |

**Contents:**
- `runs/`: one TOML per new run, as launched, with the volume replaced by a placeholder.
- `models/`: the model configs. Copy each folder to `/volume/models/cfg/<name>/`.
- `index.json`: run ids, steps, tokens, FLOPs, commit, and the image tag at launch.

Points reused from the earlier autoresearch loop (identical config and math) are not duplicated here:
- plain iso1e18 d8–d16;
- plain iso3e18 d14/d16/d18;
- gated iso1e18 d12;
- gated iso3e18 d16;
- plain mini d12.

`tools/` holds the planner (`ladder_plan.py`), the queue writer (`ladder_enqueue.py`) and the analysis and plotting script (`make_ladder.py`). They expect the experiment layout (`ledger.jsonl`, `scripts/`) next to them.

**Image.** Every TOML pins `image_tag = "v0.10.1.dev176"`, the base of this branch.
- On 2026-11-11 at 22:40 UTC the backend default image moved past main's torch 2.14 bump, and this branch's commits were then refused with "Source overlay cannot rebuild".
- Runs launched before that used the default image of the time, which had the same heavy deps.
- A control rerun under the pin reproduced the same bpb (0.899739 vs 0.899745).

**Result:**
- In FLOPs, gated beats plain at every budget, but the gain shrinks:

  | budget | Δ bpb | effective compute multiplier |
  |---|---:|---:|
  | 1e18 | +0.0042 | ~1.08× |
  | 3e18 | +0.0033 | ~1.10× |
  | 6e18 | +0.0029 | ~1.05× |

- In GPU-hours gated loses (0.62–0.69×), because it trains ~0.6× as many tokens/s.
- Plain reproduces #420's compute-optimal sizes (146/245/344M vs 137/240/330M) and D:N ≈ 8.

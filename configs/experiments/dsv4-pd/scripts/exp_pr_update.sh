#!/bin/bash
# Regenerate the stress and tuning tables of the exp PR body (PR_exp_body.md) and push it to PR #4574.
set -euo pipefail
E=/home/phi/exp/dsv4-pd; S=$E/stress; PY=/home/phi/lib/prime-rl-pd/.venv/bin/python; P=$E/PR_exp_body.md
cd $E/scripts
TUNE=$($PY analyze_tune.py \
  "$S/stress-pd-L1536.jsonl=P/D 1P:1D (2 nodes)" \
  "$S/tune-pd_mbt32k-L1536.jsonl=P/D 1P:1D, 32K steps (2 nodes)" \
  "$S/tune-pd2p2d-L3072.jsonl=P/D 2P:2D = 1:1 (4 nodes)" \
  "$S/tune-pd3p1d-L3072.jsonl=P/D 3P:1D (4 nodes)" \
  "$S/tune-pd3p1d_nvfp4-L3072.jsonl=P/D 3P:1D + NVFP4 (4 nodes)" \
  "$S/tune-pd2p2d_nvfp4-L3072.jsonl=P/D 2P:2D + NVFP4 (4 nodes)" \
  "$S/tune-pd3p2d-L3840.jsonl=P/D 3P:2D (5 nodes, stopped at 35 min)" \
  "$S/stress-pd-desc.jsonl=P/D 1P:1D (2 nodes)" \
  "$S/stress-pdmc-desc.jsonl=P/D 1P:1D + Mooncake (2 nodes)" \
  "$S/tune-pd2p2d-L4096.jsonl=P/D 2P:2D = 1:1 (4 nodes)" \
  "$S/tune-pd3p1d-L4096.jsonl=P/D 3P:1D (4 nodes)" \
  "$S/tune-pd1p3d_oneway-L4096.jsonl=P/D 1P:3D, token-in (4 nodes)" \
  "$S/tune-pd1p3d_bidir-L4096.jsonl=P/D 1P:3D + bidirectional KV, token-in (4 nodes)" \
  "$S/stress-agg-L1536.jsonl=reference: agg (2 nodes)" \
  "$S/tune-agg4-L3072.jsonl=reference: agg (4 nodes)" \
  "$S/stress-agg-desc.jsonl=reference: agg (2 nodes)")
STRESS=$($PY -c "from pr_results import tables; print(tables())")
$PY - "$P" "$STRESS" "$TUNE" <<'PYEOF'
import sys
p, stress, tune = sys.argv[1:]
s = open(p).read()
for name, table in (("stress", stress), ("tune", tune)):
    head, rest = s.split(f"<!-- {name}-results:start -->")
    _, tail = rest.split(f"<!-- {name}-results:end -->")
    s = f"{head}<!-- {name}-results:start -->\n{table}\n<!-- {name}-results:end -->{tail}"
open(p, "w").write(s)
PYEOF
cd /home/phi/lib/prime-rl-pd-exp && timeout 60 gh pr edit 4574 --body-file "$P" | tail -1

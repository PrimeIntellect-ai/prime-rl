# H200 FP8 cleanup SFT table, PR #3647

DeepSeek V4 Flash SFT, fake fixed-length data, H200. "before" is `main` at `88faa6dc2` (its `src` on `PYTHONPATH`,
same venv), "after" is this branch at `0b28c155c`, bf16 runs this branch without the FP8 overlay. Every run had
its own SLURM job with fresh compile caches. Median `time/forward_backward` per run over steady steps (5 to 15,
or 5 to 20 for the 8-node row), lower is better.

| setup | config | bf16 | fp8 before | fp8 after |
|---|---|---|---|---|
| 1 node, 6 layers, `cp=1`, full AC | `sft_1n_32k.toml` | 2.888 | 2.930 / 2.875 / 2.904 | 2.805 / 2.793 / 2.745 |
| 1 node, 6 layers, `cp=8`, SAC | `sft_1n_256k.toml` | 8.082 | 8.185 / 8.100 / 7.949 | 7.785 / 8.072 / 7.905 |
| 8 nodes, 43 layers, `cp=8`, full AC | `sft_8n.toml` | 9.712 | 9.853 | 9.504 |

- The first row's untraced runs are the refresh at `7d74e80d2` (code `c98735014`, same source as `0b28c155c`).
  `untraced_summary_1n_32k.csv` is a copy of its summary.
- The 8-node row first ran with selective AC and ran out of memory in the first backward
  (`results/failed_sac/`), so `sft_8n.toml` uses full AC.
- Figure: `step_accounting.png` from `step_accounting.csv`, the traced pair run serially on one node (job 3545,
  last 2 of 5 steps). The first traced pair ran on two different nodes (`step_accounting_diffnode_last3.csv`).
  Both pairs agree on compute (about -67 ms), and disagree on exposed communication (-145 ms vs +46 ms).

Commands, in order, are in `commands.sh`. Job ids are in `results/jobs.txt`.

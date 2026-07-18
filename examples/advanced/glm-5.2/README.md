# GLM-5.x — large-scale & disaggregated inference

Serving and RL examples across the GLM-5 family. The model version differs per file (kept
accurate to how each was run):

| config | model | what |
|---|---|---|
| `llmd_16node.toml`     | GLM-5.2-FP8 | 16-node inference (llm-d) — `uv run inference` |
| `disagg_inference.toml`| GLM-5-FP8   | disaggregated inference — `uv run inference` |
| `rl_pd_disagg.toml`    | GLM-5-FP8   | RL with prefill/decode-disaggregated inference — `uv run rl` |
| `rl_llmd.toml`         | GLM-5.1-FP8 | RL with llm-d inference — `uv run rl` |

# Configs

Throwaway configs for developing the framework itself. They are not guaranteed functional or up to
date. For walkthroughs, see [`examples/`](../examples); for frontier-model configs, see
[`recipes/`](../recipes). CI configs live in [`tests/integration/configs/`](../tests/integration/configs).

- **[`algo/`](algo)** — per-algorithm smokes.
- **`fake/`** — fake-data trainer and SFT smokes.
- **`eval/`** — `uv run eval` smokes against Prime Inference, one per shape: `single-turn`
  (gsm8k), `multi-turn` (terminal-bench-2 fix-git in sandboxes against a local vLLM deployment with
  the adaptive band), `resume`, `multi-env`.

```bash
uv run rl   @ configs/algo/<algo>.toml
uv run sft  @ configs/fake/sft.toml
uv run eval @ configs/eval/<shape>.toml
```

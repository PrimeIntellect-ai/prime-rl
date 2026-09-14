# Configs

Configurations for running prime-rl.

- **[`basic/`](basic)** — small, 2-GPU (1 trainer + 1 inference) configs for the core
  environments, sized to run on a single dev machine. Each mirrors the matching
  [`examples/basic/`](../examples/basic) tutorial and is smoke-tested for one step.
  Envs: `reverse-text`, `alphabet-sort`, `wiki-search`, `wordle`, `hendrycks-sanity`.
- **[`advanced/`](advanced)** — frontier-model training configs without launch walkthroughs:
  `minimax-m2.5` (swe), `nemotron-3-super` (swe), `deepseek-v4-flash` (sft + a standalone
  vLLM serving pre-flight). For the tutorialized equivalents, see
  [`examples/advanced/`](../examples/advanced).
- **`evals/`** — standalone eval configs, run with `uv run evals @ configs/evals/swe.toml`.
- **`ci/`** — integration and nightly configs used by CI.
- **`debug/`** — throwaway configs for developing the framework itself: `algo/`
  (per-algorithm smokes), `fake/` (fake-data trainer/SFT smokes), and `multi-env/`
  (two reverse-text train sources + one eval source, one env server per source). Not
  guaranteed functional or up to date.

```bash
uv run rl  @ configs/basic/<env>/rl.toml
uv run sft @ configs/basic/<env>/sft.toml   # where present
```

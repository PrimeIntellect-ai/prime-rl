# Configs

Configurations for running prime-rl.

- **[`basic/`](basic)** — small, 2-GPU (1 trainer + 1 inference) configs for the core
  environments, sized to run on a single dev machine. Each mirrors the matching
  [`examples/basic/`](../examples/basic) tutorial and is smoke-tested for one step.
  Envs: `reverse_text`, `alphabet_sort`, `wiki_search`, `wordle`, `hendrycks_sanity`.
- **`ci/`** — integration and nightly configs used by CI.
- **`debug/`** — throwaway configs for developing the framework itself: `algo/`
  (per-algorithm smokes), `fake/` (fake-data trainer/SFT smokes), and a few standalone
  entrypoint configs. Not guaranteed functional or up to date.

```bash
uv run rl  @ configs/basic/<env>/rl.toml
uv run sft @ configs/basic/<env>/sft.toml   # where present
```

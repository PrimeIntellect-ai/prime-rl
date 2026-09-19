# Configs

Configurations for running prime-rl.

- **[`basic/`](basic)** — complete, small-scale configs for the core environments. The matching
  [`examples/basic/`](../examples/basic) tutorials explain how to run them.
  Envs: `reverse-text`, `alphabet-sort`, `wiki-search`, `wordle`, `hendrycks-sanity`.
- **[`advanced/`](advanced)** — frontier-model configs, including reusable `envs/` and
  `models/` recipe layers. [`examples/advanced/`](../examples/advanced) explains composition.
- **`ci/`** — integration and nightly configs used by CI.
- **`debug/`** — throwaway configs for developing the framework itself: `algo/`
  (per-algorithm smokes), `fake/` (fake-data trainer/SFT smokes), `multi-env/`
  (two reverse-text train sources + one eval source, one env server per source), and
  `eval/` (`uv run eval` smokes against Prime Inference, one per shape: `single-turn`
  (gsm8k), `multi-turn` (terminal-bench-2 fix-git in sandboxes against a local vLLM deployment with the adaptive band), `resume`, `multi-env`).
  Not guaranteed functional or up to date.

```bash
uv run rl   @ configs/basic/<env>/rl.toml
uv run sft  @ configs/basic/<env>/sft.toml   # where present
uv run eval @ configs/debug/eval/<shape>.toml
```

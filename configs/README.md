# Configs

Configurations for running prime-rl.

- **[`advanced/`](advanced)** — frontier-model training configs without launch walkthroughs:
  `nemotron-3-super` (swe), `deepseek-v4-flash` (sft + a standalone
  vLLM serving pre-flight). For the tutorialized equivalents, see
  [`examples/advanced/`](../examples/advanced).
- **`ci/`** — integration and nightly configs used by CI.
- **`debug/`** — throwaway configs for developing the framework itself: `algo/`
  (per-algorithm smokes), `fake/` (fake-data trainer/SFT smokes), and
  `eval/` (`uv run eval` smokes against Prime Inference, one per shape: `single-turn`
  (gsm8k), `multi-turn` (terminal-bench-2 fix-git in sandboxes against a local vLLM deployment with the adaptive band), `resume`, `multi-env`).
  Not guaranteed functional or up to date.

```bash
uv run rl   @ configs/debug/algo/<algo>.toml
uv run sft  @ configs/debug/fake/sft.toml
uv run eval @ configs/debug/eval/<shape>.toml
```

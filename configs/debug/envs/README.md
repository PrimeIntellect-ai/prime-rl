# Envs — Debug Configs

Minimal end-to-end configs for bundled multi-agent envs, using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` as the policy.

| Config | Env | Notes |
|---|---|---|
| `agentic_judge.toml` | `agentic-judge` over `reverse-text-v1` | solver in a docker box, frozen `deepseek/deepseek-v4-flash` judge grades in the same box via `judge_policy.md` — a two-tool-call mirror of the deterministic check (needs docker + a Prime Inference key; run from the repo root) |

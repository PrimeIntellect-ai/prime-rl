# Replay — Debug Configs

Minimal end-to-end configs for the `replay-v1` taskset (`environments/replay_v1`), which turns
saved rollouts (`<run>/rollouts/step_*/train_rollouts.jsonl`) back into training tasks. Each config
mixes a fresh `reverse-text-v1` env with one replay env via ratios (ratios are all-or-none across
train envs), using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` as the policy and the `null`
harness with the subprocess runtime for both envs.

| Config | Mode | Buffer | Scoring | Notes |
|---|---|---|---|---|
| `offline_recheck.toml` | `recheck` | offline, a prior run's `rollouts` dir | `inner` (`reverse-text-v1`) rescores the revised answer | edit `buffer_dir` to a real prior run first |
| `online_judge.toml` | `judge` | `buffer_dir = "self"` + `online = true` — this run's own rollouts | self-contained: reward = verdict matches the source reward | forces whole-group dispatch; `max_steps_back` bounds recency |

`continue`/`recheck` need an `inner` taskset that reproduces the source run's taskset config;
`judge` forbids one. The replay env gets extra `max_input_tokens` headroom because its seeds carry
whole source conversations (recheck) or rendered transcripts (judge).

## Run the debug configs

```bash
# Offline recheck (edit buffer_dir in the TOML to a real prior run's rollouts dir first)
uv run rl @ configs/debug/replay/offline_recheck.toml --output-dir outputs/replay_recheck

# Online self-judging (no prior run needed — the buffer fills as the run trains)
uv run rl @ configs/debug/replay/online_judge.toml --output-dir outputs/replay_judge
```

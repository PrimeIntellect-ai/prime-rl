# Replay — Debug Configs

Minimal end-to-end configs for the `replay-v1` taskset (`environments/replay_v1`), which turns
saved rollouts (`<run>/rollouts/step_*/train_rollouts.jsonl`) back into training tasks. Each config
mixes a fresh `reverse-text-v1` env with one replay env via ratios (ratios are all-or-none across
train envs), using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` as the policy and the `null`
harness with the subprocess runtime for both envs.

| Config | Derivation | Buffer | Notes |
|---|---|---|---|
| `offline_recheck.toml` | `recheck` | offline, a prior run's `rollouts` dir | edit `buffer_dir` to a real prior run first |
| `online_recheck.toml` | `recheck` | `buffer_dir = "self"` — this run's own rollouts | forces whole-group dispatch; buffer fills as the run trains |

Every derivation is scored by its `inner` taskset, which must reproduce the source run's taskset
config. The replay env gets extra `max_input_tokens` headroom because its seeds carry whole source
conversations. `continue` (`anchor = "compaction" | "tool-call"`) has no debug config here:
reverse-text rollouts neither compact nor call tools — point one at an agentic run's buffer instead.

## Run the debug configs

```bash
# Offline recheck (edit buffer_dir in the TOML to a real prior run's rollouts dir first)
uv run rl @ configs/debug/replay/offline_recheck.toml --output-dir outputs/replay_recheck

# Online self-recheck (no prior run needed — the buffer fills as the run trains)
uv run rl @ configs/debug/replay/online_recheck.toml --output-dir outputs/replay_recheck_online
```

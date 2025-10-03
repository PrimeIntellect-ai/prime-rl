# Granite Wordle Ablation Plan

## Context
- Base model: `ibm-granite/granite-3.0-1b-a400m-instruct`.
- Task: Wordle environment (multi-turn) with SFT warm start + three RL ablations mirroring reverse-text study.
- Non-negotiables: preserve all logs/artifacts; run trainers with `clean=false`; checkpoint weights for every run and upload to HF under `rewardhacker00` namespace.

## Phase 0 – SFT Warmup (Completed)
- Run 200-step SFT with `--weights` flag.
- Artifacts: `outputs/granite_wordle_sft_200/weights/step_200`, logs in `outputs/logs/sft/granite_wordle_sft_200_runX.log` and trainer logs under `outputs/granite_wordle_sft_200/logs/`.
- Upload: `rewardhacker00/granite-wordle-sft-200` (pending once weights verified on disk).

## Phase 1 – RL Ablations (All via `nohup`, `clean=false`)
1. **Replay + Recompute**
   - Configs: `examples/wordle/rl/granite_train.toml`, `granite_orch.toml`, `granite_infer.toml`.
   - Output dir: `outputs/granite_wordle_rl_replay`.
   - Logs: `outputs/logs/wordle_rl_replay/{controller.log,run_supervisor.log,...}`.
   - Hugging Face upload: `rewardhacker00/granite-wordle-rl-100`.

2. **No Replay + Recompute**
   - Configs: `granite_train_noreplay.toml`, `granite_orch_noreplay.toml`.
   - Output dir: `outputs/granite_wordle_rl_noreplay`.
   - Hugging Face upload: `rewardhacker00/granite-wordle-rl-noreplay-100`.

3. **No Replay + No Recompute**
   - Configs: `granite_train_norecompute.toml`, `granite_orch_norecompute.toml`.
   - Output dir: `outputs/granite_wordle_rl_norecompute`.
   - Hugging Face upload: `rewardhacker00/granite-wordle-rl-norecompute-100`.

For each run:
- Launch `uv run rl` under `nohup` with explicit GPU binding (`CUDA_VISIBLE_DEVICES=0,1` split inference/trainer as needed).
- Ensure `ckpt.weights.interval=10` (or similar) so mid-run snapshots available (retain all).
- Stream logs to `outputs/logs/wordle_rl_*`; maintain orchestrator/trainer/inference/stdout files.

## Phase 2 – Analysis & Evaluation
- Parse rollout rewards into CSV/JSON under `outputs/analysis/wordle_*`.
- Generate combined reward plot (`docs/assets/granite_wordle_rl_rewards_comparison.png`) with legend for three configs.
- Produce Markdown tables (mean/peak training reward) plus evaluation summary (avg ± std) in `outputs/analysis/wordle_eval_summary_{quick,full}.md`.
- Run vf-eval quick (20×3) then full (1000×3) against each checkpoint via vLLM server (reuse `UV_CACHE_DIR` + `CUDA_VISIBLE_DEVICES=0`). Logs in `outputs/logs/eval/wordle_{model}_{quick,full}.txt`.

## Phase 3 – Documentation
- Extend `RUN_NOTES.md` with Wordle section (structure mirrors reverse-text entries) summarizing SFT warmup + three RL runs.
- Update `RUN_ABLATION_SUMMARY.md` with Wordle table and highlight key differences vs reverse-text.
- Mention future work (Wordle with larger Granite, router replay insights) at end.

## Phase 4 – Publishing & Cleanup
- Verify HF uploads (three RL + SFT) listed in README run notes.
- Leave all logs/artifacts intact. Optionally prune `.uv_cache` if space critical, otherwise keep.
- Stage git changes (plan, configs, notes, images) and commit with message `Add Granite Wordle ablation plan` once runs validated.

## Pending Checks
- Confirm weight files exist after current SFT rerun (watch `outputs/granite_wordle_sft_200/weights/step_200`).
- Ensure GPU availability before resuming RL.
- Blocked actions to revisit when machine back up: evaluations + uploads.

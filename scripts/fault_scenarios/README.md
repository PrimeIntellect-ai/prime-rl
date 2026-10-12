# Fault-injection scenarios

`run.py` runs one short multi-node RL job per fault (Qwen3-0.6B reverse-text, 12 steps, a checkpoint every 4 steps) with [`PRIME_FAULT`](../../docs/development.md#fault-injection) set, and reports how each run ends. It needs a 2-node allocation: one inference node with two TP=4 engines, and one trainer node.

## How to run

By hand, from the repo root:

```bash
sbatch -N 2 --ntasks-per-node=1 --gres=gpu:8 --exclusive -t 06:00:00 -o faults-%j.out \
    --wrap "uv run python scripts/fault_scenarios/run.py scripts/fault_scenarios/rl.toml outputs/faults"
```

- `--only engine_crash trainer_crash` runs a subset.
- Arguments after `--` go to every `rl` call, e.g. `-- --slurm.template-path my_template.sbatch.j2`.
- For each scenario, `rl --dry-run` renders the multi-node SLURM script into `outputs/faults/<scenario>/launcher/rl.sbatch`, and `run.py` runs it with `bash` inside the allocation. So the job settings (`slurm.partition`, `slurm.time`, ...) come from the outer `sbatch`, not from the TOML.
- Output: one line per scenario on stdout and in `outputs/faults/summary.txt`. Logs are in `outputs/faults/<scenario>/logs/` and `launcher/run_{first,resume}.log`.

Nightly, from cron on a SLURM login node (the GitHub nightly runner is a single node without SLURM, so this is not a workflow):

```cron
0 3 * * * cd /path/to/prime-rl && git pull -q && sbatch -N 2 --ntasks-per-node=1 --gres=gpu:8 --exclusive -t 06:00:00 -o $HOME/faults/\%j.out --wrap "uv run python scripts/fault_scenarios/run.py scripts/fault_scenarios/rl.toml $HOME/faults/$(date +\%F)"
```

## What is checked

For each scenario, after every run:
- **Outcome:** the exit code, or `hung` if the run did not end within the scenario's time budget (it is then killed).
- **Progress:** the last trainer step. A run that did not reach step 12 is resumed once with `--resume`, which is what a requeue does. Step faults are one-shot (`PRIME_FAULT_DIR` is the scenario's `faults/` dir), so the resumed run does not hit the fault again. The step the trainer resumed from is reported.
- **Leftovers:** processes (`vllm`, `prime_rl`, `torchrun`, `env-server`, `orchestrator`) and GPU contexts still alive on either node 30 s after the run ended.

The result is `finished in the first run`, `finished after a resume`, or `not finished`.

## Scenarios

| Scenario | Expected on `main` | Recovers in the first run with |
|---|---|---|
| `engine_crash@6`: one engine dies entering the NCCL weight receive for v6 | The engine exit ends the job (`wait -n` and `srun --kill-on-bad-exit`). The resume from step 4 finishes. | #3971 aborts the transfer, evicts the engine and resends v6 to the other engine; #4022 restarts the engine group instead of ending the job, and the engine rejoins. #3971 alone does not help: the launcher still ends the job. |
| `broadcast_hang@6`: one engine hangs entering the weight receive for v6 | The trainer master blocks in the NCCL broadcast. The other trainer ranks block in their next collective and hit the 900 s NCCL timeout, so the job fails about 15 min after the fault, before the orchestrator's `/update_weights` retries (2 × 720 s) run out. The resume from step 4 finishes. | #3971: the read timeout aborts the transfer, evicts the hung engine and resends to the other one. #3875's `step_timeout` ends the stalled job sooner and requeues it. |
| `trainer_crash@6`: one trainer rank raises at step 6 | torchrun exits 1 and the job fails. The resume from step 4 finishes. | #3875 requeues the job, which resumes from step 4. |
| `orchestrator_crash@6`: the orchestrator raises before shipping batch 6 | The orchestrator exits non-zero and the job fails. The resume from step 4 finishes. | #3875 requeues the job. |
| `ckpt_write_fail@8`: one trainer rank gets `ENOSPC` saving step 8 | The job fails. `step_8/` has an orchestrator checkpoint but no trainer checkpoint, the resume picks step 8, and the trainer fails to load it: **not finished**. | #3875 with #3906 (resume from the latest complete checkpoint): the requeue resumes from step 4. |
| `env_error@reverse-text-b`: every request to one of the two envs fails after 1 s | The failed groups are dropped and the other env fills the batches. Finishes. | Same outcome. With #4021 the breaker pauses `reverse-text-b` instead of dispatching to it. |
| `env_hang@reverse-text-b`: requests to one env never return | Its requests hold in-flight slots until none are left, and the run stalls: **hung**. | None. #4021 detects a server that stops answering health checks, but here the server is healthy. Its stall report names the env. A per-request deadline is missing. |

The envs run with the subprocess runtime, so no scenario creates Prime sandboxes. With a sandbox env (`env.agent.runtime.type = "prime"`), the crash scenarios also leave the run's sandboxes alive on `main`; #4025 deletes them when the run resumes.

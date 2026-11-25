# Inkling 2 Mini

RL on `thinkingmachines/Inkling-2-Mini` (text only): full fine-tuning from the released weights, without an SFT warm-up. Both RL configs use the custom trainer implementation with router replay, the `inkling-2` renderer at reasoning effort `low`, AdamW, NCCL weight broadcast, and W&B logging. `swe.toml` adds `cp = 8` context parallelism for its 131,072-token context.

Two things differ from the other advanced examples:

- **The inference server runs from a second Python environment.** Serving this model needs a vLLM build that has the looped Inkling 2 model. The vLLM release this repository pins (0.34.0) does not, and the `inkling2` plugin that serves Inkling 2 Small on it has no loop, so `uv run inference` cannot load the checkpoint. [Setup](#setup) builds the second environment, and each run section shows how the server starts from it.
- **Trainer and server load a BF16 copy of the checkpoint.** The release stores its routed experts packed, in NVFP4 and MXFP8. The server takes them in BF16 only, so both sides load the copy that `tools/convert_inkling2_loop_to_bf16.py` writes. That holds for a server that computes its experts in FP8 too: it rounds the BF16 weights itself as it loads them (see [Start a run with FP8 experts](#start-a-run-with-fp8-experts)).

| Config | Trains on | Evals on | Topology |
|---|---|---|---|
| [`reverse-text.toml`](reverse-text.toml) | `reverse-text` | 64 held-out `reverse-text` tasks (every 5 steps) | 1 train node + 1 infer node, the server started by hand |
| [`swe.toml`](swe.toml) | `r2e-gym` (`bash` harness) | 128 held-out `r2e-gym` tasks (every 25 steps), `swebench-verified` (every 50 steps) | 4 train + 13 infer nodes |
| [`swe-terminal-run-settings.toml`](swe-terminal-run-settings.toml) | as `swe.toml` | as `swe.toml`, every task 4 times | as `swe.toml`: an overlay, passed after it (`rl @ swe.toml @ swe-terminal-run-settings.toml`) |
| [`swe-fp8.toml`](swe-fp8.toml) | as `swe.toml` | as `swe.toml` | as `swe.toml`, with the routed experts in FP8 on the servers and in the trainer: an overlay, passed after it (`rl @ swe.toml @ swe-fp8.toml`) |
| [`infer.toml`](infer.toml) | — | — | the inference server of `reverse-text.toml`: 1 node, one replica per GPU |

Rollouts take the reasoning effort from the renderer (`[orchestrator.renderer] reasoning_effort`). Online evals go through chat completions and take it from `[orchestrator.eval.sampling]`, where the router accepts `low`, `medium` and `high`.

## Requirements

- 8-GPU H100 (80 GB) nodes on a shared filesystem: 2 nodes for `reverse-text.toml`, and 17 nodes under Slurm for `swe.toml`. This guide assumes the shared filesystem is mounted at `/shared` — adjust to your own path. The server uses vLLM's Hopper kernels, and every number in this guide is from 8×H100 nodes.
- A vLLM build that has the looped Inkling 2 model (`vllm.models.inkling2.nvidia.loop`), built for torch 2.14.0, the version this repository pins. The numbers in this guide are from a source build of vLLM at commit `b426b46e6` of pull request #69028, which has since been merged into vLLM and is not in release 0.34.0; vLLM main after that merge has not been run with this setup.
- **Sandboxes** (`swe.toml`) — the agents run in [Prime Intellect Sandboxes](https://docs.primeintellect.ai/sandboxes/overview). Install the `prime` CLI separately and log in:

```bash
uv tool install prime
prime login   # or: prime config set-api-key <your-key>
```

  `swe.toml`'s `slurm.pre_run_command` then cleans up the run's orphaned sandboxes by label before launch.

  To run on your own infrastructure instead, swap `env.agent.runtime` on each source for a runtime your environments support (e.g. a local Docker backend) and drop the `slurm.pre_run_command` cleanup line.
- Environment variables, exported in the shell you launch from — the launcher passes its environment to every component:

  - `HF_TOKEN` — for downloading the checkpoint, if your account needs one for it.
  - `WANDB_API_KEY` — both RL configs log to W&B (`[monitors.wandb]`).

`reverse-text.toml` needs no sandboxes: its agents run as local subprocesses in the env workers.

## Setup

Clone prime-rl onto the shared filesystem and install everything, including the environments — they are opt-in uv workspace members, so `--all-packages` is required:

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git /shared/prime-rl
cd /shared/prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
uv sync --all-extras --all-packages
```

### Convert the checkpoint to BF16

```bash
uv run hf download thinkingmachines/Inkling-2-Mini --local-dir /shared/models/Inkling-2-Mini
uv run python tools/convert_inkling2_loop_to_bf16.py /shared/models/Inkling-2-Mini /shared/models/Inkling-2-Mini-bf16 --verify
```

The tool dequantizes the routed experts and copies every other tensor unchanged. `--verify` re-quantizes each NVFP4 tensor and checks that it gives back the stored codes. All three configs point at `/shared/models/Inkling-2-Mini-bf16`. The trainer writes its own converted copy into a `prime/` folder inside that directory the first time it loads it, so the directory must be writable.

### Build the inference environment

Create a second virtual environment on the shared filesystem and install into it:

1. Your vLLM build with its own requirements.
2. This repository without its dependencies. That puts the `inference` entrypoint and prime-rl's two vLLM plugins next to that vLLM. On a vLLM that has its own Inkling 2, the `inkling2` plugin (the Inkling 2 Small port for the pinned release) logs a warning and does nothing, and the `prime_rl` plugin serves the checkpoint through `Inkling2LoopForRL`: routed experts in BF16, or in FP8 when the config asks for it, weight updates loaded in place, one replica per GPU. Run this step again after pulling a change to `pyproject.toml`: vLLM finds plugins through the installed metadata, not through the source tree.
3. The router and the NCCL version that `pyproject.toml` pins, so both ends of the NCCL weight broadcast load the same NCCL, and `tomli`, which reads the TOML configs (step 2 installs without dependencies and leaves it out).

```bash
uv venv --python 3.12 /shared/envs/vllm-inkling2
# 1. install your vLLM build into /shared/envs/vllm-inkling2, then:
# 2. prime-rl and the workspace packages its inference entrypoint imports
uv pip install --python /shared/envs/vllm-inkling2/bin/python --no-deps \
  -e . -e packages/prime-rl-configs -e deps/pydantic-config -e deps/verifiers -e deps/renderers
# 3. the router wheel and the NCCL pin from pyproject.toml, and the TOML reader
uv pip install --python /shared/envs/vllm-inkling2/bin/python --no-deps \
  "vllm-router @ https://github.com/PrimeIntellect-ai/router/releases/download/v0.2.8/vllm_router-0.2.8-cp38-abi3-manylinux_2_28_x86_64.whl" \
  "nvidia-nccl-cu13==2.32.3" tomli
```

Check that the entrypoint imports from it, and that vLLM, after loading its plugins, resolves the checkpoint's architecture to prime-rl's class (this needs no GPU):

```bash
/shared/envs/vllm-inkling2/bin/python -c "
import prime_rl.entrypoints.inference
from vllm.plugins import load_general_plugins
from vllm import ModelRegistry
load_general_plugins()
print(ModelRegistry.models['Inkling2LoopForConditionalGeneration'].load_model_cls())"
```

The last line must be the class `prime_rl.inference.vllm.models.inkling2_loop.model.Inkling2LoopForRL`.

Commands that run from this environment have no `uv run` in front: `uv run` would use the repository environment.

## Tweak before launching

- The lines marked `# FILL IN`: the model path in all three configs, `[slurm] partition` in `swe.toml`, and the host names in `reverse-text.toml` (the launch command below passes those on the command line instead).
- Use one spelling of the model path everywhere: the orchestrator checks that the server serves the model under the name the trainer uses.
- `--output-dir` defaults to `outputs/` next to the checkout; on a cluster, point it at the shared filesystem.
- Validate a config without starting anything by appending `--dry-run`. For `swe.toml` it writes `<run_dir>/launcher/rl.sbatch` plus the resolved per-process configs, and exits.
- Learning rate: `reverse-text.toml` uses `3e-6`. `1e-6` is stable and slower, and `1e-5` collapses after about 8 steps. `swe.toml` uses `1e-6`; with `swe-terminal-run-settings.toml` after it, the SWE run uses `3e-6`.
- `trainer.model.cp` must divide the model's 24 query heads and 8 KV heads, so it is 2, 4 or 8.

## Start the reverse-text run

`reverse-text.toml` runs 40 steps of 128 rollouts (16 tasks, 8 attempts each) on rows of up to 4,096 tokens. The trainer takes all 8 GPUs of its node: full fine-tuning peaks at 56 GiB per GPU for rows of up to 8k tokens. The rollouts come from a server on a second node with one replica per GPU. The config has no `[inference]` table, so the `rl` entrypoint starts the trainer, the orchestrator and the env worker pools, and expects the server to be running.

On the inference node, start the server from the second environment:

```bash
cd /shared/prime-rl
source /shared/envs/vllm-inkling2/bin/activate
inference @ examples/advanced/inkling2-mini/infer.toml
```

Its router listens on port 8000 and the engine behind it on port 8100. Wait until it answers:

```bash
curl --fail http://<inference-node>:8000/health
```

Then, on the trainer node:

```bash
cd /shared/prime-rl
uv run rl @ examples/advanced/inkling2-mini/reverse-text.toml \
  --weight-broadcast.host <trainer-node> \
  --orchestrator.model.client.base-url http://<inference-node>:8000/v1 \
  --orchestrator.model.client.admin-base-url http://<inference-node>:8100/v1 \
  --output-dir /shared/outputs/inkling2 \
  --run.name inkling2-reverse-text
```

`--weight-broadcast.host` is the trainer node's host name as the inference node reaches it: the server's GPUs join the trainer's NCCL group through it to receive weight updates. The launcher warns that no `[inference]` block is configured, which is expected here. Pass `--run.name`: the run directory is `<output_dir>/<run_name>`.

The server's admin routes (weight updates, pause, resume) accept every request unless `PRIME_RL_ADMIN_TOKEN` is set. To guard them, export the same value in both shells before you start the server and the run (see [Admin Routes](../../../docs/inference.md#admin-routes)).

What to expect:

- The held-out reward starts between 0.59 and 0.65 and reaches about 0.98. Over 150 steps (`--max-steps 150`) training stays stable and the reward reaches 0.995. The untuned model answers correctly but reasons at length, so part of its rollouts run out of the 4,000-token budget; the gain is the model learning to finish inside it.
- `mismatch_kl/all/mean` at step 1 is about 4e-5. Without router replay it is 1.4e-3.
- `max_inflight` equals the batch size, which keeps the trained tokens about 2 policy versions old instead of 4 to 5.

## Start the SWE run

`swe.toml` trains on R2E-Gym tasks with the `bash` harness in sandboxes, 256 episodes per step (32 tasks, 8 attempts each). The server context and the training rows are both 131,072 tokens, so an episode trains whole. The agent keeps its full history up to 16,384 tokens below that context: `swe.toml` leaves the compaction of the `bash` harness on, and from there the harness would replace the history by a summary. Context parallelism splits each row 8 ways along the sequence: every GPU holds 16,384 tokens of it, and each trainer node works on one row at a time. The reward is the task's own, without a length penalty.

The config uses the standard multi-node Slurm deployment: one job starts the trainer nodes, the orchestrator, the env worker pools, the router, and 8 inference servers on every inference node, one per GPU. The job starts each inference server with `uv run inference`, which is the repository environment. To start them from the second environment, give the job a copy of the launch templates in which that one command is replaced:

```bash
mkdir -p /shared/inkling2-mini/slurm
cp src/prime_rl/templates/multi_node_rl.sbatch.j2 src/prime_rl/templates/_launch_rank.sh.j2 /shared/inkling2-mini/slurm/
sed -i 's|uv run inference|/shared/envs/vllm-inkling2/bin/inference|' /shared/inkling2-mini/slurm/_launch_rank.sh.j2
```

A custom template resolves its includes next to itself before the bundled templates, so the copied `multi_node_rl.sbatch.j2` picks up the edited `_launch_rank.sh.j2`. Everything else in the job still runs from the repository environment. Submit with:

```bash
uv run rl @ examples/advanced/inkling2-mini/swe.toml \
  --slurm.template-path /shared/inkling2-mini/slurm/multi_node_rl.sbatch.j2 \
  --output-dir /shared/outputs/inkling2 \
  --run.name inkling2-swe
```

What to expect. The numbers are from a run with one eight-replica server per inference node and 4,096 completion tokens per turn (`swe.toml` allows 16,384), on a renderers checkout whose `inkling-2` renderer has `bridge_to_next_turn` (the renderers version this repository pins renders the whole conversation again at every turn, and none of these numbers was measured with it):

- The trainer processes about 24k tokens/s (6k per trainer node) at 40 GiB per GPU, and a step arrives about every 4 minutes.
- `mismatch_kl/all/mean` in the first steps is about 5e-4.
- The untuned model solves about 0.65 of the held-out R2E-Gym tasks and 0.70 of SWE-bench Verified.
- A reply that the per-turn token limit cuts inside a tool call ends the episode with an error: the chat template refuses arguments that are not valid JSON. That was at most 28 of about 21,600 recorded episodes at 4,096 completion tokens per turn, and none in three runs at 16,384.

Sizing:

- With the untuned model, 80-86% of the R2E-Gym episodes carry no training signal: all 8 attempts at their task end the same way and the group is dropped. A step of 256 trainable episodes therefore costs 1,300-1,800 episodes, which is why the inference side is 13 nodes.
- `max_inflight` is 8 episodes per inference GPU (64 per node). At this context length each engine logs a KV cache of 302,176 tokens (18 GiB); beyond that vLLM evicts conversations and prefills them again. One inference node sustains about 27 SWE episodes a minute at this setting. If you change `num_infer_replicas`, change `max_inflight` with it.
- Each trainer node adds about 6k tokens/s.

To check that a node serves the 131,072-token context before you commit 17 nodes, start one server by hand as for reverse-text, with the long-context settings:

```bash
inference @ examples/advanced/inkling2-mini/infer.toml \
  --weight-broadcast.type filesystem \
  --vllm.max-model-len 131072 \
  --vllm.max-num-batched-tokens 2048
```

## Start the SWE run with the terminal-run settings

`swe-terminal-run-settings.toml` is an overlay for `swe.toml`: pass it after `swe.toml`, as in the command below. Alone it is rejected, because it has no eval source. In the other order nothing fails, and the result is a mix of the two files: `swe.toml` then wins wherever both set a value, so the learning rate, the batch, the group size and the whole train source stay those of `swe.toml`, and only the clipping, `max_importance_ratio`, `max_staleness` and the two eval settings arrive. Its values are the training settings of one earlier run of this model. That run trained for 300 steps at the same 131,072-token context, with the same `bash` harness in sandboxes, but on terminal tasks. Its task set is not public (517 training tasks and 73 held-out tasks), so the tasks here stay those of `swe.toml`. It is the one multi-turn run of this model so far with a gain on held-out tasks that repeat evaluations confirmed.

The command is the one of "Start the SWE run" with the overlay added and a run name of its own, and it needs the template copy made there. If `/shared/inkling2-mini/slurm/multi_node_rl.sbatch.j2` is missing, the launch does not fail: the job is written from the bundled template, and its inference servers start with `uv run inference`, which cannot load this checkpoint.

```bash
uv run rl @ examples/advanced/inkling2-mini/swe.toml \
  @ examples/advanced/inkling2-mini/swe-terminal-run-settings.toml \
  --slurm.template-path /shared/inkling2-mini/slurm/multi_node_rl.sbatch.j2 \
  --output-dir /shared/outputs/inkling2 \
  --run.name inkling2-swe-terminal-run-settings
```

**This combination has not been run on R2E-Gym.** Only its config has been resolved (`--dry-run`). The numbers under "What to expect" and "Sizing" above are from runs with the values of `swe.toml` and do not describe it, and nothing here shows that the gain carries over to SWE tasks. Like those runs, that run had a renderers checkout whose `inkling-2` renderer has `bridge_to_next_turn`.

What the overlay sets, and why that run used it. Some values were copied from an earlier run of another model on the same tasks, and the run's notes give no reason of their own for them. They are marked "copied".

- **Learning rate `3e-6`** (`swe.toml`: `1e-6`). The trainer keeps FP32 master weights and computes with their BF16 rounding, so part of a small update does not reach the weights it computes with. The run's notes give this reason: after 50 steps of an R2E-Gym run at `1e-6`, about 13% of the update had reached the BF16 weights. That was not measured again, and it does not show that `3e-6` trains better on SWE tasks: the R2E-Gym runs behind this guide used `1e-6`, and a 166-step run on other SWE tasks at `3e-6`, with a length penalty, lost ground on SWE-bench Verified.
- **Gradient clipping at `0.5`** (default: `1.0`). The gradient norms of this model were 0.1 to 0.3, with rare spikes above 1: `0.5` clips the spikes and leaves the usual step alone.
- **`max_importance_ratio = 10`** (default: `10000`; copied). It caps the weight of a token that the trainer finds far more likely than the server did when it sampled it.
- **128 episodes per step in groups of 16** (`swe.toml`: 256 in groups of 8). That is 8 tasks per step, 16 attempts each. The group size is copied. The batch is the run's own choice: episodes on those tasks are long, and with the 512 of the earlier run, 300 steps would have taken about two days.
- **`max_staleness = 16`** (default: `8`). A group trains once its slowest episode has returned, which can be 1,800 s after it started, and at that run's pace 8 steps did not cover that. The run started at 14 and went to 16 at step 44.
- **The `difficulty_pool` task sampler** (copied). The run's notes say what it does, not why: a task whose latest group had mixed rewards is drawn 2 times as often as a task not tried yet, and a task whose latest group was all failed or all solved 0.05 times as often. A likely reason, which the notes do not give: a group whose attempts all end the same way trains nothing.
- **Two token limits on a training episode, no compaction**, and `scoring.skip_on_timeout`. Training drops an episode that runs into the 1,800 s rollout limit, while an episode that ends on a token limit is scored on what it left in the sandbox, and trains. The run's first launch had a 1,200 s limit and no limit on output tokens: 24% of its training episodes were dropped while still working. `max_output_tokens = 40000` ends such an episode first; in the base model's eval on those tasks, about 3% of the solved episodes had written more. `max_total_tokens = 106496` (copied) ends an episode once its conversation has that length. That is below the point where the `bash` harness would replace the history by a summary (16,384 tokens below the context), and `harness.compaction = "None"` (copied) turns that off for the train source. With `skip_on_timeout` (copied), an episode that timed out is not scored; training drops it anyway.
- **Evals: every task 4 times** (copied), **at temperature 1.0**, as the run evaluated its held-out tasks. Both eval sources of `swe.toml` get the 4 attempts: one eval is 512 episodes on the held-out R2E-Gym tasks and 2,000 on SWE-bench Verified.

The overlay is not that run's whole config. It leaves out what belongs to those tasks or to that cluster, among them: the task lists, the limit of 64 turns (the 100 of `swe.toml` stay), the 3,600 s rollout limit of the evals and the time limits of the other stages, the retry rules for sandbox and provider errors, a checkpoint every 10 steps, a fixed number of episodes in flight, and the node counts (6 trainer nodes, 9 to 10 inference nodes). It also leaves the eval sources of `swe.toml` as they are, so an eval episode has no limit on total tokens and keeps the harness's compaction, while a training episode has the limit and no compaction. That run's evals had the limit and no compaction, like its training episodes.

What the run showed. After the run, its starting weights and its step-300 checkpoint were evaluated again on the 73 held-out tasks, 4 attempts per task: twelve evaluations, three per model in each of two sets that differ in server load. With 64 episodes in flight per inference node, the starting weights solved 0.209 of the attempts and the checkpoint 0.282 (183 and 247 of 876). With 24 per node, it was 0.224 and 0.285 (196 and 250 of 876). That is a gain of 7.3 points in the first set and 6.2 in the second. The 95% intervals on these 73 tasks are +4.7 to +9.9 and +3.5 to +8.9; with the tasks as the unit, both sets together, +2.1 to +11.6. It is uneven: 8 of the 73 tasks carry 81% of the net gain. It is one run with one seed. The evals made during the run are single, noisy readings, and none of them is the result.

Before you launch it:

- The overlay keeps the Slurm job name and the sandbox label of `swe.toml`, and every launch first deletes the sandboxes with that label. So do not start one of the two while the other is running.
- A later file replaces a list of sources as a whole, so the overlay spells out the train source of `swe.toml` again, with the values that file has today. A change you make to the train source in `swe.toml` (its runtime, its label, its task selection, its turn limit) is undone by the overlay, for the train source only and without a message: make it in the overlay as well. The eval sources are not in the overlay and keep your change.
- How many episodes a step needs will change during a run, because the sampler moves towards tasks with mixed outcomes. For this combination that was not measured.
- To resume, pass both files again. The command under "Checkpoint and resume" names `swe.toml` alone and the run name `inkling2-swe`.
- In this checkout the sampler is too slow for this task list. Each draw computes the key of every task in its list again. On the 4,394 training tasks of R2E-Gym a draw took 8 to 10 s of CPU time in tests on one core, and the orchestrator does nothing else during a draw. It draws once for every group of 16 episodes it starts, so by these figures it can start about 100 to 120 episodes a minute, where 13 inference nodes take about 350 at the rate given under "Sizing". No run measured that. The run these settings are from had a sampler that computes the keys once. Launch this combination only on a checkout whose `DifficultyPoolSampler` (`src/prime_rl/orchestrator/curriculum/samplers/pool.py`) does the same.

## Start a run with FP8 experts

The routed experts can be computed in FP8 instead of BF16, on the servers and in the trainer together. For the SWE run that is an overlay, and for the reverse-text pair two settings on the command lines (at the end of this section).

`swe-fp8.toml` is an overlay for `swe.toml`: pass it after `swe.toml`, as in the command below. Alone it is rejected. It sets two values, and the rest of the resolved config stays that of `swe.toml`:

- **`[inference.vllm] quantization = "fp8_per_block"`.** Every server holds its routed experts in vLLM's online `fp8_per_block` format: 8-bit E4M3 weights with one FP32 scale per block of 128 x 128, and inputs of both expert projections rounded to E4M3 with one scale per token per 128 channels. The server rounds the weights itself as it loads them. For this model the setting does not reach vLLM's own quantization: `Inkling2LoopForRL` takes it.
- **`[trainer.model.moe.compute] type = "deepgemm_fp8"`.** The trainer computes its routed experts in FP8, in the forward and in the backward pass, and its forward pass rounds weights and inputs the same way as the servers. It keeps its FP32 master weights, and it rounds again at every call.

Everything but the routed experts is computed as without the overlay, on both sides.

```bash
uv run rl @ examples/advanced/inkling2-mini/swe.toml \
  @ examples/advanced/inkling2-mini/swe-fp8.toml \
  --slurm.template-path /shared/inkling2-mini/slurm/multi_node_rl.sbatch.j2 \
  --output-dir /shared/outputs/inkling2 \
  --run.name inkling2-swe-fp8
```

The command needs the template copy made under "Start the SWE run". The overlay can also follow `swe-terminal-run-settings.toml` (`rl @ swe.toml @ swe-terminal-run-settings.toml @ swe-fp8.toml`): the two overlays set different values, so their order between them does not matter.

**This combination has not been run on R2E-Gym.** The numbers under "What to expect" and "Sizing" above are from runs with BF16 experts and do not describe it. What is known about FP8 experts for this model is in the three lists below. Every number in them is for Inkling 2 Mini on H100 GPUs. All but one were measured before this code was put into its files of this repository, on a trainer whose model files are not this repository's line for line (the files that compute the FP8 expert products are the same). The one is the 20-step table at the end of the first point of the last list: it was made with this code in its files of this repository.

What must match between the servers and the trainer:

- **Both settings, together.** Nothing checks it. A run with FP8 servers and a BF16 trainer starts and trains, on tokens whose probabilities its trainer computes in another format: on the same weights, with the routing replayed, the mismatch KL of that pair was 0.0014, against 0.00087 with FP8 on both sides (parity runs: a server samples about 4,500 tokens on six general prompts, and the trainer model scores them on the same weights). In the reverse-text test below, three such runs gave 0.925, 0.889 and 0.921.
- **The BF16 copy of the checkpoint, on both sides.** The servers round what they load: the BF16 copy at start-up, and every weight update, which the trainer sends in BF16 as before. There is no FP8 model folder, and the packed release is still rejected.
- **All routed experts, and nothing else.** Leave `apply_to` of the trainer's table at its default `"all"`: the server has no choice of layers. Leave `[trainer.model.quantization]` unset: it would put the trainer's dense layers into FP8, and the server's stay in BF16.
- **H100 nodes on both sides.** On SM100 GPUs the trainer rounds its FP8 scales up to powers of two, and the server never does, so by the code the two sides would round the same weights to different bytes there. Nothing was run on other GPUs than H100.
- Everything else is as with BF16 experts: router replay on, one replica per GPU, no LoRA, and weight updates over NCCL or from files (the two transports this model has; both were run with FP8 experts). No `dtype`, no `use_deep_gemm` and no environment variable goes with the two settings. The trainer's DeepGEMM comes with the vLLM release that `uv sync --all-extras` installs, and the servers' FP8 kernels are those of their own vLLM. `quantization = "mxfp8"` is refused by this model.

What changes:

- **Server memory and KV cache.** One server GPU holds 25.8 GiB of weights with FP8 experts and 48.6 GiB with BF16 experts (server start-up logs and a single-GPU test of the experts module). At the same memory fraction and the 131,072-token context a server GPU commits 41.0 GiB of KV cache (about 688,000 tokens) with FP8 experts and 18.0 GiB (about 302,000 tokens) with BF16 experts (server logs of the two long runs, after warm-up).
- **`max_inflight` stays at 8 episodes per inference GPU.** The overlay does not raise it for the larger KV cache. What was measured is one model at that load: a model trained with BF16 experts, evaluated on 292 episodes with 64 in flight per inference node, lost 5 episodes to its one-hour limit when served with FP8 experts (one eval) and 42 to 48 when served with BF16 experts (three evals), with the fullest KV cache of any engine at 50% against more than 99%.
- **Time of the experts.** One layer's experts take 0.31 / 0.50 / 0.56 / 1.40 ms per call at 8 / 64 / 512 / 4,096 tokens with FP8, and 0.39 / 0.76 / 0.81 / 1.92 ms with BF16. A call with a single token is slower with FP8: 0.28 against 0.17 ms (single-GPU test, median of 30 calls).
- **Weight updates.** A server pauses about 3 s for an update instead of about 2 s, because it rounds the experts again (reverse-text runs).
- **Trainer.** Step time and peak memory were the same with FP8 and with BF16 experts in a trainer-only run on generated data, with rows of 131,072 tokens and `cp = 8`: 2 min 10 s per step and 56.8 GiB per GPU in both. That setup is not the one of the trainer figures above.

What the measurements show, and what they do not:

- **The servers and the trainer round the experts to the same bytes.** The serving module's rounding was compared with the trainer's own on one H100, for five experts of one layer (three with trained and two with the released weights; the model has 128 experts in each of 24 layers): none of their 39.8 million codes and 2,430 scales differs. The two sides multiply with different kernels: on identical inputs, 99.89% of one layer's outputs are bitwise equal, with a relative difference of 0.011%. That comparison, and one of the server's forward with exact arithmetic, were made at one batch size, 384 tokens; vLLM's kernel picks its tile sizes by the number of tokens, and the other sizes were only timed. For the whole model, with the routing replayed, the mismatch KL on the same weights in parity runs is 0.00087 on the released weights and 0.00083 to 0.00095 on three checkpoints of an FP8 run (0.0013 on prompts cut from task episodes, on the last of them); with BF16 on both sides it is 0.00008. The project's rule for a pull request that adds a model is a mismatch KL below 0.015 at each of 20 steps on the `math` environment (`docs/development.md`). With FP8 experts on both sides that table was made twice: every entry of both tables is between 0.00044 and 0.00057.
- **An FP8-served run trains.** Reverse-text RL from the untuned model, 40 steps at learning rate `3e-6` with batch 128 and router replay, was judged by a measure and a gate fixed in advance. The measure is the mean held-out reward over the evals of steps 5 to 40. The gate: the FP8 mean at most 0.03 below the BF16 mean, and no 20-step mean of an FP8 run's mismatch KL at 0.015 or above. Three runs with FP8 on both sides gave 0.948, 0.926 and 0.941, and the two BF16 runs made with them 0.929 and 0.922 (four earlier BF16 runs of this task at the same learning rate and batch: 0.923 to 0.941). Over the two fixed windows of 20 steps, the largest mean of the mismatch KL was 0.0060 with FP8 and 0.0053 with BF16. The gate passed. The test is coarse: it shows a setting that slows or destabilises learning, not small differences, and the three runs with FP8 servers and a BF16 trainer pass the same gate. Single steps reached 0.015 or more in both formats while the policy moved fast, in steps 8 to 10 (0.0170 at the most with FP8, 0.0167 in the two BF16 runs made with them), so the per-step rule of the `math` environment cannot be read off this test.
- **Not shown: that an FP8 run reaches the final result of a BF16 run.** There is one long run with FP8 on both sides: 300 steps on the terminal tasks of the section above, with the config that run was resumed with at step 100 and the two settings of this overlay. Evaluated like the second set there, each run's models served in its own format, its final model is 2.4 points above the untuned model on the held-out tasks, which is not separated from zero (95% interval -0.5 to +5.3). The final model of the BF16 run is 6.2 points above (3.5 to 8.9). That is one run per format.

Before you launch it:

- The overlay keeps the Slurm job name and the sandbox label of `swe.toml`, and every launch first deletes the sandboxes with that label. So do not start it while another run from `swe.toml` is going.
- Check both sides in the logs, since nothing else does. In the trainer log, look for `Configured 24/24 MoE layers with compute=DeepGemmFP8ExpertCompute`. A server has no line of its own that names the format of its experts, in its log or in its health output, and vLLM's engine line reads `quantization=None` in both formats, because the setting is taken off vLLM's arguments. Look at vLLM's start-up line `Model loading took ... GiB memory`: it reads about half of what it reads with BF16 experts (the figures above).
- To resume, pass the same files again. The command under "Checkpoint and resume" names `swe.toml` alone and the run name `inkling2-swe`.

The reverse-text pair takes the two settings on its command lines, not from the overlay. `reverse-text.toml` has no `[inference]` table, because its server is started by hand. The overlay would give it one, and nothing fails when the config is read: by the launcher's code, `rl` then starts an inference server of its own on the trainer node. So start the server with its setting, from the second environment:

```bash
inference @ examples/advanced/inkling2-mini/infer.toml --vllm.quantization fp8_per_block
```

and add the trainer's setting to the command of "Start the reverse-text run":

```bash
uv run rl @ examples/advanced/inkling2-mini/reverse-text.toml \
  --trainer.model.moe.compute.type deepgemm_fp8 \
  --weight-broadcast.host <trainer-node> \
  --orchestrator.model.client.base-url http://<inference-node>:8000/v1 \
  --orchestrator.model.client.admin-base-url http://<inference-node>:8100/v1 \
  --output-dir /shared/outputs/inkling2 \
  --run.name inkling2-reverse-text-fp8
```

The 4e-5 given there for `mismatch_kl/all/mean` at step 1 is for BF16 experts. On the same weights the FP8 pair measured 0.00087 in a parity run.

## Monitor with the dashboard

Start the local run dashboard on the head node — it only reads the run directories, so it is safe to point at a live run while the job is training:

```bash
uv run dashboard /shared/outputs/inkling2   # serves http://localhost:7788
```

If the head node is remote, forward the port from your laptop and open `http://localhost:7788` in a browser:

```bash
ssh -L 7788:localhost:7788 <head-node>
```

Pick the run and you get five views:

- **Metrics** — the W&B-style overview, read from the run's `metrics.jsonl`. Watch `reward/{all,env}/mean` trend upward over steps, `mismatch_kl/all/mean` for trainer-inference agreement, and `seq_len/*` + `is_truncated/*` for rollout health.
- **Configs** — the launch TOML next to the merged, resolved per-process configs the run actually started with.
- **Trace** — a per-episode rollout viewer with per-token overlays (advantage, entropy, sampling mismatch, loss/content masks), showing the transcript, a wall-clock timeline of the rollout, a terminal replay of model and tool activity, and a semantic graph of the model-call chain.
- **Logs** — the merged component logs (trainer, orchestrator, inference), also on disk under `<run_dir>/logs/`. The hand-started server of the reverse-text run logs to its own terminal.
- **Reports** — markdown reports written to `<run_dir>/reports/`, if any tooling produces them.

SLURM stdout/stderr and the generated sbatch script land in `<run_dir>/launcher/`.

## Checkpoint and resume

`swe.toml` writes a checkpoint to `<run_dir>/checkpoints/step_<N>` every 25 steps and keeps the last two. A checkpoint is 292 GB. `reverse-text.toml` writes none; add `--ckpt` to save one at the end of training.

To resume, re-run the same command with `--resume` (latest checkpoint) or `--resume.step <N>`, the same `--run.name` / `--output-dir`, and a `--max-steps` at least the target final step:

```bash
uv run rl @ examples/advanced/inkling2-mini/swe.toml \
  --slurm.template-path /shared/inkling2-mini/slurm/multi_node_rl.sbatch.j2 \
  --output-dir /shared/outputs/inkling2 \
  --run.name inkling2-swe \
  --resume --max-steps 300
```

`retrigger_on_resume = true` runs the evals of the checkpoint step again after a resume.

See [Training](../../../docs/training.md) for the full knobs and metrics reference, [Scaling](../../../docs/scaling.md) for SLURM and multi-node details, and [Advanced](../../../docs/advanced.md#custom-modeling) for how the trainer and the server implement this model.

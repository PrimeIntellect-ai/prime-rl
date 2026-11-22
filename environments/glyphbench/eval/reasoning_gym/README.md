# Reasoning Gym held-out evaluation

This directory contains the supported held-out transfer evaluation used to
compare `Qwen/Qwen3.5-4B` with a GlyphBench-trained, architecture-compatible
checkpoint. It is independent of the main GlyphBench environment evaluation.

The evaluator is pinned and resumable: it records the upstream Reasoning Gym
commit, materialized dataset digest, model fingerprint, generation settings,
raw completions, scorer outputs, and protocol hash. `protocol.json` is the
paper's 2,048-token protocol; `protocol_8192.json` is the documented 8,192-token
sensitivity variant.

## Requirements

- Python 3.12 and `uv`
- a local OpenAI-compatible server such as vLLM
- enough GPU memory for the selected model
- `PYTHONHASHSEED=0` set before starting the evaluator

Install the inference stack only on a GPU machine:

```bash
uv sync --extra rl
```

Prepare and verify the pinned Reasoning Gym environment once:

```bash
uv run python eval/reasoning_gym/reasoning_gym_eval.py prepare
uv run python eval/reasoning_gym/reasoning_gym_eval.py verify
```

The default cache is `~/.cache/glyphbench-reasoning-gym-eval`; results are
written under the ignored `runs/reasoning-gym-transfer/` directory.

## Serve a model

This example uses all eight GPUs on one node. Change `--data-parallel-size` for
your hardware. Both models must use the pinned baseline tokenizer and the same
serving options.

```bash
uv run --no-sync vllm serve Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --tokenizer Qwen/Qwen3.5-4B \
  --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --served-model-name glyphbench-reasoning-gym-target \
  --host 127.0.0.1 --port 8004 \
  --tensor-parallel-size 1 --data-parallel-size 8 \
  --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --compilation-config '{"custom_ops":["-silu_and_mul"]}' \
  --max-num-seqs 64 --seed 0 --dtype bfloat16 \
  --generation-config vllm --trust-remote-code \
  --enable-prefix-caching --gdn-prefill-backend triton
```

The batch and compilation settings mitigate a known issue in the pinned vLLM
runtime for dense Qwen3.5-4B and checkpoints with the same architecture. See
the [dependency notes](../../docs/DEPENDENCIES.md) before changing the model
or overriding these settings.

## Run or resume

Start with the smoke slice, then run the full protocol against the same server:

```bash
export PYTHONHASHSEED=0

uv run python eval/reasoning_gym/reasoning_gym_eval.py run \
  --model Qwen/Qwen3.5-4B --label qwen35-4b-baseline \
  --base-url http://127.0.0.1:8004/v1 --smoke

uv run python eval/reasoning_gym/reasoning_gym_eval.py run \
  --model Qwen/Qwen3.5-4B --label qwen35-4b-baseline \
  --base-url http://127.0.0.1:8004/v1
```

Stop the server, serve the local checkpoint with the same tokenizer/options,
and repeat with its actual path and a stable label:

```bash
uv run python eval/reasoning_gym/reasoning_gym_eval.py run \
  --model /path/to/glyphbench-checkpoint \
  --label glyphbench-checkpoint-step-450 \
  --base-url http://127.0.0.1:8004/v1
```

Re-running the same command resumes matching completion files. The evaluator
rejects incompatible model identities and comparison inputs.

## Compare results

Resolve run directories and build the paired comparison:

```bash
uv run python eval/reasoning_gym/reasoning_gym_eval.py resolve-run \
  --model Qwen/Qwen3.5-4B --label qwen35-4b-baseline

uv run python eval/reasoning_gym/reasoning_gym_eval.py compare \
  --baseline runs/reasoning-gym-transfer/BASELINE_RUN \
  --checkpoint runs/reasoning-gym-transfer/CHECKPOINT_RUN \
  --output runs/reasoning-gym-transfer/comparison
```

W&B publication is opt-in. Add `--publish-wandb` to a full `run` or `compare`
command after authenticating with `WANDB_API_KEY`. The project/group come from
the protocol; the entity follows `WANDB_ENTITY` or the authenticated default
account.

To use the 8,192-token sensitivity protocol, set
`REASONING_GYM_PROTOCOL_PATH=eval/reasoning_gym/protocol_8192.json` for every
prepare, verify, run, resolve, and compare command. Keep its results in a
separate directory with `--runs-root` so the two protocols remain unambiguous.

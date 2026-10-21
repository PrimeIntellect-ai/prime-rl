<!-- PR description for adding a model. Fill every <...>, delete rows that don't apply, keep the
     section order. Numbers come from the runs, not from expectations. Write "not run" instead of
     leaving a section out. Keep it in sync with the branch on every push. -->

## Summary

Adds `<arch>` (`<HF config model_type>`) to prime-rl: <dense | MoE, total / active params, attention type, anything unusual>.

- Checkpoint: [`PrimeIntellect/<Model>-BF16`](https://huggingface.co/PrimeIntellect/<Model>-BF16), converted from [`<org>/<Model>`](https://huggingface.co/<org>/<Model>) (<source format>) | upstream bf16 checkpoint used as-is
- EP: <yes/no>, CP: <yes/no>, router replay: <yes/no>

## 0. Checkpoint

| | |
|---|---|
| Source format | <fp8 blockwise / mxfp4 / ...> |
| Conversion | `<command>` |
| Checks | no scale tensors, no `quantization_config`, all tensors bf16 except: <list or "none"> |
| vLLM sanity vs original | <metric, e.g. mean logprob diff or eval score both sides> |

## 1. Modeling and trainer performance

Files: `src/prime_rl/trainer/models/<arch>/`, registration, `docs/advanced.md` row, `scripts/mini_moe.py` preset, `tests/unit/train/models/test_<arch>.py`.

Parity tests: <pass/fail, max abs diff of logits/grads>

`trainer-mfu.toml`, 1 × 8 <GPU>, `num_layers = <n>`, seq len <n>, random init, forced balanced routing:

| Model | MFU | Tokens/s/GPU | Peak memory (GiB) |
|---|---|---|---|
| `<new model>` | | | |
| `<reference model>` | | | |

## 2. vLLM vs trainer

vLLM `<version>`, attention backend `<name>`, MoE kernel `<name>`, KV cache `<dtype>`.

| Module | vLLM: weight / operand / output | vLLM kernel | Trainer: weight / operand / output | Trainer kernel | Configured by | Match? |
|---|---|---|---|---|---|---|
| | | | | | | |

Differences and justifications: <list or "none">

vLLM self-consistency (decode vs prefill KL): <value>

## 3. RL on math

`rl-math.toml`: `math-env`, 8k, batch 64 × group 8, sign-SGD lr <lr>, 100 steps, <n> trainer + <n> inference nodes. W&B: <link>

Step-1 mismatch KL: <value>. Max over all steps: <value> (bar: < 0.015).

<paste `uv run python skills/add-model/report.py <run_dir> ...` output>

Verdict: <flat / drifting, with numbers>

## 4. SWE ablation

`rl-swe.toml`: `scaleswe`, 65k, `max_turns = 30`, reasoning effort <level | n/a>, <steps> steps, <n> trainer + <n> × <n> inference nodes. W&B: <link>

<paste `report.py --block 20` output>

| Eval step | `swebench-verified` avg@1 | errors |
|---|---|---|
| 0 | | |

Verdict: <...>

## Checklist

- [ ] Parity tests pass, mini preset roundtrip is lossless
- [ ] Every row of the dtype table matches or is justified
- [ ] Math: mismatch KL < 0.015 on every step and flat over 100 steps
- [ ] SWE: mismatch KL flat, eval runs clean
- [ ] Renderer and vLLM parsers resolve for the checkpoint name
- [ ] `PrimeIntellect/<Model>-BF16` made public (`hf repos settings PrimeIntellect/<Model>-BF16 --no-private`) before merge

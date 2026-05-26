# Configuration

Every `prime-rl` entrypoint uses [`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config): TOML files for reproducible base configs, CLI flags for one-off overrides.

> **AI agents working in this repo:** the equivalent runbook is at [`skills/configs/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/configs/SKILL.md), with extra runtime hints (where config classes live, validator conventions, the trainer-side `token_export` flag) that aren't surfaced here.

## Table of Contents

- [Sources and Precedence](#sources-and-precedence)
- [TOML Composition](#toml-composition)
- [CLI Overrides](#cli-overrides)
- [Inspecting and Validating](#inspecting-and-validating)
- [Syntax](#syntax)
  - [Booleans](#booleans)
  - [Lists](#lists)
  - [Dicts](#dicts)
  - [Optional Sub-Configs](#optional-sub-configs)
  - [None](#none)
  - [Discriminated Unions](#discriminated-unions)
  - [Environments (`[[orchestrator.train.env]]`)](#environments-orchestratortrainenv)
- [Examples](#examples)

## Sources and Precedence

Field values come from three sources — Pydantic defaults, TOML files (passed with `@`), and CLI flags. They're layered in this order, with later sources winning:

1. **Defaults** declared on the Pydantic model.
2. **TOML files** passed with `@`, left to right — later files override earlier ones.
3. **CLI flags** in dotted, kebab-case form (`--model.name`).

## TOML Composition

The `@` token introduces a TOML file. Multiple `@` arguments compose left-to-right, deep-merged — unset fields in an overlay keep the base value:

```bash
uv run rl @ examples/reverse_text/rl.toml                      # one file
uv run rl @ base.toml @ overlay.toml                           # left to right
uv run rl --trainer @ trainer.toml --orchestrator @ orch.toml  # per-section
uv run rl @ base.toml --trainer @ trainer.toml                 # mixed
```

> Mind the space: `@ path/to/x.toml`, not `@path/to/x.toml`.

## CLI Overrides

CLI flags mirror the TOML tree using dots:

```bash
--max-steps 50                              # top-level
--model.name Qwen/Qwen3-4B                  # nested
--trainer.optim.lr 1e-5                     # double-nested
--inference.parallel.tp 4
```

> Field names are snake_case in TOML (`max_model_len`) and kebab-case on the CLI (`--max-model-len`).

> Renamed fields keep their old name as a validation alias — e.g. `rollouts_per_example` is still accepted in TOML and CLI after being renamed to `group_size`. Mixing the two names across sources is safe.

## Inspecting and Validating

```bash
uv run rl --help                                       # full schema
uv run rl @ rl.toml --dry-run --output-dir /tmp/check  # write resolved configs
```

## Syntax

### Booleans

CLI uses paired flags: bare `--flag` sets `True`, `--no-flag` sets `False`. TOML must be explicit:

```bash
uv run rl @ rl.toml --clean-output-dir       # True
uv run rl @ rl.toml --no-clean-output-dir    # False
```

```toml
clean_output_dir = true
```

### Lists

CLI accepts space-separated values or a JSON literal. TOML uses an array literal. Both forms target the same field:

```bash
uv run rl @ rl.toml --trainer.model.lora.target-modules q_proj k_proj v_proj
uv run rl @ rl.toml --trainer.model.lora.target-modules '["q_proj", "k_proj", "v_proj"]'
```

```toml
[trainer.model.lora]
target_modules = ["q_proj", "k_proj", "v_proj"]
```

Overlay TOMLs **replace** lists wholesale — an overlay that wants to add one item must still spell out the full list. For arrays of tables (e.g. environments), see [Environments](#environments-orchestratortrainenv).

### Dicts

CLI takes a JSON literal. TOML uses a table or inline-table. CLI dicts deep-merge with TOML dicts — CLI keys win on conflict but don't wipe the file's keys:

```bash
uv run rl @ rl.toml --orchestrator.train.env.0.args \
  '{"dataset_name": "openai/gsm8k", "dataset_subset": "main"}'
```

```toml
[[orchestrator.train.env]]
args = { dataset_name = "openai/gsm8k", dataset_subset = "main" }
```

### Optional Sub-Configs

Many sub-configs are typed `SomeConfig | None`. Two patterns enable them:

- **Bare flag with defaults**: `--model.compile` or, in TOML, an empty section `[model.compile]`. The sub-config materializes with all-default values.
- **Enable and set fields together**: `--model.compile.fullgraph` (CLI) or any populated `[model.compile]` table (TOML).

To **disable** a sub-config that's on by default, use `--no-<name>` on the CLI or assign the string `"None"` in TOML (see [None](#none)). This is how `[ckpt]`, `[model.lora]`, `[model.compile]`, `[trainer.wandb]`, etc. are turned on and off.

### None

TOML has no `null`. Use the string `"None"`, which the loader coerces:

```toml
[inference.model]
max_model_len = "None"
```

On the CLI: `--inference.model.max-model-len None`.

### Discriminated Unions

Loss, advantage, optimizer, scheduler, weight broadcast transport, and several others are discriminated unions. Set the `type` field to pick a variant:

```toml
[trainer.optim]
type = "muon"
lr = 1e-5
mu = 0.95
```

Omit `type` to keep the default variant.

### Environments (`[[orchestrator.train.env]]`)

Training environments are an array of tables — set one per env, optionally with sampling weights:

```toml
[[orchestrator.train.env]]
id = "math-env"
name = "gsm8k"
args = { dataset_name = "openai/gsm8k", dataset_subset = "main" }

[[orchestrator.train.env]]
id = "reverse-text"
ratio = 0.25  # 25% of batches; remaining 75% goes to math-env

[[orchestrator.eval.env]]
id = "math-env"
name = "gsm8k-eval"
args = { dataset_name = "openai/gsm8k", dataset_subset = "main" }
```

`args` is forwarded verbatim to the environment's `load_environment(**args)`.

The same `id` can appear multiple times across train and eval (or with different `args`) — useful for evaluating on a held-out split of the env you're training on, or comparing two configurations of the same env side by side. When `id` is reused, set a distinct `name` on each entry; `name` defaults to `id` and must be unique across all envs in the same group.

## Examples

The shipped end-to-end examples in [`examples/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples) are the canonical, kept-up-to-date references — the rest of the repo's TOMLs (under `configs/`) are CI- and debug-internal and may drift. Each example directory has its own README with the full launch story.

**Basic** (1–8 GPUs):

- [**Reverse Text**](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/reverse_text) — `Qwen3-0.6B` reversing a chunk of text. Tiny single-turn SFT + RL; runs on a single consumer GPU in minutes.
- [**Wordle**](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/wordle) — `Qwen3-1.7B` playing Wordle. Multi-turn SFT + RL; 2–4 H100s.
- [**Alphabet Sort**](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/alphabet_sort) — `Qwen3-4B-Instruct-2507` sorting names alphabetically. Multi-turn LoRA RL without SFT warmup; one H100.
- [**Wiki Search**](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/wiki_search) — `Qwen3-4B-Instruct-2507` answering trivia by web-searching Wikipedia. Multi-turn with tool use.
- [**Hendrycks Sanity**](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/hendrycks_sanity) — `DeepSeek-R1-Distill-Qwen-1.5B` on a filtered MATH subset. Useful for algorithm ablations.

**Advanced** (32–2048 GPUs, SLURM):

- [**Qwen 3 30B – A3B Math**](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/qwen30b_math) — `Qwen3-30B-A3B` on hard math.
- [**Qwen 3 30B – A3B SWE**](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/qwen30b_swe) — `Qwen3-30B-A3B` on hard SWE.
- [**INTELLECT-3.1**](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/Intellect-3.1) — reproduces our INTELLECT-3.1 training run.
- [**MiniMax-M2.5 SWE**](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/minimax_m2.5_swe) — `MiniMax-M2.5` on agentic SWE.
- [**High-throughput GLM-5**](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/glm5_pd_disag) — `GLM-5` with P/D disaggregation and FP8 inference.

### Worked Example: Compose, Override, Dry-Run

Start from a shipped base config, override two fields on the CLI, and dry-run:

```bash
uv run rl @ examples/reverse_text/rl.toml \
  --wandb.name my-experiment \
  --trainer.optim.lr 5e-6 \
  --output-dir /tmp/reverse-dry \
  --dry-run
```

Then inspect the resolved config:

```bash
ls /tmp/reverse-dry/configs/
# rl.toml  trainer.toml  orchestrator.toml  inference.toml
```

Each per-process TOML reflects the final, validated configuration that the actual run would consume — exactly what each process sees when started standalone (`uv run trainer @ /tmp/reverse-dry/configs/trainer.toml`, etc.). This is the easiest way to bisect a misbehaving config: dry-run a known-good base, dry-run your overlay, diff the two.

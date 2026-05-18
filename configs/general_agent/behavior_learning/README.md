# General Agent RLM Behavior Learning

This folder contains the configs, prompt appendices, uploaded-run links, and rollout evidence used to mine harness-specific behavior-learning signals from GPT-5.5 `general-agent-solver-rlm` rollouts.

## Table Of Contents

- [Artifacts](#artifacts)
- [Workflow Walkthrough](#workflow-walkthrough)
  - [1. Prepare The Harness](#1-prepare-the-harness)
  - [2. Run GPT-5.5 Discovery Rollouts](#2-run-gpt-55-discovery-rollouts)
  - [3. Upload The Saved Evals](#3-upload-the-saved-evals)
  - [4. Read And Analyze Rollouts](#4-read-and-analyze-rollouts)
  - [5. Distill Harness Behavior Rubrics](#5-distill-harness-behavior-rubrics)
  - [6. Use The Evidence Review](#6-use-the-evidence-review)
- [Harness-Specific Task-Agnostic Behavior Rubrics](#harness-specific-task-agnostic-behavior-rubrics)
- [How The Rubrics Are Wired Into The Env](#how-the-rubrics-are-wired-into-the-env)
- [Prompt Variants](#prompt-variants)
- [Curated Evidence Highlights](#curated-evidence-highlights)

## Artifacts

- `eval_rlm_gpt55_discovery.toml`: preserved GPT-5.5 discovery eval config.
- `README.md`: this walkthrough and curated evidence review.
- `rl_qwen3_30b_a3b_rlm_baseline.toml`: Qwen3-30B-A3B RLM baseline config.
- `rl_qwen3_30b_a3b_rlm_behavior.toml`: behavior-reward config.
- `rl_qwen3_30b_a3b_rlm_extended_prompt.toml`: extended-prompt ablation config.
- `prompts/standard.md`: task-solving guidance plus compact operational guidance.
- `prompts/extended.md`: standard guidance plus generic metapattern code examples.

## Workflow Walkthrough

### 1. Prepare The Harness

The PR wires `general-agent` and `rlm` into the root dependency graph, adds behavior-reward plumbing plus prompt loading from config-side text files in `deps/research-environments`, and adds matched Qwen3 RLM ablation configs in this folder.

Install and import checks:

```bash
uv sync --all-extras
uv run --no-sync python -c "import general_agent, rlm"
```

### 2. Run GPT-5.5 Discovery Rollouts

The initial discovery did not use a judge. It ran `general-agent-solver-rlm` directly against GPT-5.5 through Prime inference.

Smoke test:

```bash
uv run vf-eval general-agent-solver-rlm -n1 -r1 -d -v
```

Tier sweep pattern:

```bash
uv run vf-eval general-agent-solver-rlm -n10 -r5 -c -1 -i -s -d -v -m openai/gpt-5.5 -a '{"min_tier": 0, "max_tier": 0}'
```

Repeat the tier sweep for `min_tier = max_tier = 0..4`. The saved runs used 10 tasks per tier and 5 rollouts per task, for 250 total rollouts.

The preserved TOML config is kept for reproducible discovery runs:

```bash
uv run vf-eval configs/general_agent/behavior_learning/eval_rlm_gpt55_discovery.toml
```

### 3. Upload The Saved Evals

Each tier output directory was uploaded with `prime eval push`:

```bash
uv run prime eval push <eval-dir> --env primeintellect/general-agent --name "<descriptive name>" --plain --output json
```

Uploaded runs:

- t0: `outputs/evals/general-agent-solver-rlm--openai--gpt-5.5/e90cffd1/results.jsonl`; Prime eval: [fbnl8oi9h1prxrrmx5t2mqb6](https://app.primeintellect.ai/dashboard/evaluations/fbnl8oi9h1prxrrmx5t2mqb6)
- t1: `outputs/evals/general-agent-solver-rlm--openai--gpt-5.5/131446d1/results.jsonl`; Prime eval: [ioeqs48lckl2cblpl0cjjhh1](https://app.primeintellect.ai/dashboard/evaluations/ioeqs48lckl2cblpl0cjjhh1)
- t2: `outputs/evals/general-agent-solver-rlm--openai--gpt-5.5/796eb36a/results.jsonl`; Prime eval: [gzok7tipb9xa2236x2s2xsk0](https://app.primeintellect.ai/dashboard/evaluations/gzok7tipb9xa2236x2s2xsk0)
- t3: `outputs/evals/general-agent-solver-rlm--openai--gpt-5.5/9b324998/results.jsonl`; Prime eval: [gfgzfmxwep7juxaavwhmk0u4](https://app.primeintellect.ai/dashboard/evaluations/gfgzfmxwep7juxaavwhmk0u4)
- t4: `outputs/evals/general-agent-solver-rlm--openai--gpt-5.5/2659145c/results.jsonl`; Prime eval: [cpd3zxs2dhzpd14q9ywkvpio](https://app.primeintellect.ai/dashboard/evaluations/cpd3zxs2dhzpd14q9ywkvpio)

GPT-5-mini prompt smoke uploads:

- standard prompt: `outputs/evals/general-agent-solver-rlm-prompt-smoke-gpt5mini/standard/evals/general-agent-solver-rlm--openai--gpt-5-mini/dbc79cda`; reward `0.0`; Prime eval: [buwz3yqn7dcqyk0egq6twls1](https://app.primeintellect.ai/dashboard/evaluations/buwz3yqn7dcqyk0egq6twls1)
- extended prompt: `outputs/evals/general-agent-solver-rlm-prompt-smoke-gpt5mini/extended/evals/general-agent-solver-rlm--openai--gpt-5-mini/12caab9b`; reward `1.0`; Prime eval: [g4335v8rqh7ffz9a585410d0](https://app.primeintellect.ai/dashboard/evaluations/g4335v8rqh7ffz9a585410d0)

### 4. Read And Analyze Rollouts

Coverage: all 250 rows were read, split by tier across subagents and checked again with a local indexer.

The curated examples below were checked manually against the saved `completion` tool calls.

Tool-call references below use completion message indexes from the saved row: `t2/r35/c24` means tier 2, row 35, `completion[24].tool_calls[0]`.

### 5. Distill Harness Behavior Rubrics

The behavior categories below are deliberately harness-specific and task-agnostic. They score how the agent uses the RLM/IPython/tool harness, not whether it knew a particular domain rule.

### 6. Use The Evidence Review

The detailed sections after the behavior rubrics provide positive and negative examples. They should be used for prompt examples, judge rubric calibration, and sanity-checking whether a proposed behavior reward is too task-specific.

## Harness-Specific Task-Agnostic Behavior Rubrics

These are proposed behavior rubrics for the RLM harness. They describe how the agent uses the notebook-like execution environment, async skills, local files, and verifier/state artifacts. They intentionally avoid task-semantic rubrics such as entity matching, exact budget arithmetic, appointment safety, or domain-specific constraint satisfaction.

### Tool Contract Discovery

Key: `tool_contract_discovery`

Discover how to call the available skills before using them for consequential actions.

Good evidence: `inspect.signature`, `help`, `SKILL.md`, wrapper source, CLI `--help`, or harmless probe calls that clarify argument names, return shapes, side effects, and error modes.

### Persistent IPython State Management

Key: `persistent_ipython_state_management`

Use the persistent IPython workspace as memory: store compact variables, reuse fetched data, name intermediate results, and avoid repeatedly dumping large raw outputs.

Good evidence: cached lists or maps, selected candidate variables, compact summaries, reusable helper functions, and later cells building on earlier state instead of starting over.

### Programmatic Multi-Call Sequencing

Key: `multi_call_programmatic_sequencing`

Execute dependent tool calls as an explicit ordered workflow inside code, rather than as disconnected one-off calls.

Good evidence: operation plans encoded in lists or dictionaries, `step` / `do` / `call` wrappers, labeled logs, and ordered create-update-verify style workflows.

### Batch, Loop, And Parallel Execution

Key: `loop_batch_parallel_execution`

Use loops, comprehensions, helper functions, and `asyncio.gather` to scale repeated tool use across candidates or repeated checks.

Good evidence: batched availability checks, candidate loops, parallel independent reads, table/group summaries, and compact printed results.

### Harness Introspection

Key: `harness_introspection`

Escalate from public skill docs to local harness artifacts when needed: skill source, task files, state files, logs, or verifier code.

Good evidence: reading `SKILL.md`, generated skill wrapper source, task source files, `db.json`, `db_final.json`, or verifier functions to clarify behavior.

### Error-Aware Recovery

Key: `error_aware_recovery`

Use tool errors and surprising outputs as diagnostics, then adapt the plan while preserving useful state.

Good evidence: inspecting the failing signature or policy, changing only the invalid call arguments or invalid step, retrying with a justified alternative, and continuing from prior successful work.

### Verification And Audit

Key: `verification_and_audit`

Use available harness feedback to check correctness before and after mutations.

Good evidence: precondition checks, simulation on copied state, final state re-listing, `db_final.json` inspection, direct verifier calls when available, and final summaries grounded in observed state.

### IPython Syntax

Key: `ipython_syntax`

Use IPython-native syntax when it is the shortest reliable way to inspect objects, recover notebook state, capture shell output, interpolate Python values into shell commands, or debug failures.

Good evidence: `?` / `??`, `%pdoc`, `%pdef`, `%pfile`, `%psource`, `%psearch`, `%who`, `%whos`, `In`, `Out`, `_`, `__`, `___`, `%history`, captured shell output like `files = !cmd`, shell interpolation with `$var` or `{expr}`, `%debug`, `%pdb`, and `%xmode`.

Not top-level harness rubrics: grounded entity resolution, exact arithmetic, deterministic mutation IDs, appointment safety, budget compliance, or any domain-specific constraint. Those can be evidence inside examples, but the rubrics should score harness strategy rather than task content.

## How The Rubrics Are Wired Into The Env

Runtime source of truth: `deps/research-environments/environments/general_agent/general_agent/solver/rlm/behavior.py`.

The `BEHAVIORS` tuple defines the rubric keys, titles, descriptions, positive cues, and negative cues. `BehaviorRewardRubric` uses that tuple in three places:

- Judge prompt construction: `_judge_system_prompt()` renders every behavior in `BEHAVIORS` and asks the judge to return JSON with a top-level `summary` plus a top-level `behaviors` object.
- Applicability: every behavior judgment includes `applicable`, `score`, and `evidence`; inapplicable behaviors are excluded from the behavior mean. Parsed state also records whether each behavior key was present as `judged`.
- Reward aggregation: `task_reward = max(db_hash, verify)`, `behavior_reward` is the solution-gated applicable behavior mean in `[0, 1]`, and `final_reward = task_reward + behavior_reward_alpha * behavior_reward`.
- Metrics: every behavior becomes a separate metric named `behavior_<key>`, for example `behavior_tool_contract_discovery`. Aggregate metrics include `task_reward`, `behavior_reward`, `final_reward`, `behavior_applicable_mean`, and `behavior_judged_count`.

Behavior rewards are solution-gated. If `task_reward` is not exactly `1.0`, `behavior_reward` is `0.0` even though behavior judging still runs and logs behavior metrics for audit. `task_reward` is the max of the existing `db_hash` and `verify` checks, so behavior shaping cannot reward unsolved rollouts.

The behavior judge scores only operating strategy, not task correctness. For applicable behaviors it uses these score anchors:

- `0.0`: absent or harmful.
- `0.25`: weak, accidental, or mostly ineffective evidence.
- `0.5`: partial evidence with important gaps.
- `0.75`: solid useful evidence with minor omissions or limited opportunity.
- `1.0`: exemplary use for the available opportunity.

The env args are exposed by `general_agent.solver.rlm.env.load_environment`:

```toml
[orchestrator.train.env.args]
behavior_judge_model = "..."
behavior_reward_alpha = 1.0
behavior_judge_sampling_args = { temperature = 0.0 }
```

Behavior judging is enabled when `behavior_judge_model` is set. Omit `behavior_judge_model` to disable behavior rewards. The judge defaults to Prime inference at `https://api.pinference.ai/api/v1` using `PRIME_API_KEY`; override `behavior_judge_base_url` and `behavior_judge_api_key_var` only for a different provider. Judge mode fails early if the configured API key env var is missing. One judge JSON response is cached per rollout and reused for all behavior metrics. The parsed judge response is kept in `state["behavior_judge_response"]`, the normalized per-behavior state is kept in `state["behavior_results"]`, and the top-level summary is copied to `state["behavior_judge_summary"]`.

To save the judge artifacts from `vf-eval`, enable result saving and pass comma-separated state columns:

```bash
uv run vf-eval general-agent-solver-rlm -n1 -r1 -s -d -v \
  -m openai/gpt-5-mini \
  -C trajectory,behavior_judge_summary,behavior_judge_response,behavior_results \
  -a '{"behavior_judge_model":"openai/gpt-5-mini"}'
```

Prompt guidance is configured through `append_to_system_prompt`, which can be either literal prompt text or a path to a prompt file:

```toml
[orchestrator.train.env.args]
append_to_system_prompt = "configs/general_agent/behavior_learning/prompts/extended.md"
```

The env first tries to load the configured value as a path. If no file exists at that path, it treats the value as literal prompt text. In both cases, it forwards the final text to RLM as `append_to_system_prompt`, so RLM's generated base prompt still includes the dynamic cwd, skills, and message context. This is different from RLM's `system_prompt_path`, which replaces the generated prompt.

For additional behavior guidance, set `append_to_system_prompt` to one of the prompt files. Omit it to use RLM's generated base system prompt unchanged:

- `prompts/standard.md`: append the eight task-solving guidance items with minimal descriptions and operational guidance.
- `prompts/extended.md`: append the standard prompt plus generic metapattern code examples.

The README and `behavior.py` should stay aligned. The README documents the rubrics and evidence; `behavior.py` is what training and judge scoring execute.

## Prompt Variants

The prompt variants live in `configs/general_agent/behavior_learning/prompts` instead of the `general-agent` package:

- `standard.md` enumerates the eight harness-specific task-agnostic guidance items with short descriptions and operational guidance.
- `extended.md` includes the standard guidance plus generic metapattern code examples.

The extended-prompt ablation config uses:

```toml
[orchestrator.train.env.args]
local_checkout = "~/rlm"
append_to_system_prompt = "configs/general_agent/behavior_learning/prompts/extended.md"
```

## Curated Evidence Highlights

Tool-call references use completion message indexes from the saved row: `t2/r35/c24` means tier 2, row 35, `completion[24].tool_calls[0]`.

### Tool Contract Discovery

- t4 row 0, `3d_print_shop_t4`, reward 1.0: inspects signatures and docs before material, compatibility, cost, or submission calls.

  ```text
  t4/r0/c0 ipython
  for name in ['list_materials','list_printers','check_compatibility','get_print_cost','submit_print_job']:
      print(inspect.signature(obj.run))
      print(obj.__doc__)
  ```

### Persistent IPython State Management

- t4 row 26, `adventure_tour_t4`, reward 1.0: keeps selected equipment maps in variables and reuses them for final assignment.

  ```text
  t4/r26/c22 ipython
  day1_eq = {...}
  day2_eq = {...}
  for eid in sum(day1_eq.values(), []):
      assign_equipment_to_booking.run(b1, eid)
  ```

### Programmatic Multi-Call Sequencing

- t1 row 20, `additive_mfg_t1`, reward 1.0: executes cancel, calibrate, submit, start, inspect, complete, and final reads through a labeled wrapper.

  ```text
  t1/r20/c6 ipython
  await call('cancel old failed job J-OLD', ...)
  ...
  await call('complete Lens Prototype', complete_job.run(...))
  ```

### Batch, Loop, And Parallel Execution

- t2 row 25, `adventure_tour_t2`, reward 1.0: batches independent guide and equipment availability checks with `asyncio.gather`.

  ```text
  t2/r25/c4 ipython
  gav=await asyncio.gather(*[check_guide_availability.run(g, DATE) for g in gids])
  av=await asyncio.gather(*[check_equipment_availability.run(e, DATE) for e in ids])
  ```

### Harness Introspection

- t3 row 47, `airport_ops_t3`, reward 1.0: reads task `tools.py` and then verifies final DB state with task code.

  ```text
  t3/r47/c12 ipython
  print(Path('/workspace/general-agent/tasks',name,'tools.py').read_text()[:8000])
  ```

  ```text
  t3/r47/c20 ipython
  db=mod.TaskDB.load(Path('/workspace/.solver/db_final.json'))
  print('verify', mod.verify(db))
  ```

### Error-Aware Recovery

- t4 row 6, `3d_print_t4`, reward 1.0: after rush/price policy issues, inspects policy and changes only the invalid print quality while preserving the requested model.

  ```text
  t4/r6/c10 ipython
  print(await get_shop_policy.run())
  ```

  ```text
  t4/r6/c12 ipython
  submit_print_job.run(customer, model, fil, printer, quality=quality, ...)
  ```

### Verification And Audit

- t4 row 47, `airport_ops_t4`, reward 1.0: simulates assignments on copied DB before real mutations.

  ```text
  t4/r47/c14 ipython
  for f in db.flights:
      ... f.assigned_gate=g
  print('verify', mod.verify(db))
  ```

- t4 row 30, `adventurer_guild_t4`, reward 1.0: loads `db_final.json` and calls the task verifier after quest mutations.

  ```text
  t4/r30/c30 ipython
  db=mod.TaskDB.load(Path('/workspace/.solver/db_final.json'))
  print('verify', mod.verify(db))
  ```

### Negative Examples

- t4 row 7, `3d_print_t4`, reward 0.0: selected a semantic decoy instead of preserving the canonical requested model.

  ```text
  t4/r7/c12 ipython
  specs=[('Dragon Head Bust','mod-decoy-001'), ...]
  submit_print_job.run(... model_id=mid ...)
  ```

- t4 row 25, `adventure_tour_t4`, reward 0.0: accepted same-location bookings despite the task requiring different locations.

  ```text
  t4/r25/c20 ipython
  create_booking.run('TOUR-015',...)
  create_booking.run('TOUR-046',...)
  ```

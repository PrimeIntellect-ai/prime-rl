---
name: model-factory
description: Launch and monitor prime-rl runs on Model Factory with prime train, and retrieve their traces. Use for requests such as "run this on model factory", including custom code revisions, private datasets, credentials, and team setup. Explicit local or SLURM jobs use start-run.
---

# Model Factory

Use `prime train run CONFIG.toml` for managed training. `prime train CONFIG.toml`
is the same command. The platform provisions and launches the components. Pass a
single TOML filename directly; `@`, local `uv run rl` / `uv run sft`, and arbitrary
dotted training flags belong to the local entrypoints. Put training overrides in
the TOML. An explicit request to submit a SLURM job on a Model Factory cluster
still uses the [start-run skill](../training/start-run/SKILL.md).

All local paths below are relative to the selected prime-rl checkout. Discover
the user's checkout, credentials, team, and available CLI; do not assume a home
directory, secret-file location, organization, GPU type, or previous run.

## CLI, authentication, and team

```bash
command -v prime
prime --version
prime train run --help
```

Install a missing CLI with `uv tool install prime`; update an existing tool with
`uv tool upgrade prime` when a required feature needs it. Check which executable
the shell resolves: a project environment can shadow the tool installation.

Authenticate with `prime login`, an existing exported `PRIME_API_KEY`, or the
secure prompt from `prime config set-api-key`. For automation, load the key from
the user's chosen secret store into the process environment. Do not put its value
in command arguments, TOML, Git, or chat. A file on disk is not automatically
exported. The [CLI config implementation](https://github.com/PrimeIntellect-ai/prime/blob/main/packages/prime/src/prime_cli/core/config.py)
gives `PRIME_API_KEY` and nonempty `PRIME_TEAM_ID` precedence over saved settings.

For an API-key workflow, after loading the key securely:

```bash
: "${PRIME_API_KEY:?Load a Prime API key for the launching account}"
export PRIME_API_KEY
```

Inspect the effective account configuration and available teams:

```bash
prime config view --plain
prime teams list --plain -o json
```

`config view` masks the API key. Reuse the account/team the user selected; if the
intended owner is ambiguous, resolve it before launching billable work. Set the
chosen team ID from that list with `prime config set-team-id "$TEAM_ID"`, or use
`export PRIME_TEAM_ID="$TEAM_ID"` for the current process. An exported team ID
overrides `set-team-id`. To intentionally use a personal account, unset
`PRIME_TEAM_ID` and use `prime config remove-team-id`. Team choice also scopes
volumes and run access. See [CLI configuration](https://docs.primeintellect.ai/cli-reference/config-cli)
and [teams](https://docs.primeintellect.ai/cli-reference/teams).

The launching credential and the remote run credential have different roles:
the platform supplies a run-scoped `PRIME_API_KEY` to the Model Factory job. Do not
include the launching key in `-e` or `--env-file`. Read
[data and secrets](references/data-and-secrets.md) before forwarding credentials
or using private Hugging Face data.

## Choose the source and a checked-in example

For custom training code, branches, commits, dependencies, or environments, read
[custom code](references/custom-code.md) first. Select the runtime revision before
adapting or validating its config.

Start from an existing example in that revision, copying it to a task-specific
TOML if edits are needed. Do not use personal debug configs or prior experiment
artifacts as the default recipe.

| Task | Starting point |
| --- | --- |
| Small RL run | [examples/basic/reverse-text/rl.toml](../../examples/basic/reverse-text/rl.toml) |
| SFT | [examples/basic/reverse-text/sft.toml](../../examples/basic/reverse-text/sft.toml) |
| Larger model or environment | Select a matching checked-in TOML under [examples/advanced](../../examples/advanced) |

```bash
prime train models --plain -o json
prime train gpus --plain -o json
```

Use the exact model ID and GPU type available for the selected account/team.
The [Model Factory RL docs](https://docs.primeintellect.ai/hosted-training/hosted-rl)
describe uncached HF downloads on on-demand capacity, while reserved clusters
require cached weights or weights on a volume. Check the actual dispatch path; an
`HF_TOKEN` grants repository access, not cluster capacity or model-cache access.

Adapt the example to the requested experiment:

- Remove SLURM configuration and machine-local `output_dir` / `run.dir` paths.
  Put prompts and other required assets in the launched source or a supported
  volume. A path in TOML does not upload the file.
- Set `[deployment]` explicitly, including trainer and inference GPU counts.
  Check TP/DP/EP/CP and context length against the model, hardware, and schema.
  Preserve `optimization_dtype` and `reduce_dtype` unless explicitly requested.
- Use the revision's `[run]` and `[monitors.wandb]` schema for names and logging;
  consult the [configs skill](../configs/SKILL.md) for config structure. The
  selected revision's schema takes precedence over snippets from another build.
- Keep RL train/eval sources separate under `orchestrator.train.source` and
  `orchestrator.eval.source`. Dataset arguments belong to the selected taskset's
  config under `env.taskset`; inspect its supported fields before setting them.
- SFT uses `[data]` and needs real data staged on a volume. The checked-in SFT
  example also enables online eval; check CLI and cluster support before using
  it. See [data and secrets](references/data-and-secrets.md) for both paths.
- Keep `[ckpt]` and attach a named volume when checkpoints must persist. Choose
  its size for the dataset, model, and retained checkpoints; see
  [volume sizing](https://docs.primeintellect.ai/hosted-training/volume-capacity).

## Validate and dispatch

If `prime train validate --help` is available, use that command with the same
`--ref` / `--image-tag` and `--sft` selection as the intended launch. It is a
validation command, not a GPU run. Otherwise, validate from the matching checkout
and its installed environment (`CONFIG_PATH` names the prepared TOML):

```bash
uv run --no-sync python - "$CONFIG_PATH" <<'PY'
import sys
import tomllib
from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig

data = tomllib.loads(Path(sys.argv[1]).read_text())
schema = SFTConfig if "data" in data and not {"trainer", "orchestrator"} & data.keys() else RLConfig
schema.model_validate(data)
print("Config valid")
PY
```

Keep dispatch metadata (`--ref`, image, GPU, volume) in CLI flags so this fallback
receives only native prime-rl fields. Validation checks the config, not remote
imports, private dataset access, capacity, or feature entitlements. For custom
code, verify that the validator actually imports the selected revision's schema.

Summarize the prepared config, source/image, owner, GPU allocation, and expected
run length before dispatch. Carry out an already authorized launch; ask only for
unresolved choices or credentials that affect it. For an initial smoke run, use
a small step budget unless the user specified otherwise.

```bash
prime train run "$CONFIG_PATH" \
  --gpu-type "$GPU_TYPE" --volume "$VOLUME_NAME" \
  -e WANDB_API_KEY -e HF_TOKEN --yes --plain -o json
```

Include only the flags and exported secrets that the run needs. Add
`--ref "$CODE_REF"` for authorized custom source access, and `--sft` for SFT.
`--yes` skips the CLI prompt for the launch the user requested. Save the returned
`.run.runId`, config, source SHA, image selection, team, volume, and reported
placement/rate. Share the dashboard URL returned by the CLI.

## Monitor and finish

For metric comparisons, plots, or post-restart performance, read
[analyzing runs](../training/monitor-run/references/analyze-runs.md). Separate
pipeline warmup from steady-state metrics and establish trainer progress before
combining histories across attempts.

```bash
prime train get "$RUN_ID" --plain -o json
prime train components "$RUN_ID" --plain
prime train logs "$RUN_ID" --plain --tail 100
prime train logs "$RUN_ID" -c trainer --plain --since 5m --tail 500
prime train logs "$RUN_ID" -c inference --plain --since 5m --tail 500
prime train metrics "$RUN_ID" --plain --limit 10
prime train progress "$RUN_ID" --plain
```

For environment logs, use `--env` with a name returned by `components`. Follow
with `-f` or filter with `--search` / `--level ERROR` when needed. Confirm completed
training steps and inspect rollout/eval infrastructure errors; `RUNNING` alone
does not establish progress. Export needed component logs while pods still exist.

Uploaded traces persist separately from pods and can be retrieved after the run
ends. For rollout inspection or trace downloads, read
[retrieving traces](references/traces.md): list by run ID with `prime traces list`,
then save raw documents with `prime traces get --raw --dest`. Listing summaries
alone does not download the traces. The reference covers whole-run pagination,
per-step episodes, separate eval run IDs, and missing-upload checks.

Queued and image-pull states can be normal. After an ambiguous dispatch timeout,
check `prime train list` for the submitted run before retrying; do not create
duplicates. Deterministic config, import, credential, or sandbox setup errors
need a fix before another launch. Respect the user's run budget and stop conditions.

When asked to stop, or when an already agreed stop condition is reached:

```bash
prime train stop "$RUN_ID" --force --plain
prime train get "$RUN_ID" --plain -o json
```

Verify terminal state, wait for volume sync, and retrieve requested checkpoints
with `prime volumes get`. Report the run ID/link, actual completed steps, outcome,
and checkpoint/trace artifact locations. Follow-up runs need to remain within the
user's request.

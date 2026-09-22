# Flow

Launch a pipeline exported by an installed Python package. Verifiers supplies the execution
engine; Prime-RL owns the config, run directory, logs and dashboard registration.

```toml
output_dir = "outputs"

[run]
name = "candidates"

[flow]
id = "parallel"
model = "MODEL"
samples = 4
```

```sh
uv run flow @ run.toml
uv run flow inspect --root outputs/candidates
uv run flow steer --root outputs/candidates --unit task --status ready
uv run flow @ run.toml
uv run flow drain --root outputs/candidates
```

The package exports exactly one `Flow[Config]` subclass in `__all__`. Its Pydantic config
is addressed under `flow`, including overrides such as `--flow.samples 8`. An existing
pipeline config can be loaded with `--flow @ pipeline.toml --flow.id PACKAGE`.
See the [Verifiers examples](https://github.com/PrimeIntellect-ai/verifiers/tree/feat/flow/examples/flow)
for authoring, native agent calls, parallel recovery and artifact revisions.

Repeat the launch command with the same run name to resume saved workflow state and reuse
successful keyed calls. `inspect` reports current state and executing stages. `steer`
changes a unit's next stage/status or adds a note; it does not interrupt a model conversation.
Data edits use `--data patch.json --expected REVISION` while the unit is settled;
`REVISION` is the integer at `state.revision` in the inspection. Each unit keeps one
current `state.json`, replaced atomically under its write lock.

Drain finishes running calls and stops new work. Remove the run's `drain` file before
relaunching. Ctrl-C drains once; a second signal cancels. With `flow.stay_alive = true`,
the process stays available between stages for a monitor to release work. It reads named
limits from `pools.json` about every two seconds; replace that file atomically to resize.

`flow.json`, `transitions.jsonl`, saved traces, call records and reports use Verifiers'
contracts. The dashboard reads these files; it does not schedule or recover the pipeline.

The Traces tab shows live agent work using the same delta reader and viewer as served
episodes. A retry replaces the live attempt; completion opens the saved trace. Flow's
`live/<trace_id>.jsonl` files carry the unit, stage, execution and call identity.
Monitors can query `GET /api/runs/{run}/live`, then `/api/runs/{run}/live/{trace_id}`
for an assembled trace, or read locally with
`uv run python -m prime_rl.monitors.file.traces.live <run_dir> [trace_id]`.

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

Repeat the launch command with the same run name to resume Git checkpoints and reuse
successful keyed calls. `inspect` reports committed state and executing stages. `steer`
changes a unit's next stage/status or adds a note; it does not interrupt a model conversation.
Data edits use `--data patch.json --expected SHA` while the unit is settled.

Drain finishes running calls and stops new work. Remove the run's `drain` file before
relaunching. Ctrl-C drains once; a second signal cancels. With `flow.stay_alive = true`,
the process stays available between stages for a monitor to release work. It reads named
limits from `pools.json` about every two seconds; replace that file atomically to resize.

`flow.json`, `transitions.jsonl`, saved traces, call records and reports use Verifiers'
contracts. The dashboard reads these files; it does not schedule or recover the pipeline.

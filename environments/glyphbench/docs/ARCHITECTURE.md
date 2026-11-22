# Architecture

GlyphBench separates game simulation from model prompting, rollout control,
and result inspection. A game can run through the direct Python API without
creating a model client.

## Directory layout

```text
src/glyphbench/
    core/                   # Environment base class, observations, actions, registry
    envs/                   # Game implementations and optional upstream adapters
    envs/craftax/docs/      # Rules and tutorials included in Craftax prompts
    protocol.py             # Framework-independent prompting and action parsing
    termination.py          # Context-limit stop classification
    verifiers_v1.py         # Native Verifiers taskset and environment for training
    verifiers_harness.py    # In-process transport for v1 model interactions
    verifiers_integration/  # Evaluation loader, memory handling, and rubric
    pro_harness/            # Long-horizon agent and provider clients
    plotting/               # Result loading and plotting
    cli.py                  # Environment listing and trajectory replay

eval/                       # Evaluation scripts and held-out reasoning panel
configs/                    # Model endpoints, training recipes, and task manifests
scripts/                    # Demo, replay, asset, and training utilities
docs/                       # User and contributor guides
notebooks/                  # Return-distribution analysis
third_party/                # Pinned optional Craftax, NLE, and AgenticK submodules
```

## Simulation

`BaseGlyphEnv` owns the public `reset` and `step` lifecycle, turn counts,
termination, and the standard return bound. Subclasses implement game rules
and produce `GridObservation` values. `ActionSpec` defines the action names,
descriptions, and aliases. The registry maps task IDs to environment classes;
optional adapters load their external dependencies when instantiated.

The six standard suites contain 303 tasks. The archival Atari, CraftaxFull,
and NetHack suites add 59 extended tasks. Experimental AgenticK adapters and
the internal `__dummy` fixture are separate from the benchmark. The canonical
evaluation exclusions live in `eval_defaults.py`; the
[catalog](ENVIRONMENTS.md) is generated from the registry.

## Model interaction

`protocol.py` provides prompt rendering and strict action parsing for the
native training path. `verifiers_v1.py` streams tasks and seeds and drives each
game episode through Verifiers interactions. `verifiers_harness.py` runs the
conversation in-process while routing model requests through Verifiers'
authenticated interception endpoint. Upstream Verifiers and Prime-RL handle
scheduling, tokenization, prefix caching, and packing.

`verifiers_integration/` provides the existing evaluation loader and result
format, including configurable frame history, task-specific memory defaults,
and parse retries. `pro_harness/` provides a separate long-horizon agent with
persistent notes, provider integrations, and full-game observation options.
See [agent integration](INTEGRATION.md) for the supported interfaces.

## Inspection

The CLI reads saved Verifiers evaluation `results.jsonl` files and renders them
with Rich. Replay uses the evaluation parser and environment action vocabulary
to display the parsed action alongside the observation and available
trajectory metadata. The standalone trajectory script reads its own JSONL
format and can export glyph-based GIFs. The plotting helpers load evaluation
results for aggregate and per-suite analysis.

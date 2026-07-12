# Deferred Prime-RL topology work

This is the living list of topology work that is known, intentionally deferred, and not
required for the first end-to-end `AgentGraph` training path. It records the boundaries of
the current implementation so temporary restrictions do not become accidental contracts.

The central path is already graph-native:

- Prime accepts only topology configs for native v1 environments.
- One dispatch runs one topology invocation and receives one `AgentGraph`.
- `group_size=N` groups N independently executed graphs for algorithmic credit.
- Dispatch, train/eval sinks, persistence, and monitoring carry graphs rather than flattening
  them into rollout objects.
- Each trainable trace is compiled from its existing token data using its own resolved sampling
  temperature; non-trainable traces remain available to the algorithm as graph context.
- General-purpose algorithms fail loudly unless a graph contains exactly one trainable trace.
- `proposer_solver` is the first explicit multi-trace algorithm: it centers solver rewards
  within each graph and proposer rewards across the graph group.
- The graph RPC preserves training-only tensors such as routed-expert data; JSON persistence
  strips them as before.

## General multi-trace algorithms

### Support arbitrary trainable graph shapes

`proposer_solver` is a topology-specific implementation, not a generic promise that every
multi-trace graph is trainable. All other built-in algorithms deliberately require exactly one
trainable trace. Variable fan-out is supported only where an algorithm explicitly validates the
graph and selects its training traces.

Future multi-trace algorithms must define, rather than infer:

- the accepted topology and agent/parent layout;
- which successful traces become training samples;
- how failed or missing traces affect credit;
- which comparisons happen within one graph and which happen across a Prime group; and
- how credit is assigned when fan-out differs between graphs.

Do not weaken the default one-trace check to make new topologies appear compatible. Add an
algorithm with an explicit graph contract.

### Add per-agent algorithms and loss routing

Prime currently selects one algorithm per configured topology. Its `action_loss_type` is stamped
onto every trace that algorithm accepts. A graph cannot yet route different agents through
different algorithms or loss components, such as RL for a proposer and CE or reference KL for a
solver.

A future design should keep the graph as the orchestration unit while making trace selection and
loss routing explicit. It must also define whether model references and algorithm lifecycle are
owned per topology, per role, or per algorithm instance.

### Add multi-seat and self-play credit

There is no built-in contract for chess seats, debate participants, adversarial agents, population
play, or other cases where multiple policy traces have asymmetric roles. These should be separate
algorithms that validate seat/role structure and define credit under draws, missing seats, failed
agents, and role swaps. They should not be forced through ordinary GRPO.

### Add algorithm scopes beyond graph and group only when needed

The current algorithm surface intentionally has two scopes:

- `score_graph(graph)` for credit available from one completed invocation; and
- `score_group(graphs)` for credit across the independently sampled replicas sharing a
  `group_id`.

Algorithms needing replay buffers, population state, cross-batch ranking, or trainer-time graph
objectives need a new explicit ownership and persistence contract. Do not store durable algorithm
state accidentally on an in-process algorithm object.

## Batching and filtering

### Define token batching for multi-trace algorithms

Multi-trace algorithms currently require graph-count `batch_size`; Prime rejects
`token_batch_size`. A proposer-solver graph can contain a variable number and length of trainable
traces, so one graph is not a stable amount of trainer work.

The eventual implementation should preserve complete algorithm groups through credit assignment,
then pack their accepted traces by tokens without splitting or silently dropping graph members.
It must define the meaning of batch progress, oversampling, overflow, and trainer step size when
graph fan-out varies.

### Define filters for multi-trace graphs

Prime currently rejects pre- and post-batch filters for algorithms that support multiple traces.
Existing filters inspect the sole training trace and annotate or drop the whole graph; that is
ambiguous when a graph contains several roles.

Before enabling filters, define whether each filter applies per trace, per agent role, or per
graph, and whether dropping one child:

- drops only that sample;
- drops the entire graph;
- invalidates within-graph credit; or
- changes the cross-graph baseline.

Filtering must happen after the signal it reads is available and must not silently change an
algorithm's comparison set.

## Failure and provenance semantics

### Define algorithm-owned partial-group policy

Prime dispatches exactly `group_size` graph invocations. Failed graphs still reach the sink so its
barrier completes, then the current training path removes unusable graphs and scores the surviving
partial group. Prime does not automatically schedule replacement graphs to restore the requested
group size.

This is acceptable for the first path but should become an explicit algorithm policy. Depending on
the algorithm, a partial group may be valid, require replacement replicas, or require the entire
group to be discarded. Any replacement logic must stay bounded and preserve task, policy-version,
and `group_id` semantics.

Within a graph, `training_traces()` excludes failed trainable traces. Multi-trace algorithms must
continue to decide whether the remaining shape is meaningful rather than relying on a universal
Prime rule.

### Stamp and validate per-trace model provenance

Verifiers stamps `trainable` and resolved sampling config onto every trace, but a trace does not yet
carry a self-contained identity for the resolved model/client that produced it. Prime therefore
validates token availability and temperature but cannot prove from the returned graph alone that an
RL or reference-KL trace came from the live policy rather than an agent-level client override.

If mixed model routing becomes common, stamp enough non-secret provenance onto each trace to let
the algorithm validate its source. Never persist API keys or full credential-bearing client
configs. Non-trainable judges may continue to use either the policy endpoint or an external model;
the stricter provenance requirement applies to traces selected for training.

### Consume a self-contained topology completion verdict

Verifiers has a deferred item to stamp `Topology.complete(graph)` onto the graph. Once that exists,
Prime should use the stamped verdict when classifying graph failures and deciding whether a handled
child failure invalidates training. Prime should not need to load topology code to reproduce that
decision.

## Metrics, records, and platform presentation

### Make multi-trace metrics role-aware

Token counts already sum over trainable traces, and sample upload iterates them. Scalar graph views
such as `reward`, `rewards`, `metrics`, `timing`, `stop_condition`, and scalar advantage still
project only the sole-trainable-trace case. For a graph with multiple trainable traces, several of
those views intentionally return neutral or empty values.

Consequently, topology-specific training can be correct while aggregate reward and advantage
telemetry is incomplete. Add role-keyed and graph-level metric views before treating dashboards as
authoritative for multi-trace algorithms. Avoid inventing one scalar reward for a graph whose
algorithm has several distinct credit populations.

### Preserve graph identity in the trainer only if an objective needs it

After the orchestrator assigns credit, accepted traces are flattened into the existing
`TrainingSample` batch. The trainer does not receive an `AgentGraph` or use graph structure. This is
the lean boundary for current objectives because graph-aware work is complete before shipping.

If a future loss needs graph structure at trainer time, extend the transport with stable graph,
trace, agent, and parent identifiers. Do not send the full Verifiers object graph merely for
observability.

### Make the platform UI fully graph-native

Prime's sample uploader emits one row per trainable trace and includes `graph_id`, topology, agent,
and parent metadata. This preserves useful lineage in the existing schema but does not make the
platform graph-native: the primary presentation unit is still a trace row, non-trainable context is
not uploaded as part of the same object, and the UI does not render graph structure or separate
within-graph from across-group credit.

The eventual platform contract should present one invocation as one graph with expandable traces,
role-specific rewards/advantages, parent links, and graph-level completion/error state.

## Scheduling and scalability

### Account for variable topology cost

The Prime dispatcher charges one permit per topology invocation. A one-call agentic judge and a
many-agent, many-turn topology therefore have the same scheduling cost. The topology server still
reuses the existing scalable process/server lifecycle, but Prime is intentionally blind to internal
model calls, runtimes, fan-out, and session turns.

Defer weighted permits, per-agent admission, and dynamic cost estimation until production evidence
shows which signal is useful. Any future scheme must coordinate with Verifiers' internal session
concurrency work and must not require Prime to understand topology implementation details.

### Add topology-internal operational visibility

Prime can report graph-level latency, tokens, errors, and topology name, but it does not yet expose
queueing, latency, model routing, or failure rates by agent role. Add role-aware telemetry alongside
cost-aware scheduling; do not couple it to the training algorithm contract.

## Repository migration and validation

### Port checked-in configs and examples to topology syntax

The new Prime schema intentionally rejects native v1 `id`, `taskset`, and `harness` environment
forms. Many checked-in configs, examples, and CI fixtures still use those forms. They should be
rewritten to explicit topology configs, usually the built-in `single-agent` topology, rather than
made valid through a compatibility branch.

This migration should include associated README commands and expected config snapshots. Until it
is complete, the entire historical test suite is not expected to validate under the topology-only
schema; focused graph-native tests are the source of truth for this branch.

### Run a broader live end-to-end matrix

The topology port has focused unit coverage for graph conversion, strict algorithm compatibility,
sampling provenance, training-sample extraction, and proposer-solver credit. Broader GPU/integration
runs remain useful for:

- ordinary GRPO over a single-agent topology;
- GRPO over an agentic-judge topology with a non-trainable judge;
- proposer-solver with variable solver fan-out and failures;
- routed-expert and multimodal tensors through the topology server;
- frozen-source SFT and reference-scoring algorithms on a single-trace topology; and
- external rather than Prime-spawned topology servers.

Failures in that matrix should update the relevant code, docs, configs, and skills together.

### Finish the terminology sweep

Some stable configuration keys, logs, comments, and class names still say `env` or `rollout`
(`train.env`, `max_inflight_rollouts`, and related operational messages) even though native
execution is graph-native. Rename only where it materially clarifies the public contract; avoid a
large mechanical rewrite of server lifecycle code that already has the correct behavior.

## Explicit non-goals

The following are deliberate exclusions, not deferred compatibility work:

- accepting native Prime v1 `taskset + harness` configuration;
- restoring Verifiers group rewards or environment-side atomic group execution;
- supporting the legacy v0 rollout/group protocol in the native topology path;
- flattening graphs back into traces at the dispatcher or algorithm boundary; and
- renaming Verifiers `Topology` to `Environment`.

Repository configs that still use an old syntax should be ported, not supported by adding another
runtime route.

## Cross-repository work

Verifiers-owned follow-ups—including session concurrency, graph completion persistence, graph-native
replay/dashboard/push, protocol isolation, and run-level state—are tracked separately in
[`deps/verifiers/TOPOLOGY_DEFERRED_WORK.md`](deps/verifiers/TOPOLOGY_DEFERRED_WORK.md). Prime should
consume those contracts when they land, but should not duplicate their implementation.

# Experimental deferred payload benchmark

This benchmark is stacked on PR #3806. It changes no runtime code, public API,
training configuration, or transport. It asks whether avoiding binary copies or
materializing fewer episodes is useful beyond the finite scalar accounting and
reference release already implemented by that PR.

## Compared paths

| Path | Receive behavior | Accepted output |
| --- | --- | --- |
| `native` | Native MessagePack unpacking, full `WireEpisode` validation, #3806 numeric extraction; release rejected episodes | Native `TrainingSample` bytes |
| `typed-bytes` | Typed MessagePack decoder with owned binary bytes, then identical native validation/accounting | Same native sample bytes |
| `typed-views` | Same typed schema, binary fields as views on immutable input; identical native validation/accounting | Same native sample bytes |
| `deferred-views` | Small scalar summary plus an opaque encoded episode view; materialize selected episodes later | Same native sample bytes |

The typed-bytes control separates decoder changes from binary-copy avoidance.
The native and typed paths preserve the episode interface in this experiment.
The deferred interface exists only inside the benchmark: `CompletedRollout`
contains a summary and a `PayloadRef` whose `materialize()` returns a native
`WireEpisode`. `release()` drops the handle's ownership; existing NumPy views
retain their backing allocation until the native episode is released.

**Deferred admission is an upper-bound experiment.** It assumes scoring,
callbacks, group finalization and admission have already produced trustworthy
summaries and selection decisions. The existing sink needs native graphs for
those operations. This benchmark does not implement their relocation, and does
not establish that the deferred interface can replace `EnvClient.run()`.

## Run

From a checkout with the pinned submodules and normal CPU dependencies installed:

```bash
git submodule update --init --recursive
uv run --locked python tools/experimental_payloads/benchmark.py /tmp/payload-results
```

The output directory must not already exist. Defaults are three repeats, eight
independent episodes, 32,768 tokens per episode, and a 1 MiB encoded-image-shaped
string per episode. Each case runs in a fresh process, sequentially. Mode order
rotates between repeats. Every episode has a branched graph, routing arrays,
sampling masks, custom metrics and synthetic assigned advantages.

To exercise a private saved complete episode instead:

```bash
uv run --locked python tools/experimental_payloads/benchmark.py /tmp/saved-results \
  --episode /private/complete-wire-episode.msgpack --episodes 8
```

That file must use the complete Python-mode MessagePack episode format, including
raw tensor fields. JSON trace records exclude routing arrays and sampling masks;
they are not suitable substitutes. Each replay is decoded independently. Schema
or fidelity failures abort the benchmark; private payloads are never committed.

## Measurements and checks

Receive, deferred materialization and native sample construction/encoding each
report elapsed time and process CPU time. The deferred producer's additional
summary calculation and envelope encoding are reported separately, so receiver
savings cannot hide shifted work. Fixture generation and reference construction
are outside those timings. Selection rates are 0%, 25% and 100%; output records
also include the actual integer number selected.

Exact checks compare every accepted episode field, complete native sample
bytes (including routing, masks and image references), per-group numeric
observations, and aggregated metric key sets and values. Full comparisons are
outside the receive/materialize/sample timings. Their duration is reported.

A 2 ms asyncio heartbeat runs while decoding and sample construction are
thread-offloaded. Work-phase lag is separate from lag including synchronous
verification and final collection. Memory is measured as RSS and PSS at phase
boundaries on Linux. The process high-water RSS explicitly includes fixture and
reference setup; it is not claimed as the receive-phase peak. No allocator or GC
settings are changed.

Safety checks demonstrate immutable binary views, native field fidelity,
materialized-episode survival after handle release, rejected handle reuse, and
malformed-input rejection. A mutable-input control demonstrates why an input
slot cannot be overwritten while views survive.

## Scope and decision

This is a completed-episode microbenchmark using the actual #3806 metric
extraction and actual `trace_to_samples` implementation. It does not replay
streaming delta ordering, network transfer, multi-process output handoff, sink
callbacks, sampler state, checkpointing, trainer packing, image pixel decoding,
or GPU execution. The synthetic image payload exercises transport/reference
handling and is not a valid image for a vision forward pass.

Lower receiver time is not automatically lower total cost: include the deferred
producer and later materialization costs, and compare the 100% case. Binary views
can pin the entire input frame, including text already decoded into strings, so
memory can increase. More work is needed before any runtime API proposal.

The next stage would replay heterogeneous saved groups through the full sink,
logging and batch transport, with lifecycle and bookkeeping comparisons, before
considering an API change. This draft deliberately makes no deployment claim.

## Initial measurements

Three repeats on a shared Linux devbox, Python 3.12.14. These are small synthetic
CPU cases, not a prediction for production throughput. The complete records,
versions, source hashes and limitations are in `results.json`.

Median wall seconds for eight independent episodes:

| Consumed | Native receiver | Typed views receiver | Deferred receiver | Deferred extra producer work | Deferred accounted total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.076 | 0.063 | 0.001 | 0.032 | 0.034 |
| 25% | 0.147 | 0.132 | 0.080 | 0.033 | 0.113 |
| 100% | 0.330 | 0.285 | 0.293 | 0.033 | 0.326 |

At 100% consumption, deferral alone is effectively tied with native decoding
once producer work is counted. Typed views reduce receiver time by approximately
14% in this case. The typed-bytes decoder is slower than native in these runs.
At 25% consumption, deferred accounted work is approximately 23% lower, subject
to the admission assumption above. These differences have no confidence bounds.

After-materialization median PSS at 100% consumption was approximately 297 MiB
for native, 355 MiB for typed views and 318 MiB for deferred views. Boundary memory
includes interpreter/dependency overhead and allocator history. It does not
justify claiming a memory improvement from binary views.

All 36 cases passed complete native sample-byte, accepted-episode and numeric
metric comparisons. The existing #3806 metrics and sink tests passed (22 tests).
Ruff and the repository's pre-commit checks passed for the new benchmark files.

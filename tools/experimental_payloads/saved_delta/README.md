# Saved-delta comparison

This directory replays the private saved-delta workload used by the previous
complete-native-episode benchmark. Samples, task prompts, answers and raw graphs
are excluded from the branch. Runtime implementations and live training are unchanged.

The workload has 188 saved frames, approximately 388 MB of encoded deltas, and 337
final graph nodes. The full matrix uses 32 independent replays and two heap cycles;
one quarter of returned episodes stays buffered across cycles. The original wire
bytes were not captured: the earlier fixture restored tensor bytes from saved
logger representations and repacked the deltas. Every case checks those same
frame hashes and the same native episode fingerprint.

## Experiments

`replay.py` compares one decode thread, typed decoding with owned bytes, typed
decoding with binary views, and worker-owned graphs with deferred SHM export.
The deferred path emits scalar envelopes for every episode, exports selected
graphs, reconstructs complete native episodes in the parent, and releases all
worker graphs. Worker encoding, parent reconstruction, summary handoff and owner
release are included in receive time. It uses a predetermined selection fraction,
not a production admission decision. The summary is an experimental count record,
not the full Prime RL metrics cache.

`exact_harness.py` additionally runs the previous `contract_bench_v3.py` harness
unchanged. Typed cases replace its decoder and permit read-only array backing views;
native return, callbacks, verification, queue bounds, buffering and release stay
on the original path. Thread, full process IPC, per-delta SHM and final-episode SHM
controls are rerun on the same allocation. Comparing these cases prevents harness
changes from being mistaken for decoder improvements.

Two extra controls add `--zero-copy-receive`: native decoding and typed views,
both without callbacks. They receive a ZMQ Frame buffer instead of copying it to
Python bytes first. The readonly view retains its Frame owner. This distinguishes
receive-side copying from decoded tensor copies; neither removes native object
construction. Automatic GC remains enabled in every case.

The original benchmark source and matching archived Verifiers package are supplied
privately. Their hashes are recorded in results. Original retry fields, `dispatch`,
and string routing-repair indices are retained. The typed Struct decoder changes
raw dictionary insertion order; native model values and public field types still
must match. For callbacks, binary views are copied to owned bytes before assembly
and observation. Thus the callback-compatible path gives up binary-copy avoidance.

`check_contract.py` runs the unchanged native `EnvClient.run()` and validation
against an offline byte source. It checks callback values, mutable-info aliasing,
callback exceptions, complete episode fingerprints and mutable native lists for
native, typed-byte and typed-view-with-owned-callback decoding. This substitutes
`_request`; it does not test a live receive loop or cancellation/timeout APIs.

## Run

Use the matching archived Verifiers package and CPU dependencies. The measured
environment pins MessagePack 1.1.2 and msgspec 0.21.1; records include the other
versions and native source hashes. Do not silently adapt the fixture to a newer
schema by deleting fields.

```bash
export PYTHONPATH=/private/matching-native-source:/private/pinned-cpu-dependencies
uv run --no-project --python /path/to/matching/python \
  tools/experimental_payloads/saved_delta/replay.py \
  /private/saved-fixture /private/new-result \
  --mode typed-views --callbacks --replicas 32 --cycles 2

uv run --no-project --python /path/to/matching/python \
  tools/experimental_payloads/saved_delta/exact_harness.py \
  /private/contract_bench_v3.py /private/isolated-harness-root \
  --mode typed-views --callbacks --replicas 32 --cycles 2 --name comparison

uv run --no-project --python /path/to/matching/python \
  tools/experimental_payloads/saved_delta/check_contract.py \
  /private/saved-fixture /private/contract-checks.json
```

The fixture root contains `fixture.json` and `frames/`. The isolated original
harness root also contains its `sender.py`, helper modules and `native-source`.
Read-only links to private inputs are sufficient. The three Slurm scripts record
the exact India allocation and sequential commands used for this experiment.
No GPUs are requested.

## What earlier implementations had that the first microbenchmark omitted

| Detail | Saved-delta experiment | Remaining integration work |
| --- | --- | --- |
| Incremental decode and ordered graph application | Same saved frames, per-request lanes, reply after deltas | Real network and reconnect behavior |
| Complete native output | Every episode for native controls; selected episodes for deferred | Production sample construction and shipping for deferred |
| Public callback values and aliasing | Owned binary fields, native run controls | Preserve dictionary order if callers depend on it |
| Bounded queues | 2 GiB / 64 items, ZMQ HWM 8; one outstanding input per SHM lane | Production dispatch cancellation and backpressure |
| Shared-buffer ownership | Input reused after worker completion; output decoded to owned storage before close/unlink | Safe cancellation during SHM materialization |
| Buffered references and heap cleanup | Two cycles, 25% retained; historical GC/malloc_trim policy | Long-running lifetimes and production release paths |
| Accounting for shifted work | Encoding/reconstruction/release in receive; process-tree memory observer | Full scoring, callbacks and admission before deferred selection |

Images remain encoded message content at this boundary. These runs do not decode
image pixels. Routing fields return native NumPy arrays; token, mask and logprob
fields remain native lists. Input views pin their entire immutable frame, so less
binary copying does not necessarily mean lower memory.

Memory uses both parent high-water RSS and independently sampled receiver-tree
PSS/RSS, excluding the fake sender. Tree RSS double-counts shared mappings; PSS
apportions them. Maxima include setup. Receive lag and whole-workflow lag are
reported separately because equality verification itself can hold the GIL.

Two heap cycles are not independent repetitions or confidence bounds. Deferred
25% returns fewer episodes and assumes an admission oracle: it must not be ranked
as an equivalent replacement for complete native return. No production API change,
sampler change, deployment or restart is proposed by this experiment.

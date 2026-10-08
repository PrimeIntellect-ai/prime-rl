# Saved-delta payload experiment

With copying ZMQ receive, the real-data replay does not reproduce the synthetic binary-view speedup. Two additional controls show a larger improvement from zero-copy receive itself; adding typed views then gives a small measured gain with higher memory. Deferred materialization reduces receiver work only when most completed graphs are deliberately not returned; this changes admission and return semantics.

This report accompanies experimental draft [Prime RL #3940](https://github.com/PrimeIntellect-ai/prime-rl/pull/3940). It changes no runtime code or live training. Only benchmark code and sanitized numerical results are committed.

## Matched saved-delta replay

India node `ltc-idc3-hgx8-h200-25`, 16 allocated CPU cores, no GPUs requested. One saved episode supplies 188 delta frames and 337 graph nodes. Each full case independently replays it 32 times across two heap cycles. All input hashes match the earlier workload; returned complete native episode fingerprints match exactly. One quarter of returned episodes stays buffered across cycles.

| Path | Callbacks | Native episodes returned per cycle | Receive + native return | Worst workflow loop pause | Peak receiver-tree PSS |
| --- | --- | ---: | ---: | ---: | ---: |
| thread | Off | 32 | 30.33 s | 0.94 s | 16.10 GiB |
| thread | On | 32 | 31.40 s | 0.83 s | 15.82 GiB |
| typed-bytes | On | 32 | 30.89 s | 1.39 s | 16.34 GiB |
| typed-views | Off | 32 | 32.93 s | 1.61 s | 18.77 GiB |
| typed-views | On | 32 | 30.90 s | 0.76 s | 16.32 GiB |
| deferred | Off | 32 | 50.82 s | 0.87 s | 28.32 GiB |
| deferred (25% selected) | Off | 8 | 23.93 s | 1.07 s | 17.43 GiB |

Receive time includes input delivery, decoding/application, output encoding where needed, native reconstruction, and temporary/worker graph release. It excludes the separate whole-episode fingerprint verification phase. Whole-workflow maximum lag includes that verification and final cleanup. Memory peaks include setup and are independently sampled every 250 ms across receiver processes; the fake sender is excluded. RSS and PSS are separate, non-atomic estimates; small cross-snapshot differences are possible during rapid allocation.

At 25% consumption, deferred receive time is about 21% lower than the matched native no-callback control (23.93 versus 30.33 seconds). It returns eight native episodes rather than 32 and assumes selection decisions already exist. At 100%, it is about 68% slower (50.82 seconds) and raises receiver-tree peak PSS from 16.10 to 28.32 GiB. Graph ownership alone does not remove full output handoff cost.

Binary views avoid copies of tensor binary fields but pin entire encoded input frames. Token lists, message strings and native model reconstruction still allocate objects. Without callbacks, the measured view path is about 9% slower and uses about 17% more peak PSS than native. With callbacks, views are converted back to bytes to preserve native callback values/types, and there is no clear speedup.

The synthetic 14% view improvement and 23% deferred accounted-work reduction remain separate small-workload observations. They should not be substituted for these full saved-delta measurements.

## Rerun of the previous complete-native-return harness

The following controls use the original `contract_bench_v3.py` unchanged on India
node `ltc-idc3-hgx8-h200-30`, with the same 16 CPU / 160 GiB allocation, input,
32 replicas and two cycles. The adapter changes decoding only. Compare rows within
this allocation; historical 26.27 / 27.12 / 55.39-second measurements ran earlier,
and timings from the separate replay harness above cover a different lifecycle.

| Path | Receive | Callbacks | Receive + native return | Worst workflow loop pause | Peak receiver-tree PSS |
| --- | --- | --- | ---: | ---: | ---: |
| Native thread | Copy | Off | 33.48 s | 1.14 s | 16.38 GiB |
| Native thread | Copy | On | 32.82 s | 0.90 s | 16.45 GiB |
| Typed bytes | Copy | On | 31.97 s | 1.34 s | 16.23 GiB |
| Typed views | Copy | Off | 35.44 s | 1.25 s | 19.12 GiB |
| Typed views | Copy | On | 34.82 s | 0.92 s | 16.44 GiB |
| Per-delta process pickle | Copy | On | 67.07 s | 1.46 s | 17.66 GiB |
| Per-delta shared memory | Copy | On | 43.51 s | 2.17 s | 18.15 GiB |
| Graph owner, final shared output | Copy | Off | 51.56 s | 1.09 s | 29.50 GiB |
| Native thread | Zero-copy | Off | 23.73 s | 0.82 s | 16.35 GiB |
| Typed views | Zero-copy | Off | 22.46 s | 0.71 s | 19.14 GiB |

With copying receive, the original harness likewise finds no binary-view improvement. Shared memory
reduces the process IPC penalty versus pickle, but returning complete native
episodes still costs more than the thread path here. Graph-owner ACK benchmarks
do not account for that full-output cost.

The earlier implementations also offered optional zero-copy ZMQ receive. The
first comparison above had omitted that flag. The two added controls use it in
the same original harness and on the same node, with all episode fingerprints
and native field types preserved. Receive buffers are made readonly and retain
their ZMQ Frame owner. Native decoding still creates owned bytes for binary
fields, keeping the returned native contract unchanged.

Native zero-copy receive is about 29% faster than the copying-receive control
here. Typed views on that receive path are about 5% faster than native but use
about 17% more peak PSS. Two cycles are insufficient to establish the small
incremental view advantage. Neither zero-copy control runs callbacks, so callback
performance for this receive mode remains unmeasured. The final two controls ran
in a subsequent allocation on the same node; longer repetitions and order
rotation are needed to separate run-to-run variation from effect size.

## Native callback and ownership controls

Six native `EnvClient.run()` controls passed: native/typed-byte/typed-view decoding, each with and without a deliberately mutating and throwing callback. All 188 callback values match; callback info mutation affects the same retained graph; final native episode fingerprints match. The offline byte-source wrapper catches the same deliberately raised callback exception in each mode; this does not certify production RPC exception handling. The no-fault fingerprint is `63cc33538ea62aca108eb98396c4c3311c1797aef712f0b13d12444a27217fe8`.

All seven smoke cases, seven full replay cases and ten original-harness controls passed their native episode fingerprint/type checks. The seven full replay cases also verify experimental summaries. Deferred worker graph tables are empty at completion. Input SHM slots are reused only after decode/application finishes; output SHM is decoded into owned storage before close/unlink.

Typed Struct decoding does not preserve original raw dictionary insertion order. Callback data values and field types match, but callers depending on map order need additional handling. Safe cancellation during materialization, live RPC/reconnect behavior, production callbacks and full Prime RL admission/shipping are not certified by these controls.

## Differences from earlier experiments

| Earlier experiment | What it measured | What it did not establish |
| --- | --- | --- |
| Synthetic completed-episode microbenchmark | Native validation, #3806 metric extraction and actual training sample encoding on eight small synthetic episodes | Saved streaming costs, real callback/IPC/lifecycle performance |
| Earlier real raw-delta view benchmark | Saved-frame decode/application into raw graphs; exact raw-value comparisons | Complete `WireEpisode` return, native callback compatibility, process-tree PSS |
| Earlier graph-owning process benchmark | Decode/apply in owners and compact acknowledgements | Cost of returning actual complete native outputs |
| Original full-native episode benchmark | Full return, callbacks, SHM handoff and cleanup | Deferred admission or typed-view integration |

The earlier view code explicitly documented frame pinning and unsafe slot reuse. Its latest v3 already retained `dispatch` and string repair keys. Our initial saved adapter missed those two schema fixes; strict replay checks caught them and the final decoder retains both unchanged. An initial packed-byte comparison also incorrectly treated map insertion order as a semantic value mismatch; final checks distinguish those properties. Failed prototype attempts are excluded from performance tables. Optional zero-copy receive is now included in the original-harness controls; earlier automatic-GC-disabled variants are not rerun here, and automatic GC stays enabled throughout.

## Integration needed before any API proposal

A `CompletedRollout(summary, payload)` handle does not make materialization free. Production admission currently needs finalized scoring, callbacks, graph statistics and group selection. This prototype uses predetermined selection and an experimental count summary, not a complete #3806 metrics envelope. It still consumes every incoming delta; producer-side elimination of rejected wire payloads is not measured.

Before introducing deferred results, preserve complete metric/annotation observations, rewards and custom metrics; error/cancellation state; group/sampler/checkpoint bookkeeping; accepted training sample bytes; and owner lifetime/backpressure. A cancellation-safe release protocol and optional callback semantics are still required. There is no production transport or public API modification in this PR.

Shared memory here connects local receiver workers and their parent on one host.
It cannot replace network transport from environment workers on other hosts;
those deltas still arrive over ZMQ before local ownership/export decisions.

Two heap cycles are not independent repetitions and provide no confidence intervals. The workload repeats one saved episode, and its missing original episode head is reconstructed identically for all variants. It is not a representative survey of all production trajectories or proof of the cause of live stalls.

## Evidence

- [Replay measurements](saved-delta-results.json) and [CSV](saved-delta-results.csv)
- [Original-harness measurements](exact-harness-results.json) and [CSV](exact-harness-results.csv)
- [Native callback controls](contract-checks.json)
- [Replay code](replay.py), [original-harness adapter](exact_harness.py), and [run instructions](README.md)

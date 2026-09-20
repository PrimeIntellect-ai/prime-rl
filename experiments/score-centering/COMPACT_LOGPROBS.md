# Compact logprob transport

The renderer requests compact logprobs when `PRL_SCORE_TOPK` is nonzero.
Set `PRL_COMPACT_LOGPROBS=0` to request the nested response for comparison.
Both experiment arms use the same transport. No training jobs were started.
The API worker count remains unchanged pending a live serving comparison.

The request sets `sampling_params.extra_args.prl_compact_logprobs=true`.
The PRL handler enables vLLM's `flat_logprobs` before engine submission.
It copies the response formatter's sampling parameters and disables nested logprob construction there.
The engine's sampling parameters keep their requested logprob count.
Streaming requests cannot use this format.

Each response choice has `logprobs=null` and a `compact_logprobs` object:

- `format`: `u32-f32-le-v1`.
- `data`: one base64 string. All uint32 token IDs precede all float32 logprobs, in little-endian order.
- `offsets`: candidate offsets for each generated token, with a final end offset.
- `num_top_logprobs`: the requested candidate count.

The client removes base64 fields before JSON parsing, then decodes numeric buffers.
It resolves repeated sampled tokens in the same order as vLLM's dictionary representation.
It preserves the sampled logprob separately from the selected top-k head.
It preserves the existing candidate limit, stable sorting, and logprob clamp.
It rejects malformed buffers, missing sampled evidence, and nonfinite scores.
Older servers can return the nested response, which the client still accepts.
Routed experts, sampling masks, usage, and KV transfer metadata keep their existing forms.

## CPU benchmark

Command:

```bash
uv run --no-sync python experiments/score-centering/benchmark_compact_logprobs.py experiments/score-centering/results/api-diagnostic/compact-logprobs.json
```

The benchmark uses 4,096 generated tokens and a top-128 score head.
It runs the actual full-response formatter, JSONResponse encoder, and client parsers.
The response also includes synthetic routed experts and sampling masks.
Half the sampled tokens fall outside the candidate head.
The table reports medians of three repetitions on the development host.

| Measurement | Nested | Compact |
| --- | ---: | ---: |
| Response formatting | 5.488 s | 0.0369 s |
| JSON encoding | 0.699 s | 0.0253 s |
| Total server response work | 6.189 s | 0.0622 s |
| Client decoding and score extraction | 0.540 s | 0.111 s |
| Total measured CPU path | 6.730 s | 0.176 s |
| Response size | 34.91 MB | 5.76 MB |

The sampled logprobs and top-k IDs/logprobs match exactly in every repetition.
The formatter preserves usage, sampling masks, routed experts, and KV transfer metadata.
The benchmark also checks that formatting does not mutate the engine's sampling parameters.

These measurements exclude GPU generation, engine output processing, network transport, and live concurrency.
They do not establish the required API worker count or training throughput.
Live one-worker versus four-worker measurements remain pending.

## Verification

- Serving and codec tests: 21 passed.
- Renderer client tests: 32 passed.
- Ruff checks and `git diff --check` passed.
- The archived renderer patch reconstructs both modified files exactly from the pinned submodule.

```bash
uv run --no-sync pytest tests/unit/inference/test_serving_tokens.py -q
uv run --no-sync pytest --noconftest deps/renderers/tests/test_client.py -q
```

The client module needs no repository fixtures.
`--noconftest` avoids a collision between the two repositories' `tests` packages in this editable installation.

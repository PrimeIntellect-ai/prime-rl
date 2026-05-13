# GPT-OSS LoRA NaN Repro Handover

Last updated: 2026-05-14

## Current Branch State

This document was written on `prime-rl` branch:

```text
daniel/gptoss-lora-nan-repro @ f65f948b8
```

The branch contains investigation and diagnostic commits on top of the previous
`origin/main`:

- `99ad49296` - applied the vLLM layerwise alias-buffer reload patch.
- `ef5d091a6` - instrumented the LoRA adapter reload path.
- `48a56c5f8` - fixed a logger import in the LoRA diagnostics.
- `751faf3d0` - dumped non-finite TITO requests without middleware.
- `f65f948b8` - normalized missing TITO tool parameter descriptions.

Current upstream `origin/main` is:

```text
b2ba40b5e Patch vLLM layerwise reload alias-buffer handling (#2482)
```

That upstream commit already contains the layerwise alias-buffer fix matching
Jackmin's simpler PR #2486 behavior for `src/prime_rl/inference/patches.py`.
The new clean repro branch should therefore start from `origin/main` and should
not carry the diagnostic commits above unless a fresh repro needs them.

## Relevant Hosted Evidence

Hosted issues read locally under `/tmp/hosted-rl-issues`:

- `#1699`: `openai/gpt-oss-20b`, `zahlmann/minesweeper-agent`, stalled at step 16.
- `#1703`: `openai/gpt-oss-20b`, `primeintellect/wiki-search`, stalled at step 8.
- `#1152`, `#1158`, `#1210`, `#1716`: Nemotron NaN JSON/liveness/router crash-loop family.

The GPT-OSS hosted failures are the relevant LoRA target. They are operationally
different from the Nemotron crash-loop family:

- GPT-OSS: hard rollout stall with repeated HTTP 400 responses, pods can remain healthy.
- Nemotron: inference server/router health degradation and liveness restarts.

For `hfa85vr6qahwuk6k5u6of167` (`primeintellect/wiki-search`), the hosted
config from the API was:

```toml
model = "openai/gpt-oss-20b"
max_steps = 50
batch_size = 32
rollouts_per_example = 4

[sampling]
max_tokens = 1024

[sampling.extra_body]
chat_template_kwargs = { reasoning_effort = "low" }

[[env]]
id = "primeintellect/wiki-search"
```

Additional API metadata:

- `seq_len = 65536`
- env version `0.1.23`
- progress froze at `latest_step = 8`
- `last_updated_at = 2026-05-08T20:13:12.048650+00:00`

Pulled artifacts:

```text
/tmp/hosted-run-logs/hfa85vr6qahwuk6k5u6of167/
```

The retained env-server log only covers the final minute before the run was
stopped, but it is a clean failure slice:

- window: `2026-05-11T17:38:00Z` to `2026-05-11T17:39:01Z`
- active env tasks: 128 (`W0..W3 = 32 each`)
- 668 requests to the gpt-oss router returned `400 Bad Request`
- 668 matching rollout aborts had:
  `Out of range float values are not JSON compliant: nan`
- env-side OpenAI calls were still `200 OK`

Rollout samples were pulled for completed steps:

```text
/tmp/hosted-run-logs/hfa85vr6qahwuk6k5u6of167/rollouts/step-{0,1,2,3,4,5,7,8}.json
```

Those are successful samples, not the failing NaN requests.

## Local Repro State

Local PTFT config outside this submodule:

```text
/home/daniel/git/research-prod/repro/gptoss_lora_nan/gptoss20b_ptft_lora.toml
/home/daniel/git/research-prod/repro/gptoss_lora_nan/launch.sh
```

Important launch details:

- launcher sources `/home/daniel/git/research-prod/.env`
- verifies `HF_HOME=/beegfs/huggingface`
- last committed variant used MITO:
  `[orchestrator] use_token_client = false`

Observed local result:

- PTFT MITO run completed all 22 rollout steps without NaN dumps.
- One non-NaN `EmptyModelResponseError` was rescheduled.
- Therefore PTFT is currently a weak repro for the hosted GPT-OSS issue.

Earlier TITO PTFT attempts hit a GPT-OSS chat-template/tool-schema issue first:
some tool parameter schemas had a `title` but no `description`. The current
diagnostic branch contains a compatibility patch for that, but this is only an
unblocker and is not believed to be the LoRA NaN root cause.

## Current Hypotheses

Most relevant to GPT-OSS LoRA:

1. Repeated LoRA reload/update may poison or fail to refresh vLLM worker-side
   adapter state, especially when the same adapter id is reused.
2. The GPT-OSS MoE LoRA path may have a model-specific adapter format or worker
   cache issue not covered by the Nemotron layerwise alias-buffer fix.
3. The hosted issue may be router/response containment around non-finite values:
   the user reported inference logs looked healthy for both replicas, while the
   env-server definitely received 400s from the router.

Less useful for the immediate GPT-OSS LoRA hunt:

- Nemotron Mode B is probably easier to reproduce as a serving stress test, but
  it is a different operational failure mode.
- PTFT did not reproduce locally; continuing to iterate only on PTFT is likely
  lower signal than matching the hosted `wiki-search` workload.

## Recommended Next Plan

Start from a clean `prime-rl` branch containing only current upstream main, which
already has the Jackmin/layerwise alias-buffer fix. Do not carry the old
diagnostic commits until needed.

1. Reproduce the hosted `hfa85vr6` workload locally first:
   - `openai/gpt-oss-20b`
   - `primeintellect/wiki-search`
   - `batch_size = 32`
   - `rollouts_per_example = 4`
   - `max_tokens = 1024`
   - `chat_template_kwargs = { reasoning_effort = "low" }`
   - `max_steps >= 12`
2. Run with the production default orchestrator path first.
3. If startup/tool-schema issues block TITO, switch to MITO only as an unblocker
   and record that the path differs from hosted/default.
4. If NaNs reproduce, add the minimum diagnostics:
   - dump the exact failing request body and response boundary near JSON serialization
   - log `/load_lora_adapter` path/name/id/load_inplace metadata
   - log whether the worker actually reloads or only touches an already-loaded adapter
5. If NaNs do not reproduce, ask for/pull router logs for hosted windows:
   - onset: `2026-05-08T20:10:30Z` to `2026-05-08T20:14:00Z`
   - final retained env failure slice: `2026-05-11T17:38:00Z` to `2026-05-11T17:39:01Z`

Useful hosted log commands:

```bash
unset PRIME_API_KEY
PRIME_DISABLE_VERSION_CHECK=1 prime train get hfa85vr6qahwuk6k5u6of167 --plain -o json
PRIME_DISABLE_VERSION_CHECK=1 prime train progress hfa85vr6qahwuk6k5u6of167 --plain
PRIME_DISABLE_VERSION_CHECK=1 prime train metrics hfa85vr6qahwuk6k5u6of167 --plain -n 2000
PRIME_DISABLE_VERSION_CHECK=1 prime train logs hfa85vr6qahwuk6k5u6of167 --plain -c env-server --env wiki-search -n 20000 -r
PRIME_DISABLE_VERSION_CHECK=1 prime train rollouts hfa85vr6qahwuk6k5u6of167 --plain --step 8 -n 100
```

Useful grep:

```bash
rg -n "Out of range float values|BadRequestError|nan|NaN|load_lora_adapter|LoRA|adapter|400 Bad Request" <log-dir>
```

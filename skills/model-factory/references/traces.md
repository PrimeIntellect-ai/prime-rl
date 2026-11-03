# Retrieve Model Factory run traces

Uploaded traces are stored by Prime Traces and remain available after training
pods terminate. Use the same personal/team workspace as the run and a local
`PRIME_API_KEY` with Prime Traces read permissions. `PRIME_TEAM_ID` or saved CLI
team settings select the workspace. See the
[Traces overview](https://docs.primeintellect.ai/traces/overview).

## Inspect and download

Use the Model Factory run ID initially. Choose a local `TRACE_DIR` for artifacts:

```bash
mkdir -p "$TRACE_DIR"
prime traces get "$RUN_ID" --runs --plain -o json
prime traces list --run-id "$RUN_ID" --limit 100 --plain -o json
```

The list contains summaries in `.items`, each with a `trace_id`. Download the
selected raw record to a file rather than printing large or private conversations:

```bash
prime traces get "$TRACE_ID" --raw --dest "$TRACE_DIR/$TRACE_ID.json" --plain
```

`--raw` retrieves the stored trace document; a summary is not the full trace.
For the [SDK alternative](https://docs.primeintellect.ai/traces/managing-traces#download-a-trace),
use `TracesClient.download_raw(trace_id, destination)` from a `uv run` script.

For a whole run, iterate the list response's `.items[].trace_id`, download each
raw record, and follow `.next_cursor` with the same filters until it is empty:

```bash
prime traces list --run-id "$RUN_ID" --limit 100 \
  --cursor "$NEXT_CURSOR" --plain -o json
```

Use cursor pagination rather than repeatedly selecting numbered pages while new
records arrive. There is no working SDK bulk-export operation documented; build
the export from paginated summaries and individual downloads. Record exported
IDs, count, and destination. For a final export, wait for uploads to drain and
compare against the run summary's `trace_count`; reconcile new or missing IDs
before claiming completeness. Keep any failed downloads visible in the report.

## Training steps and evaluations

To find episodes from one training step and their member traces:

```bash
prime traces list --episodes --run-id "$RUN_ID" --run-step "$STEP" --plain -o json
prime traces list --episode-id "$EPISODE_ID" --plain -o json
prime traces get "$EPISODE_ID" --episodes --raw \
  --dest "$TRACE_DIR/$EPISODE_ID.episode.json" --plain
```

Follow pagination for both episode and member-trace lists. Download each member
trace separately; an episode record groups traces and is not their raw content.

Online evals may have their own evaluation IDs. The
[Prime monitor](../../../src/prime_rl/monitors/prime.py) logs each environment/step's
evaluation URL; use that ID as `--run-id` for its traces instead of assuming they
all share the training ID. Start with the run dashboard and evaluation links or
the orchestrator/eval logs to identify the right evaluation.

The training CLI also exposes a per-step sample view:

```bash
prime train rollouts "$RUN_ID" --step "$STEP" --page 1 --num 100 --plain
```

Its JSON includes `samples`, `total`, and `total_pages`; request remaining pages
when needed. Use `prime train progress` to discover populated steps. Do not treat
one sample page as a full raw trace export. For older runs backed by legacy sample
storage, consult the [archive export documentation](https://docs.primeintellect.ai/traces/legacy-storage)
and its November 6, 2026 retirement date.

## Missing traces

Check workspace, run/evaluation ID, upload completion/errors, and the configured
[upload controls](https://docs.primeintellect.ai/traces/pricing-and-settings).
Account-level opt-out can disable SDK/CLI reads while stored traces remain visible
in the dashboard. A completed training run does not prove all traces uploaded.
Trainer-only SFT produces no rollout traces; the Model Factory SFT docs also currently
list online-eval trace uploads as unavailable, so verify runtime support before
promising those artifacts.

`prime traces search "Traceback" --run-id "$RUN_ID" --plain` can locate failures,
but search coverage may lag ingestion. An empty search result is not evidence
that the raw records are absent. See
[retrieval troubleshooting](https://docs.primeintellect.ai/traces/managing-traces#troubleshooting).

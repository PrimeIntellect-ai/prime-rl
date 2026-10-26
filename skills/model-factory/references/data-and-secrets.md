# Private datasets and secrets

## Local credentials and remote credentials

Load secrets from the user's chosen environment or secret manager. Check presence
without printing values. For example, once the required credentials are loaded:

```bash
: "${HF_TOKEN:?Load an HF token with access to the requested repository}"
export HF_TOKEN
prime train run "$CONFIG_PATH" -e HF_TOKEN --yes --plain -o json
```

`-e KEY` reads the exported value; exporting alone does not forward it to Model Factory
pods. `--env-file "$MODEL_FACTORY_ENV_FILE"` is an alternative for a file containing
only intended remote secrets. Do not forward a general-purpose `.env` wholesale.
Avoid inline `KEY=secret` arguments and shell tracing. Keep secret files outside
version control with restrictive permissions. Configured local Prime/HF login
state is not automatically copied to the remote job.

| Credential | Purpose |
| --- | --- |
| Local `PRIME_API_KEY` | Authenticates CLI/API requests; keep on the launching side. |
| Local `PRIME_TEAM_ID` | Selects the run/volume owner; configure before dispatch. |
| Remote `PRIME_API_KEY` | Generated and injected by the platform for the run; do not override with a personal key. |
| `HF_TOKEN` | Pass with `-e HF_TOKEN` when remote model/dataset downloads require HF access. |
| `WANDB_API_KEY` | Pass with `-e WANDB_API_KEY` when W&B monitoring is requested. |
| Environment-specific keys | Forward only required names when the CLI and backend support them. |

### Check secret-forwarding support

The [Model Factory RL docs](https://docs.primeintellect.ai/hosted-training/hosted-rl#launching-a-run)
describe arbitrary secrets reaching trainer, inference, orchestrator, and
env-server processes. However, the CLI 0.14.0 implementation and public CLI source
checked on 2026-10-23 still restrict dedicated runs to `WANDB_API_KEY` and
`HF_TOKEN`. Check the installed version and
[secret collection code](https://github.com/PrimeIntellect-ai/prime/blob/main/packages/prime/src/prime_cli/commands/rl.py)
before relying on a third-party key. Upgrade to a release with the required
support if available; otherwise report the blocker. Do not omit a required key,
rename it as an unrelated credential, or put it in the config to bypass rejection.

`PRIME_API_KEY` is reserved by the service. The run's token is supplied by the
platform, including for Prime services such as sandbox creation. Secrets projected
into a training/env-server process do not automatically enter a child sandbox:
if sandbox code itself needs a key, configure that runtime's supported secret
injection. See [sandbox secrets](https://docs.primeintellect.ai/sandboxes/secrets).

## Private Hugging Face datasets for RL

Use an HF token with read access to the requested repository, including any gated
dataset approval. Set the dataset ID, subset, split, and revision only through
fields the selected taskset supports. A metric source `name` is not a dataset
selector. Inspect the taskset config and loader if the mapping is unclear.

Verify access with the same token before reserving GPUs, without displaying
private samples. Pass `-e HF_TOKEN` to the Model Factory launch and ensure the loader uses
it. If the taskset downloads in a sandbox rather than the env server, deliver the
credential there explicitly. Custom loaders/scorers must be installed remotely;
see [custom code](custom-code.md).

Alternatively, stage data on a volume and set the taskset's supported local-path
field to its remote mount path. Verify that the process reading it actually has
the mount; a training volume is not automatically mounted in rollout sandboxes.

## Private Hugging Face datasets for SFT

Start from [examples/basic/reverse-text/sft.toml](../../../examples/basic/reverse-text/sft.toml)
in the selected revision. Model Factory SFT requires real datasets on a named volume;
`HF_TOKEN` alone does not turn a Hub ID in `[data].name` into a volume dataset.
Use HF `load_dataset`-readable Parquet/JSON with `messages` or
`prompt`/`completion` columns, not a `save_to_disk` directory.

Choose `DATASET_ID`, `DATASET_REVISION`, `DATASET_DIR` (an isolated local staging
directory), `VOLUME_NAME`, and `VOLUME_SIZE` for this task. With `HF_TOKEN` already
exported, download and upload through the public CLI paths:

```bash
uv run --no-sync hf download "$DATASET_ID" --repo-type dataset \
  --revision "$DATASET_REVISION" --local-dir "$DATASET_DIR"
prime volumes create "$VOLUME_NAME" --size "$VOLUME_SIZE"
prime volumes put "$VOLUME_NAME" "$DATASET_DIR/" datasets/training-data
```

Reuse an existing appropriate volume instead of recreating it. These commands
upload dataset files, not the HF credential. Check the staging directory contains
only the intended files. For HF authentication and download options, see the
[HF CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli).

In the copied SFT config, set `[data].name = "datasets/training-data"` (or
`"/volume/datasets/training-data"`), `type = "sft"`, and the appropriate splits /
subsets / column mapping. Verify loading in a fresh process with the same dataset
library and options, without logging private rows. Pin and record the download
revision; a local dataset path does not fetch `[data].revision` from the Hub.

The repository's basic SFT example includes `[eval]` and `[inference]`. The
[Model Factory SFT docs](https://docs.primeintellect.ai/hosted-training/hosted-sft#online-evals)
require chart 0.24.0+ for online eval. Check the CLI too: some versions reject those
blocks. Preserve requested online eval and resolve missing support; for an intended
trainer-only run, remove both blocks and their nested tables from the copied
example and size the trainer deployment accordingly. Keep the model renderer
consistent with any model substitution.

```bash
prime train run "$CONFIG_PATH" --sft --volume "$VOLUME_NAME" \
  --gpu-type "$GPU_TYPE" --yes --plain -o json
```

Add the remote secrets/source flags the job needs. For cluster-side dataset
downloads, `prime volumes ssh "$VOLUME_NAME" --read-write` is an alternative
requiring eligible cluster access and the account's primary SSH key. Local
`HF_TOKEN` is not automatically forwarded over SSH; supply it securely inside the
session. Download under `/volume/datasets/`, then explicitly end the session with
`prime volumes stop "$VOLUME_NAME" "$SESSION_ID"` and wait for final sync.
Exiting the SSH shell alone can leave a writer lock that blocks dispatch.

Use `prime volumes put/get` when no SSH allocation is available. Wait for a run's
final volume sync before retrieving `runs/<runId>/checkpoints/`. See
[volumes](https://docs.primeintellect.ai/hosted-training/volumes) for transfer and
session behavior.

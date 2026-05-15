# Dynamo SLURM Smoke Test

This example runs a single-node smoke test with Dynamo serving inference and
prime-rl training against that Dynamo endpoint.

It is intended to mirror the local/Kubernetes Dynamo smoke flow in
`tools/dynamo/` and `k8s/dynamo-deploy/`, but under one SLURM allocation:

- GPU 0: Dynamo frontend + vLLM worker, using `--discovery-backend file`
- GPU 1: prime-rl orchestrator + trainer

`--discovery-backend file` keeps the smoke test self-contained on one node and
does not require etcd or NATS.

## Prerequisites

- prime-rl is cloned on a filesystem visible to the SLURM compute node.
- The prime-rl virtualenv exists at `<prime-rl>/.venv`.
- Dynamo with vLLM support is installed either in the same virtualenv or in a
  separate virtualenv pointed to by `DYNAMO_VENV`.
- The SLURM node has at least two GPUs.

If Dynamo is installed separately:

```bash
export DYNAMO_VENV=/shared/dynamo/.venv
```

## Dry Run

Render the resolved config and sbatch script without submitting:

```bash
uv run rl @ tools/dynamo/slurm_smoke/smoke_rl.toml \
  --slurm.project-dir /shared/prime-rl \
  --slurm.partition <partition> \
  --dry-run
```

The generated script is written to:

```text
outputs/dynamo-slurm-smoke/rl.sbatch
```

## Submit

```bash
uv run rl @ tools/dynamo/slurm_smoke/smoke_rl.toml \
  --slurm.project-dir /shared/prime-rl \
  --slurm.partition <partition>
```

Useful overrides:

```bash
export DYNAMO_MODEL=PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT
export DYNAMO_SERVED_MODEL_NAME=$DYNAMO_MODEL
export DYNAMO_GPU=0
export PRIME_RL_GPU=1
export DYN_HTTP_PORT=8000
export DYNAMO_MAX_MODEL_LEN=2048
export DYNAMO_MAX_NUM_SEQS=32
```

Additional Dynamo worker flags can be passed with `DYNAMO_EXTRA_ARGS`.

## Logs

After submission, logs are under the run output directory:

```text
outputs/dynamo-slurm-smoke/job_<jobid>.log
outputs/dynamo-slurm-smoke/logs/dynamo/frontend.log
outputs/dynamo-slurm-smoke/logs/dynamo/vllm.log
outputs/dynamo-slurm-smoke/logs/orchestrator.log
outputs/dynamo-slurm-smoke/logs/trainer.log
```

The prime-rl config disables `use_token_client` and points both inference and
admin traffic at the Dynamo frontend:

```toml
[orchestrator.client]
base_url = ["http://127.0.0.1:8000/v1"]
admin_base_url = ["http://127.0.0.1:8000/v1/rl"]
```

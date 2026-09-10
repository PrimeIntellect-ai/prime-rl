# IFBench / IF-RLVR Training Runbook

This directory adds a minimal `prime-rl` training configuration for the IFBench / IF-RLVR bounty.

## Source material

- Bounty: IF-RLVR/Bench on Algora / Prime Intellect
- Ai2 code: https://github.com/allenai/IFBench
- IFBench data collection: https://huggingface.co/collections/allenai/ifbench-683f590687f61b512558cdf1
- IF-RLVR training data named by Ai2: `allenai/IF_multi_constraints_upto5`
- Prime Intellect Lab environment discovered publicly at: https://app.primeintellect.ai/dashboard/environments/shashwat23/ifbench

## Configuration

The training config is:

```text
configs/ifbench/rl.toml
```

It follows the same lightweight pattern as `configs/gsm8k/rl.toml` and `configs/hendrycks_math/rl.toml`, but uses the Prime Lab registry environment ID:

```toml
[[orchestrator.train.env]]
id = "shashwat23/ifbench"
name = "ifbench"
```

In `prime-rl`, slash-qualified environment IDs are installed before training via:

```bash
uv run --no-sync prime env install shashwat23/ifbench
```

## Run

From the repo root:

```bash
uv run prime-rl rl configs/ifbench/rl.toml
```

If the environment is not already installed locally, the orchestrator should install `shashwat23/ifbench` automatically before loading it.

## Validation done here

This contribution is intentionally small: it wires the IFBench environment into `prime-rl` as a train config/runbook. No heavy GPU training was run locally.

Lightweight checks:

```bash
python3 - <<'PY'
import tomllib, pathlib
tomllib.loads(pathlib.Path('configs/ifbench/rl.toml').read_text())
print('ok configs/ifbench/rl.toml')
PY
```

## Maintainer confirmation needed

Please confirm whether `shashwat23/ifbench` is the intended canonical environment ID for the bounty. If the final registry name differs, only the `id` in `configs/ifbench/rl.toml` should need changing.

# Development

This page covers workflows for developing on `prime-rl` itself — running the test suite, contributing changes, and adding new model architectures with the small-scale tooling we use to iterate on MoE families without booting up a 100B+ run.

## Table of Contents

- [Test Suite](#test-suite)
  - [Layout](#layout)
  - [Running Tests Locally](#running-tests-locally)
  - [CI Workflows](#ci-workflows)
  - [Markers](#markers)
- [Pre-Commit Hooks](#pre-commit-hooks)
- [Adding a New Model](#adding-a-new-model)
  - [Implement the Modeling Code](#implement-the-modeling-code)
  - [Register a Mini Preset](#register-a-mini-preset)
  - [Run the Smoke Test](#run-the-smoke-test)

## Test Suite

The test suite is split into three tiers, each with its own CI workflow.

### Layout

- **`tests/unit/`** — fast-running, hermetic tests for isolated logic: config parsing and validation, advantage / loss / scheduler / packer math, individual dataset paths, model-conversion roundtrips, etc. Tests that need a GPU are tagged with the `gpu` marker.
- **`tests/integration/`** — full-stack RL/SFT runs on a tiny model end-to-end through inference + orchestrator + trainer.
- **`tests/nightly/`** — runs the configs in [`examples/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples) every night to catch regressions in the shipped examples.

### Running Tests Locally

```bash
uv run pytest -v                                           # everything
uv run pytest tests/unit -v                                # unit only
uv run pytest tests/integration -v                         # integration only
uv run pytest -v -m "not gpu"                              # CPU-only subset (mirrors CPU CI)
uv run pytest -v -m gpu                                    # GPU-only subset
uv run pytest tests/integration/test_reverse_text.py -vvs  # one specific scenario
```

### CI Workflows

| Workflow | Trigger | What runs | Where |
|---|---|---|---|
| [`cpu_tests.yaml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/cpu_tests.yaml) | every PR + push to `main` | `pytest tests/unit -m "not gpu"`, plus a slim-wheel install check that `prime-rl-configs` imports cleanly without heavy deps (no torch / vllm / transformers / wandb / verifiers / datasets / liger / loguru in `sys.modules`) | `ubuntu-latest` |
| [`gpu_tests.yaml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/gpu_tests.yaml) | every non-draft PR + push to `main` | `pytest tests/unit -m gpu`, plus a matrix of named integration scenarios (`reverse_text`, `reverse_text_sft`, `reverse_text_lora`, `reverse_text_moe`, `reverse_text_multi_run`, `reverse_text_rl_opd`, `reverse_text_rl_sft`, `reverse_text_sft_lora`, `alphabet_sort`, `benchmark_regression`) | self-hosted GPU runners (`vm`, `4xa6000`) |
| [`nightly_tests.yaml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/nightly_tests.yaml) | 03:00 PST daily + manual `workflow_dispatch` (single-file filter optional) | every file in `tests/nightly/`, one matrix job per file | `research-cluster` |

The GPU + Nightly workflows skip drafts — open the PR as **Draft** until you're ready to consume CI compute, then mark it ready for review to trigger the GPU matrix.

### Markers

Two pytest markers are declared in `pyproject.toml` (`addopts = "--strict-markers"`):

- `gpu` — gate a test that needs CUDA. CPU CI uses `-m "not gpu"`; the GPU unit job uses `-m gpu`.
- `slow` — gate a test that's expensive enough you'd usually skip it locally. Deselect with `-m "not slow"`.

## Pre-Commit Hooks

Install the [pre-commit](https://pre-commit.com) hooks before your first commit so ruff check + format run on staged Python files automatically:

```bash
uv run pre-commit install
```

## Adding a New Model

Bringing up a new model family is three steps: implement the modeling code, register a mini preset, and run the smoke test. The preset and smoke test let you iterate on the modeling code at ~0.5B scale on 1–2 GPUs instead of paying the cost of the full-size model — useful for catching bugs in modeling code, state-dict conversions, and pipeline integration before scaling.

### Implement the Modeling Code

Drop the modeling code under `src/prime_rl/trainer/models/<arch>/` (HF-compatible config, modeling, and weight conversion). Mirror the layout of an existing family — `glm4_moe/` or `qwen3_moe/` are good starting points.

### Register a Mini Preset

Add an entry to [`scripts/mini_moe.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/scripts/mini_moe.py) so the smoke-test workflow can build a ~0.5B test model in your architecture. The preset names the config class, picks small dimensions, and wires up the HF + PrimeRL model classes plus a tokenizer source:

```python
ARCH_PRESETS = {
    "glm4_moe": {
        "config_class": Glm4MoeConfig,
        "config_kwargs": dict(hidden_size=1024, num_hidden_layers=24, n_routed_experts=8, ...),
        "hf_model_class": HFGlm4MoeForCausalLM,
        "prime_model_class": PrimeRLGlm4MoeForCausalLM,
        "tokenizer_source": "THUDM/GLM-4-9B-0414",
    },
    # add your arch here
}
```

### Run the Smoke Test

Build the mini model. This creates a ~543M-parameter GLM-4 MoE (1024 hidden, 24 layers, 8 experts) with random weights, copies the tokenizer from the original GLM-4 model, and verifies the HF↔PrimeRL roundtrip is lossless:

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe
```

To re-verify the roundtrip after a modeling-code change without re-creating the model:

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe --verify-only
```

Warm up the random-weight mini model with SFT on reverse-text so KL divergence becomes meaningful in the RL phase. Loss drops from ~12 to ~2.5 — the output won't be coherent, but the distribution is non-trivial. A pre-built SFT'd checkpoint lives at [samsja/mini-glm-moe](https://huggingface.co/samsja/mini-glm-moe) if you want to skip this step:

```bash
uv run sft @ configs/debug/moe/sft/train.toml \
  --model.name ./mini-glm-moe \
  --data.name PrimeIntellect/Reverse-Text-SFT \
  --data.type null \
  --max_steps 200 \
  --optim.lr 1e-4 \
  --ckpt.weights
```

Then run the full RL stack on reverse-text:

```bash
uv run rl @ configs/ci/integration/reverse_text_moe/start.toml \
  --model.name samsja/mini-glm-moe \
  --trainer.model.impl custom \
  --inference.gpu-memory-utilization 0.7 \
  --inference.model.max-model-len 2048
```

What to look for:

- **No crashes.** Validates the full inference + orchestrator + trainer pipeline end-to-end.
- **Finite, non-zero KL.** Confirms the reference distribution is meaningful.
- **Loss reasonable.** Not NaN, not stuck.

Don't expect reward to climb meaningfully in 20 steps on a random model.

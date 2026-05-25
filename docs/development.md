# Development

This page covers workflows for developing on `prime-rl` itself — running the test suite, contributing changes, adding new model architectures, and the small-scale tooling we use to iterate on MoE families without booting up a 100B+ run.

## Table of Contents

- [Test suite](#test-suite)
  - [Layout](#layout)
  - [Running tests locally](#running-tests-locally)
  - [CI workflows](#ci-workflows)
  - [Markers](#markers)
- [Pre-commit hooks](#pre-commit-hooks)
- [Testing MoE at small scale](#testing-moe-at-small-scale)
  - [Step 1: build and verify a mini model](#step-1-build-and-verify-a-mini-model)
  - [Step 2: SFT warmup](#step-2-sft-warmup)
  - [Step 3: RL on reverse-text](#step-3-rl-on-reverse-text)
  - [Adding a new architecture](#adding-a-new-architecture)

## Test suite

The test suite is split into three tiers, each with its own CI workflow.

### Layout

- **`tests/unit/`** — fast-running, hermetic tests for isolated logic: config parsing and validation, advantage / loss / scheduler / packer math, individual dataset paths, model-conversion roundtrips, etc. Tests that need a GPU are tagged with the `gpu` marker.
- **`tests/integration/`** — full-stack RL/SFT runs on a tiny model end-to-end through inference + orchestrator + trainer (e.g. `test_reverse_text.py`, `test_reverse_text_lora.py`, `test_reverse_text_moe.py`, `test_reverse_text_multi_run.py`, `test_alphabet_sort.py`).
- **`tests/nightly/`** — long-running training runs against shipped configs and real environments (`hendrycks_sanity`, `acereason_math`, `multimodal_color_codeword`, `wiki_search`, `wordle`, …). Each runs to completion on the research cluster with a 24h timeout.

### Running tests locally

```bash
uv run pytest -v                                           # everything
uv run pytest tests/unit -v                                # unit only
uv run pytest tests/integration -v                         # integration only
uv run pytest -v -m "not gpu"                              # CPU-only subset (mirrors CPU CI)
uv run pytest -v -m gpu                                    # GPU-only subset
uv run pytest tests/integration/test_reverse_text.py -vvs  # one specific scenario
```

### CI workflows

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

## Pre-commit hooks

Install the [pre-commit](https://pre-commit.com) hooks before your first commit so ruff and the docs-reference regenerator run automatically:

```bash
uv run pre-commit install
```

The configured hooks:

- **`ruff` check + format** on staged Python files.
- **`docs-reference`** — re-runs [`scripts/generate_docs_reference.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/scripts/generate_docs_reference.py) whenever a config class or the generator itself is staged. If `docs/reference.md` would change, the commit fails so you can re-stage the regenerated file.

## Testing MoE at small scale

When working on MoE architectures (GLM-4, Kimi, etc.), you can't iterate on a 100B+ model locally. The workflow below builds a ~0.5B model with the same architecture, warms it up with SFT, and runs RL — all on 1–2 GPUs. The goal is catching bugs in modeling code, state-dict conversions, and pipeline integration before scaling.

### Step 1: build and verify a mini model

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe
```

This creates a ~543M parameter GLM-4 MoE (1024 hidden, 24 layers, 8 experts) with random weights, copies the tokenizer from the original GLM-4 model, and verifies the HF↔PrimeRL roundtrip is lossless. To re-verify after a modeling-code change without re-creating the model:

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe --verify-only
```

### Step 2: SFT warmup

Use the shipped debug MoE SFT config with reverse-text data:

```bash
uv run sft @ configs/debug/moe/sft/train.toml \
  --model.name ./mini-glm-moe \
  --data.name PrimeIntellect/Reverse-Text-SFT \
  --data.type null \
  --max_steps 200 \
  --optim.lr 1e-4 \
  --ckpt.weights
```

Loss drops from ~12 to ~2.5. The output won't be coherent, but the model now has a non-trivial distribution so KL divergence becomes meaningful in RL. A pre-built SFT'd checkpoint lives at [samsja/mini-glm-moe](https://huggingface.co/samsja/mini-glm-moe).

### Step 3: RL on reverse-text

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

### Adding a new architecture

To add (e.g.) Kimi 2.5:

1. Add the modeling code under `src/prime_rl/trainer/models/<arch>/`.
2. Add a preset to `scripts/mini_moe.py` with the config class, small dimensions, HF + PrimeRL model classes, and tokenizer source:

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

3. Run the three steps above with `--arch <your_arch>`.

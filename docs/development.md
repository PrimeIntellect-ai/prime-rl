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
- [Adding a Custom VLM Implementation](#adding-a-custom-vlm-implementation)
- [Adding a Decision Model](#adding-a-decision-model)
  - [Implement the Model and Its Strategy](#implement-the-model-and-its-strategy)
  - [Verify It](#verify-it)
  - [Requirements for merging a decision model](#requirements-for-merging-a-decision-model)
- [Selecting a Model Class by an Extra Config File](#selecting-a-model-class-by-an-extra-config-file)

## Test Suite

The test suite is split into three tiers.

### Layout

- **`tests/unit/`** — fast-running, hermetic tests for isolated logic: config parsing and validation, advantage / loss / scheduler / packer math, individual dataset paths, model-conversion roundtrips, etc. Tests that need a GPU are tagged with the `gpu` marker.
- **`tests/integration/`** — full-stack RL/SFT runs on a tiny model end-to-end through inference + orchestrator + trainer.
- **`tests/regression/`** — nightly regression suites: `test_parallel_invariance.py`, which checks that DP/CP/EP layouts train the same first step as a single GPU, and evals from `configs/eval/` compared against `baselines/eval/`.

### Running Tests Locally

```bash
uv run pytest -v                                           # everything
uv run pytest tests/unit -v                                # unit only
uv run pytest tests/integration -v                         # integration only
uv run pytest -v -m "not gpu"                              # CPU-only subset (mirrors CPU CI)
uv run pytest -v -m gpu                                    # GPU-only subset
uv run pytest tests/unit tests/integration -m tiny -v      # fast GPU subset (mirrors the Tiny (GPU) job)
uvx modal run scripts/modal_ci.py --suite tiny             # the Tiny (GPU) job itself, on Modal
uvx modal run scripts/modal_ci.py --suite full --group rest  # one group of the full GPU suite, on Modal
uv run pytest tests/integration/test_reverse_text.py -vvs  # one specific scenario
uv run playwright install --only-shell chromium            # browser for the integration dashboard smoke tests
```

### CI Workflows

| Workflow | Trigger | What runs | Where |
|---|---|---|---|
| [`tests.yml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/tests.yml) | CPU jobs: every PR + push to `main`. `gsm8k_eval`: relevant changes in non-draft PRs, every push to `main`, manual dispatch | CPU: `pytest tests/unit -m "not gpu"`, plus a slim-wheel install check that `prime-rl-configs` imports cleanly without heavy deps (no torch / vllm / transformers / wandb / verifiers / datasets / liger / loguru in `sys.modules`). Separate hosted `gsm8k_eval` job | `ubuntu-latest` |
| [`gpu-tiny.yml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/gpu-tiny.yml) | Every push to a non-draft PR from this repo (opened, synchronized, reopened, marked ready for review). Also manual dispatch | `Tiny (GPU)`: the fast GPU subset, `pytest tests/unit tests/integration -m tiny` (distributed invariance, offloading and checkpointing, GPU loss tests, short RL and SFT runs), split by `scripts/modal_ci.py` into test groups that run in parallel | Modal, one container per test group (2xH100 or 1xH100), started from `ubuntu-latest` |
| [`gpu.yml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/gpu.yml) | On a PR: applying the `gpu-ci` label (Triage or higher). Also every merge to `main` and manual dispatch | `Tiny (GPU)` first, then the full GPU suite: every GPU test, `pytest tests/unit tests/integration -m gpu`, as one job per test group of `scripts/modal_ci.py` ([`gpu-full.yml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/gpu-full.yml)): one per integration test, the slowest unit test files on their own, and the rest | Modal, one container per test group (2xH100 or 1xH100), started from `ubuntu-latest` |
| [`regression.yml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/.github/workflows/regression.yml) | 03:00 PST daily + manual `workflow_dispatch` (`suite` picks one suite; `eval_config` picks one eval config); `eval` also on PRs that touch it | `parallel`: `tests/regression/test_parallel_invariance.py`. `eval`: every `tests/regression/configs/eval/*.toml` (gsm8k, PI-TerminalBench2.1) against Prime Inference, each compared against its baseline in `tests/regression/baselines/eval/`. Nightly and manual runs post one Slack message per config through the `SLACK_WEBHOOK_URL` secret | `eval`: `ubuntu-latest`. `parallel`: Modal (4xH100), started from `ubuntu-latest` |

Every push to a non-draft PR runs the `Tiny (GPU)` job, and a newer push cancels the PR's queued or in-flight Tiny run. Draft PRs run no GPU jobs. To run the full GPU suite on a PR's current commit, apply the `gpu-ci` label (drafts included). The workflow removes the label when it starts, so re-apply it to run again after new pushes or to retry. A new label trigger cancels the PR's queued or in-flight full-suite run. The full suite starts only when its Tiny job passes, and runs one at a time across PRs and `main`, two test groups at once, so that Tiny runs keep enough GPUs: a run waits for the one in progress, and a newer waiting run replaces an older one (re-apply the label). Fork PRs cannot run GPU CI: a maintainer can push the branch to this repo and label that PR. A PR with merge conflicts cannot run it until the conflicts are resolved.

### Markers

Three pytest markers are declared in `pyproject.toml` (`addopts = "--strict-markers"`):

- `gpu` — gate a test that needs CUDA. CPU CI uses `-m "not gpu"`; the `Full (GPU)` job runs every `gpu` test. A GPU test in a file that no group in `scripts/modal_ci.py` lists runs in the full suite's 2-GPU `rest` group, so a test file that takes minutes of its own gets a new group there, and a job in `gpu-full.yml`'s matrix.
- `slow` — gate a test that's expensive enough you'd usually skip it locally. Deselect with `-m "not slow"`.
- `tiny` — add a GPU test to the fast subset that the `Tiny (GPU)` job runs on Modal for every push to a non-draft PR: distributed invariance, offloading and checkpointing, the GPU loss tests, and short RL and SFT runs. `scripts/modal_ci.py` runs each test group in its own container, so the job takes as long as its slowest group, the RL run (~2.5 min). A tiny test in a file that no group lists runs in the 1-GPU group, so a test that needs 2 GPUs or minutes of its own gets a new group there. Check the test's time with `--durations=0` before marking it.

## Pre-Commit Hooks

Install the [pre-commit](https://pre-commit.com) hooks before your first commit so ruff check + format run on staged Python files automatically:

```bash
uv run pre-commit install
```

## Adding a New Model

Bringing up a new model family is three steps: implement the modeling code, register a mini preset, and run the smoke test. The preset and smoke test let you iterate on the modeling code at ~0.5B scale on 1–2 GPUs instead of paying the cost of the full-size model — useful for catching bugs in modeling code, state-dict conversions, and pipeline integration before scaling.

### Implement the Modeling Code

Drop the modeling code under `src/prime_rl/trainer/models/<arch>/`, mirroring an existing family (`glm4_moe/` or `qwen3_moe/` are good starting points). A model family provides:

- **Config**: a subclass of HF `PretrainedConfig`, so the `config.json` the trainer writes stays loadable by vLLM.
- **Model**: an inner model and a `*ForCausalLM`, both subclassing `PreTrainedModelPrimeRL` (`models/base.py`, a plain `nn.Module`) and ending `__init__` with `self.post_init()`, which initializes the weights off the meta device and then ties them. The inner model returns the output dataclasses from `models/base.py`.
- **Tying**: set `supports_tied_word_embeddings = True` on the `*ForCausalLM` if the architecture allows tying `lm_head` to the input embeddings. The config's `tie_word_embeddings` then decides whether a checkpoint ties.
- **Init**: override `_init_weights` for parameters the default init does not cover, calling `super()._init_weights(module)` first, and `init_buffers_post_meta` for buffers no checkpoint carries.
- **Conversion**: `conversion_chain`, `is_hf_state_dict` and `is_prime_state_dict`, which map between the HF checkpoint and the training layout.
- **Registration**: the config class / model class pair in `_CUSTOM_CAUSAL_LM_MODELS` (`models/__init__.py`), and the config class with `AutoConfig.register`. A VLM family also registers its composite `model_type` in `_CUSTOM_VLM_MAPPING`.

### Register a Mini Preset

Add an entry to [`scripts/mini_moe.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/scripts/mini_moe.py) so the smoke-test workflow can build a ~0.5B test model in your architecture. The preset names the config class, picks small dimensions, and wires up the HF + prime-rl model classes plus a tokenizer source:

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

Build the mini model. This creates a ~543M-parameter GLM-4 MoE (1024 hidden, 24 layers, 8 experts) with random weights, copies the tokenizer from the original GLM-4 model, and verifies the HF↔prime-rl roundtrip is lossless:

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe
```

To re-verify the roundtrip after a modeling-code change without re-creating the model:

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe --verify-only
```

Warm up the random-weight mini model with SFT on reverse-text so KL divergence becomes meaningful in the RL phase. Loss drops from ~12 to ~2.5 — the output won't be coherent, but the distribution is non-trivial. A pre-built SFT'd checkpoint lives at [samsja/mini-glm-moe](https://huggingface.co/samsja/mini-glm-moe) if you want to skip this step:

```bash
uv run sft \
  --model.name ./mini-glm-moe \
  --max_steps 200 \
  --optim.lr 1e-4 \
  --ckpt
```

Then run the full RL stack on reverse-text:

```bash
uv run rl @ configs/ci/integration/reverse-text-moe/start.toml \
  --model.name samsja/mini-glm-moe \
  --inference.vllm.gpu-memory-utilization 0.7 \
  --inference.vllm.max-model-len 2048
```

What to look for:

- **No crashes.** Validates the full inference + orchestrator + trainer pipeline end-to-end.
- **Finite, non-zero KL.** Confirms the reference distribution is meaningful.
- **Loss reasonable.** Not NaN, not stuck.

Don't expect reward to climb meaningfully in 20 steps on a random model.

### Requirements for merging a new model

Before merging a new model, you need to ensure the following:

- The model meets the contract in [Implement the Modeling Code](#implement-the-modeling-code).
- The small smoke test passes.

In the PR that adds the new model, you also need to provide a table covering the KL mismatch across 20 steps on `math` environment with `batch_size=64`. All the entries in the table must lower than 0.015. If this is not met, the PR will not be merged (unless reasonable justification is provided). This is to ensure all our models are consistent and their implementations match the implementations in the inference framework.

## Adding a Custom VLM Implementation

VLM training (any run with `[model.vlm]` set, SFT or RL) is custom-implementation-only: `get_model` rejects models without a custom PrimeRL VLM class at load time. To make a new VLM family trainable, extend a custom text model with a composite VLM body. The unified dense/MoE Qwen3.5 implementation in `models/qwen3_5/` is the reference. The pieces, in dependency order:

1. **Custom text model first.** The VLM body wraps a custom `*ForCausalLM` (see [Adding a New Model](#adding-a-new-model)), so the text side — including its state-dict conversion and KL-mismatch table — comes first.
2. **Composite VLM body.** A `*VLMModel` that holds the vision encoder and custom text model, embeds tokens, runs vision, scatters image embeddings over placeholder tokens, and builds MRoPE 3D positions from `mm_token_type_ids` whenever `image_grid_thw` is given (the renderer owns the token→modality mapping; the trainer always passes plain `position_ids`, which the model uses for text-only rows). Keep the vision encoder at `model.model.visual` and the text model at `model.model.language_model`: FSDP, freezing, activation checkpointing, compile, the MoE runtime and NCCL weight broadcast find them there (`trainer/vlm.py`). The unified `*ForCausalLM` dispatches on the config: composite config → VLM path, text config → text path.
3. **Always run the vision encoder.** Text-only micro-batches must feed the encoder dummy pixels and graft the result into the graph with zero contribution (`inputs_embeds + image_embeds.sum() * 0.0`) so FSDP/EP collectives stay symmetric across ranks when the encoder is trainable.
4. **Packed-boundary consumption.** Samples pack into shared rows with per-document boundaries in `seq_lens`; every custom model's `forward()` declares the typed `seq_lens`/`seq_lens_are_pre_shard` parameters (the trainer passes them unconditionally) and must honor the boundaries — varlen flash `cu_seqlens`, linear-attention state resets per document, and a loud rejection on attention paths that can't (see the packed-batch guard in any modeling file). Set `supports_packed_multimodal_training` on the VLM model once packed rows are handled — RL fails loudly at startup for VLM models without it.
5. **Registration.** Register the composite `model_type` in `_CUSTOM_VLM_MAPPING` (`models/__init__.py`) so `get_model` dispatches to the custom class. RL turns raw image refs into vision kwargs with the model class's `materialize_images(refs, processor)`; Qwen3.5 and Qwen3.8 Flash Next each define it in their `vision.py`, and a new family implements its own.
6. **Context parallelism (optional).** CP-capable VLMs hold a `CPContext` on all modules which need to be CP-aware and shard embeds/positions inside the model after the vision merge; the trainers defer sharding to the model for every batch with images.
7. **Validation.** Same bar as text models: the KL-mismatch table for the text path, plus an SFT run and an RL run on a real multimodal dataset (the `color-codeword` environment is the reference task).

## Adding a Decision Model

A decision model answers [System One records](training.md#system-one-data): SFT trains it and vLLM serves it at `/v1/systemone`. It generates no tokens and RL does not train it, so in place of the KL mismatch of [a chat model](#requirements-for-merging-a-new-model), its option probabilities are compared with vLLM's.

### Implement the Model and Its Strategy

- **Backbone.** A custom model of prime-rl. A new one comes first, with the bar of [Adding a New Model](#adding-a-new-model) or [Adding a Custom VLM Implementation](#adding-a-custom-vlm-implementation).
- **Readout.** The LM head, or a head of the class's own, computed as vLLM computes it and in the same dtype.
- **Strategy.** A `SystemOneStrategy` (see "Writing a strategy" in [System One Data](training.md#system-one-data)) whose prompts are vLLM's. Name the vLLM version and the file it is ported from; do not import vLLM.
- **Export.** `tools/convert_dcp_to_bf16.py` writes a checkpoint that vLLM serves as it serves the release.

### Verify It

`scripts/decision_parity.py` makes the records of two public datasets, [`typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (text) and [`clevr4`](https://huggingface.co/datasets/IamBusy/OpenJev-Vision-Research-v0.1) (images), gets both sides' answers and compares them; its docstring has the commands. Steps 3 to 5 run on the published weights. Their tables go in the PR description, with the commit, the GPU and the vLLM version. The training of steps 4 and 5 runs anywhere; the rest needs a machine with both the trainer and vLLM.

1. **Unit tests** (CPU CI):
   - the strategy's prompt tokens, read spans and option order equal those of the installed vLLM's strategy. Cover every question type, text and JSON states, keys and options in no sorted order, non-ASCII text, missing instructions and descriptions, the most options the strategy takes and one more, and a prompt that is cut or refused. If the pinned vLLM does not have the strategy, compare with the prompts of vLLM's own tests, or with its source, at a named commit;
   - if the strategy reads images, the prompt tokens, pixel values and grids of records with two images of different sizes equal those of the release's processor;
   - the logits of a head of the class's own equal vLLM's code's on the same hidden states;
   - the readout gives several records packed in a row with padding the logits it gives each alone;
   - the class built from the published `config.json` has the published checkpoint's tensor names and shapes, and the export of its weights loads back unchanged.
2. **GPU test** (`gpu-ci`): an integration test trains a tiny, randomly initialised checkpoint for a few steps on two GPUs with validation, and its export has the release's layout.
3. **Parity with vLLM**, on the test records of `typed-decisions` (2,000 questions of every type, and 500 in reordered copies of the records). Serve with `uv run inference`, without prefix caching, and send the requests to the engine's port. If the pinned vLLM lacks the class, use one that has it, with a float32 LM head (`head_dtype`), and name its commit. Compare the trainer with vLLM on the release and on the export of step 4's run, and the release with that export (`--changed`), which must differ.
4. **SFT on a real dataset.** Train on the training records of `typed-decisions` with a checked-in config, on at least two GPUs and with `val.eval_on_start = true`, and take the last checkpoint. Score the release and the export on the test records, once. `val/brier/mean` at the first step, which the packed training path gives, must be the Brier score of the release's answers on `validation.jsonl`.
5. **Images**, if the strategy reads them: the release's parity on `clevr4`, and an SFT run on it, scored as in step 4.

### Requirements for merging a decision model

- Steps 1 and 2 pass in CI.
- Every comparison of step 3 and 5 passes. Per question type, the mean KL(vLLM || trainer) over a question's options is at most 5e-4; over all questions, the top option is the same for at least 99.8% of the decided ones, where it leads by more than 0.04 on both sides. Both sides answer the same records, with the same number of prompt tokens.
- The export differs from the release (`compare --changed` passes), and its KL from the labels on the test records is at least 10% below the release's (on `clevr4`, below it). It is also at most 0.24 on `typed-decisions` and 0.76 on `clevr4`, three quarters of what answers that ignore the state get.
- The validation's Brier score before training is within 0.01 of the script's.
- Where the dataset's card has the release's scores from vLLM's `/v1/systemone` (for `typed-decisions`, the rows of its discussion #37), accuracy and KL are within 0.02 of them.

As with a chat model, a result outside these limits needs a justification, such as the same comparison between two of vLLM's own runs (one request at a time against 16 at once).

Not required: RL, the `math` KL table (unless the backbone is new), LoRA, context parallelism and a run from scratch. A further size of a class that meets the requirements needs the parity of its release only.

## Selecting a Model Class by an Extra Config File

Some releases ship a family's stock `config.json` plus an extra JSON file next to it, which makes the inference server load them as another model class. To train such a release with its own class, subclass the family's PrimeRL class and set:

- `extra_config_file` and `extra_config_attr`: the file's name and the config attribute that holds its content, as the inference server names them.
- `stock_model_type` and `stock_architecture`: the `model_type` and only architecture of the `config.json` it extends.
- `export_files` (optional): `{state-dict prefix: file name}` for weights the export writes to one file outside `model.safetensors.index.json`, without their prefix. Loading adds the prefix back.

Add the class to `_SELECTED_BY_EXTRA_CONFIG` (`models/__init__.py`). prime-rl then:

- builds it for a checkpoint whose `config.json` has that model type, exactly that architecture and the file next to it, with the file's content on the config under `extra_config_attr` (for a Hub model whose weights are cached, the cache alone decides);
- keeps `architectures` stock, so code that needs the model's architecture name uses `type(model).__name__`;
- writes the file next to the stock `config.json` in checkpoint step directories and in `tools/convert_dcp_to_bf16.py` exports, while the FP8 tools refuse a class with `export_files`;
- leaves a release that lacks a tensor the class keeps, or names its weights otherwise, to the class's `is_hf_state_dict`, `is_prime_state_dict` and `convert_to_prime`, which only `setup_model` runs;
- refuses it in RL training, whose weight transfer to the inference server converts weights with the family's class rather than the selected one.

`tests/unit/train/models/extra_config_stub.py` is a minimal example, and Clef-Flash (`src/prime_rl/trainer/models/qwen3_5/modeling_clef_flash.py`) a full one.

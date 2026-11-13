# Decision models

A decision model answers typed questions about a state with one probability distribution per question, read from hidden states by an added head instead of generated as text. This folder trains decision models with the stock `sft` entrypoint on [System One data](../../docs/training.md#system-one-data). `open_jev/` is the minimal reference: [Open-Jev](https://github.com/Zefan-Cai/Open-Jev), Qwen3.5-2B with a scalar decision head, fine-tuned in full. Each model has a folder of its own, and every model is fine-tuned in full with the same loss.

## The contract

A decision model is three pieces of user code, named in the config:

| Piece | Config | What it does |
|---|---|---|
| Encoder | `[data.encoder]` | A `SystemOneEncoder` subclass: a record becomes tokens, attention segments and, per question, the token spans the head reads. It owns its chat template. |
| Model class | `[model.implementation]` | A subclass of prime-rl's class for the architecture, with the head and a `decision_logits` method. The checkpoint holds the head's weights. |
| Loss | `[loss] type = "custom"` | `decision_models.loss.decision_loss`, the same function for every model. |

The method is the model's readout:

```python
def decision_logits(self, hidden_states: Tensor, batch: Batch) -> list[Tensor]
```

It returns one 1-D logit vector per question of the micro-batch, in pack order (records in `batch["sample_data"]` order, questions in record order), with one logit per option in the order of the question's `options`. `hidden_states` are the backbone's final hidden states and `batch["loss_mask"]` marks every read token. It is called inside the forward pass, so it calls the head as a module, on inputs in the head's weight dtype.

`decision_loss` calls the method on a batch without the targets, checks that it got one vector per question with one logit per option, and computes soft cross-entropy plus `brier_weight` x Brier score per labelled question. It logs `ce`, `brier` and `accuracy` under `loss/` in training and `val/` in validation, each a mean over labelled questions.

## Models

| Folder | Model | Base | How it reads a record |
|---|---|---|---|
| [`open_jev/`](#run-open-jev) | Open-Jev | Qwen3.5-2B | One yes/no prompt per candidate answer; a scalar head scores each prompt |
| [`d3/`](d3/README.md) | d3-nano | Qwen3.5-2B | One prompt per question; a 255-code readout at its last token |
| [`decider/`](decider/README.md) | pplx-decider-flash | Qwen3.5-4B | One prompt per record; the 255-code readout at each question's marker |
| [`clef/`](clef/README.md) | Clef-Flash | Qwen3.5-9B | One prompt per record; a joint head over each question's instruction and option spans |

Open-Jev trains on its own dataset. The other three train on Typed Decisions Synth, converted by [`data/convert_typed_decisions_synth.py`](data/README.md).

## Files

| File | What |
|---|---|
| `loss.py` | `decision_loss`, shared by all models |
| `open_jev/encoder.py` | `OpenJevEncoder`: every option of every question is its own yes/no prompt (one prompt for noul), each its own attention segment, read at its last token |
| `open_jev/model.py` | `Qwen3_5Decision`: prime-rl's Qwen3.5 plus `decision_head`, a linear layer that scores a prompt. A noul question's logits are `[0, score]`; a choice or score question's are its prompts' scores |
| `open_jev/prepare_checkpoint.py` | The checkpoint: the base model's files plus the head, initialised to `lm_head[Yes] - lm_head[No]` |
| `open_jev/convert_dataset.py` | Open-Jev's dataset as System One records, one question per record |
| `open_jev/train.toml`, `open_jev/smoke.toml` | The full fine-tuning run, and a 30-step mechanics check composed after it |

## Run Open-Jev

From the repository root, on one GPU. The data, the loss and both configs are checked on CPU; neither config has run on a GPU yet, so memory use and the hyperparameters are unmeasured. Prepare the checkpoint and the records once:

```bash
uv run hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir outputs/open-jev/Qwen3.5-2B
uv run hf download ZefanCai/Open-Jev --repo-type dataset --local-dir outputs/open-jev/Open-Jev
uv run python examples/decision_models/open_jev/prepare_checkpoint.py outputs/open-jev/Qwen3.5-2B outputs/open-jev/Qwen3.5-2B-decision
PYTHONPATH=examples uv run python examples/decision_models/open_jev/convert_dataset.py outputs/open-jev/Open-Jev outputs/open-jev/Open-Jev-system-one
```

Then the mechanics check, and the run. `PYTHONPATH=examples` makes `decision_models` importable in the trainer:

```bash
PYTHONPATH=examples uv run sft @ examples/decision_models/open_jev/train.toml @ examples/decision_models/open_jev/smoke.toml
PYTHONPATH=examples uv run sft @ examples/decision_models/open_jev/train.toml
```

The run is 13,700 steps: one pass over the 79,116 training records takes 13,677, and the last 23 begin a second pass. It validates on the 3,723 validation records before the first step and every 1,370 steps, the last time on the final step. The first validation measures the untrained head, a zero-shot Yes/No judge. A run without a Prime API key needs `--no-monitors.prime`.

## Differences from Open-Jev's training

- The whole language model trains, at one learning rate (1e-5, untuned, cosine schedule with warmup). Open-Jev keeps the language model's weights frozen and trains rank-8 adapters beside them at 5e-5 and the head at 1e-4, both constant.
- A step is one packed row of up to 4,096 tokens, 5.8 records on average. Open-Jev's step is 4 records.
- The head computes in bf16 like the rest of the model. Open-Jev computes it in fp32.
- The training split is the redistributable one. Open-Jev-2B trained on 1,700 more records, which its dataset does not redistribute.
- Validation reports the loss on uncalibrated logits. Open-Jev fits a temperature on the calibration split before it evaluates.

The prompts, their tokens, the head, its initialisation, the loss (soft cross-entropy plus 0.1 x Brier), AdamW with weight decay 0.01 and gradient clipping at 1.0 are Open-Jev's.

## Third-party notice

The prompt format in `open_jev/encoder.py` (`render`, `compile_question`, `candidate_prompts`) is taken from `jev/api.py` of https://github.com/Zefan-Cai/Open-Jev at commit `bd4118882f733574a3250a4b65fe4d884130c08b`, and the head's initialisation in `open_jev/prepare_checkpoint.py` follows `jev/model.py` of the same commit. Open-Jev's licence:

```text
MIT License

Copyright (c) 2026 Open-Jev contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

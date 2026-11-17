# Open-Jev

[Open-Jev](https://github.com/Zefan-Cai/Open-Jev) is Qwen3.5-2B with a scalar decision head. Every candidate answer of a question is its own yes/no prompt (one prompt for a noul question), and the head scores each prompt from its last token. This folder fine-tunes Qwen3.5-2B in full on Open-Jev's dataset, and is the minimal example of a decision model whose class and strategy are user code.

## Native and example code

The model class and its strategy are this folder's: the config names the class, which declares the strategy. The loss is prime-rl's (`[loss] type = "decision"`):

| File | What |
|---|---|
| `model.py` | `Qwen3_5Decision`, the `[model.implementation]` class: prime-rl's Qwen3.5 plus `decision_head`, a linear layer that scores a prompt, initialised to `lm_head[Yes] - lm_head[No]`. It declares `OpenJevStrategy` |
| `strategy.py` | `OpenJevStrategy`. `render_record`: every option of every question is its own yes/no prompt (one prompt for noul), each its own attention segment, read at its last token. `option_logits`: `decision_head` scores each prompt; a noul question's logits are `[0, score]`, a choice or score question's are its prompts' scores |
| `convert_dataset.py` | Open-Jev's dataset as System One records, one question per record |
| `train.toml`, `smoke.toml` | The full fine-tuning run, and a 30-step mechanics check composed after it |

## vLLM

The class has no vLLM counterpart, so vLLM cannot serve a run's export, which holds the decision head and names the base architecture ([Exporting checkpoints](../../../docs/training.md#exporting-checkpoints)).

## Run

Download the base model and the dataset, and convert the records once, then run as [Running a model](../README.md#running-a-model) says, with `PYTHONPATH=examples`:

```bash
uv run hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir outputs/open-jev/Qwen3.5-2B
uv run hf download ZefanCai/Open-Jev --repo-type dataset --local-dir outputs/open-jev/Open-Jev
PYTHONPATH=examples uv run python examples/decision_models/open_jev/convert_dataset.py outputs/open-jev/Open-Jev outputs/open-jev/Open-Jev-system-one
```

`model.name` is the base model as downloaded. It has no decision head, so when a run starts the class sets the head from the loaded LM head (`init_missing_weights`); a resumed run takes the trained head from its checkpoint, and so does a run whose `model.name` is a checkpoint step directory. With `model.init.type = "scratch"` the head starts from its random init, not from the Yes and No rows.

The run is 13,700 steps: one pass over the 79,116 training records takes 13,677, and the last 23 begin a second pass. It validates on the 3,723 validation records before the first step and every 1,370 steps, the last time on the final step. The first validation measures the untrained head, a zero-shot Yes/No judge.

## Differences from Open-Jev

- The whole language model trains, at one learning rate (1e-5, untuned, cosine schedule with warmup). Open-Jev keeps the language model's weights frozen and trains rank-8 adapters beside them at 5e-5 and the head at 1e-4, both constant.
- A step is one packed row of up to 4,096 tokens, 5.8 records on average. Open-Jev's step is 4 records.
- The head computes in bf16 like the rest of the model. Open-Jev computes it in fp32.
- The training split is the redistributable one. Open-Jev-2B trained on 1,700 more records, which its dataset does not redistribute.
- Validation reports the loss on uncalibrated logits. Open-Jev fits a temperature on the calibration split before it evaluates.
- A question without instructions reads as one with empty instructions, and a null noul description as none; Open-Jev refuses both. Open-Jev's dataset has neither.
- Instructions or a description that is not text, an object or an array (e.g. a number) makes the strategy raise and stop the run, as Open-Jev raises, where prime-rl's own strategies skip such a record. Open-Jev's dataset has none.

The prompts, their tokens, the head, its initialisation, the loss (soft cross-entropy plus 0.1 x Brier), AdamW with weight decay 0.01 and gradient clipping at 1.0 are Open-Jev's.

## Third-party notice

The prompt format in `strategy.py` (`render`, `compile_question`, `candidate_prompts`) is taken from `jev/api.py` of https://github.com/Zefan-Cai/Open-Jev at commit `bd4118882f733574a3250a4b65fe4d884130c08b`, and the head's initialisation in `model.py` follows `jev/model.py` of the same commit. Open-Jev's licence:

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

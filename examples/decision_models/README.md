# Decision models

A decision model answers typed questions about a state with one probability distribution per question, read from hidden states by an added head instead of generated as text. This folder trains decision models with the stock `sft` entrypoint on [System One data](../../docs/training.md#system-one-data). Each model has a folder of its own, and every model is fine-tuned in full with the same loss. prime-rl builds d3 and Clef-Flash from their releases with its own classes and encoders, so their folders hold only configs; Decider and Open-Jev show a model class and an encoder of one's own.

## The contract

A decision model is an encoder and a model class, plus the loss prime-rl ships for them. prime-rl ships the encoders and classes of d3 and Clef-Flash, which vLLM 0.33 serves ([Native Decision Models](../../docs/training.md#native-decision-models)); for another model they are user code, named in the config:

| Piece | Config | What it does |
|---|---|---|
| Encoder | `[data.encoder]` | A `SystemOneEncoder` subclass: a record becomes tokens, attention segments and, per question, the token spans the head reads. It owns its chat template. |
| Model class | none for a native release (d3, Clef-Flash), else `[model.implementation]` | prime-rl's class for the release, or a subclass of prime-rl's class for the architecture, with the head and a `decision_logits` method. The head's weights come from the checkpoint, or from the class's `init_missing_weights` when the checkpoint has none. |
| Loss | `[loss] type = "decision"` | prime-rl's decision loss, the same for every model. |

`decision_logits` is the model's readout: one logit vector per question of the packed row. Its contract, and what `decision_loss` computes from it, are in [System One Data](../../docs/training.md#system-one-data) and in `DecisionModel` in `src/prime_rl/trainer/sft/system_one/loss.py`.

## Models

| Folder | Model | Base | How it reads a record | Class and encoder | vLLM 0.33 |
|---|---|---|---|---|---|
| [`d3/`](d3/README.md) | d3-nano | Qwen3.5-2B | One prompt per question; a 255-code readout at its last token | prime-rl's | Serves it |
| [`clef/`](clef/README.md) | Clef-Flash | Qwen3.5-9B | One prompt per record; a joint head over each question's instruction and option spans | prime-rl's | Serves it |
| [`decider/`](decider/README.md) | pplx-decider-flash | Qwen3.5-4B | One prompt per record; the 255-code readout at each question's marker | A subclass of prime-rl's d3 class, and the folder's encoder | Refuses it (format version 2) |
| [`open_jev/`](open_jev/README.md) | Open-Jev | Qwen3.5-2B | One yes/no prompt per candidate answer; a scalar head scores each prompt | The folder's | No counterpart |

Open-Jev trains on its own dataset. The other three train on Typed Decisions Synth, converted by [`data/convert_typed_decisions_synth.py`](data/README.md).

## Running a model

From the repository root, on one GPU (Clef-Flash's full run needs more, as its README says). A model's README says what to download and convert; its configs read them where those commands put them. Then the mechanics check, a 30-step run on the first layers with validation and a checkpoint every 10 steps, and the run:

```bash
uv run sft @ examples/decision_models/<model>/train.toml @ examples/decision_models/<model>/smoke.toml
uv run sft @ examples/decision_models/<model>/train.toml
```

A config that names this folder's code (`decision_models.*`) needs `PYTHONPATH=examples` before `uv run`, which makes it importable in the trainer. A run without a Prime API key needs `--no-monitors.prime`. The first validation, before any update, measures the released model (for Open-Jev, the untrained head).

A release that stores its weights under other names than prime-rl's is converted on the first load (the mechanics check's, if it runs first) into `prime/` beside its files (or under `model.conversion_dir`), a copy every later load reads. prime-rl never refreshes the copy: delete it when the release's files change, including when an interrupted download is completed. The download stays as it is.

No config has run on a GPU: memory use and the hyperparameters are unmeasured. The records, the encoders, the loss and the configs are checked on CPU.

# Decision models

A decision model answers typed questions about a state with one probability distribution per question, read from hidden states by an added head instead of generated as text. This folder trains decision models with the stock `sft` entrypoint on [System One data](../../docs/training.md#system-one-data). Each model has a folder of its own, and every model is fine-tuned in full with the same loss. prime-rl builds d3 and Clef-Flash from their releases with its own classes and strategies, so their folders hold only configs; Decider and Open-Jev show a model class and a strategy of one's own.

## The contract

A decision model is a model class and the strategy it declares, plus the loss prime-rl ships for them. prime-rl ships the classes and strategies of d3 and Clef-Flash, which vLLM 0.33 serves ([Native Decision Models](../../docs/training.md#native-decision-models)); for another model they are user code, and the config names the class:

| Piece | Config | What it does |
|---|---|---|
| Model class | none for a native release (d3, Clef-Flash), else `[model.implementation]` | prime-rl's class for the release, or a subclass of prime-rl's class for the architecture, with the head. The head's weights come from the checkpoint, or from the class's `init_missing_weights` when the checkpoint has none. Its classmethod `system_one_strategy` returns its strategy's class. |
| Strategy | none: the model class declares it | A `SystemOneStrategy` subclass, built from the run's tokenizer and the model's config. `render_record`: a record becomes tokens, attention segments and, per question, the token spans the head reads; it owns its chat template. `option_logits`: called on one record at a time, the head's logits at those spans, one per option of each of its questions, from the model's modules. |
| Loss | `[loss] type = "decision"` | prime-rl's decision loss, the same for every model. |

vLLM 0.33's read strategy is a prime-rl strategy (`render_record` and `option_logits`) plus the softmax; the model class keeps the head's weights. The contract of `option_logits`, and what `decision_loss` computes from it, are in [System One Data](../../docs/training.md#system-one-data) and in `SystemOneStrategy` and `DecisionModel` in `src/prime_rl/trainer/sft/system_one/`.

## Models

| Folder | Model | Base | How it reads a record | Class and strategy | vLLM 0.33 |
|---|---|---|---|---|---|
| [`d3/`](d3/README.md) | d3-nano | Qwen3.5-2B | One prompt per question; a 255-code readout at its last token | prime-rl's | Serves it |
| [`clef/`](clef/README.md) | Clef-Flash | Qwen3.5-9B | One prompt per record; a joint head over each question's instruction and option spans | prime-rl's | Serves it |
| [`decider/`](decider/README.md) | pplx-decider-flash | Qwen3.5-4B | One prompt per record; the 255-code readout at each question's marker | Subclasses of prime-rl's d3 class and strategy | Refuses it (format version 2) |
| [`open_jev/`](open_jev/README.md) | Open-Jev | Qwen3.5-2B | One yes/no prompt per candidate answer; a scalar head scores each prompt | The folder's | No counterpart |

Open-Jev trains on its own dataset. The other three train on Typed Decisions Synth, converted by [`data/convert_typed_decisions_synth.py`](data/README.md).

## Running a model

From the repository root, on one GPU (Clef-Flash's full run needs two, as its README says). A model's README says what to download and convert; its configs read them where those commands put them. Then the mechanics check, a 30-step run on the first layers with validation and a checkpoint every 10 steps, and the run:

```bash
uv run sft @ examples/decision_models/<model>/train.toml @ examples/decision_models/<model>/smoke.toml
uv run sft @ examples/decision_models/<model>/train.toml
```

A config that names this folder's code (`decision_models.*`) needs `PYTHONPATH=examples` before `uv run`, which makes it importable in the trainer. A run without a Prime API key needs `--no-monitors.prime`. The first validation, before any update, measures the released model (for Open-Jev, the untrained head).

A release that stores its weights under other names than prime-rl's is converted on the first load (the mechanics check's, if it runs first) into `prime/` beside its files (or under `model.conversion_dir`), a copy every later load reads. prime-rl never refreshes the copy: delete it when the release's files change, including when an interrupted download is completed. The download stays as it is.

d3's and Clef-Flash's configs ran, with these settings, on two H100s (80 GB), and vLLM 0.33 served their exports; their READMEs give the numbers. Clef-Flash's full run does not fit one 80 GB GPU. They ran before prime-rl's d3 and Clef-Flash prompts and readouts moved into their strategies; the prompts were since checked byte-identical on CPU for the 732 validation records and 96 edge records, and the readout code was checked unchanged by reading. Decider's and Open-Jev's configs have not run on a GPU: their memory use and hyperparameters are unmeasured. Serve a d3 or Clef-Flash export with the router off, as their READMEs say.

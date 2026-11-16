# Clef

[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) (Cloudflare, Apache-2.0) is Qwen3.5-9B with a joint schema head. One prompt holds the state and the schema of every question. The head reads the hidden states of each question's instruction and of each of its options, attends over the whole record, and decides the record's questions together. This folder fine-tunes the released model in full on [Typed Decisions Synth](../data/README.md).

## Native and example code

The model class and its strategy are prime-rl's ([Native Decision Models](../../../docs/training.md#native-decision-models)); this folder holds only the configs. `model.name` is the release as downloaded, which prime-rl builds as `ClefFlashForJointSchema`, the class vLLM 0.33 serves Clef-Flash with, so the config has no `[model.implementation]`. The class declares the in-tree `ClefFlashStrategy`, which builds the prompt of vLLM 0.33's `/v1/systemone` and runs the joint head on each record as vLLM 0.33's pooler (`JointSchemaPooler`) does.

| File | What |
|---|---|
| `train.toml`, `smoke.toml` | The full fine-tuning run, and a 30-step mechanics check composed after it |

## vLLM

vLLM 0.33 serves Clef-Flash. The prompt and the head are vLLM's, and a run's [export](../../../docs/training.md#exporting-checkpoints) has the release's layout (`joint_head_config.json` and `joint_head.safetensors` beside the shards), which vLLM 0.33 serves. On the export of the full run, vLLM's probabilities and the trainer's differ by a median of about 0.001 per question, as much as vLLM's own differ under another batching.

## Run

Download the release, and prepare the records as [their README](../data/README.md) says (they go to `outputs/typed-decisions-synth-system-one`), then run as [Running a model](../README.md#running-a-model) says:

```bash
uv run hf download Cloudflare/clef-flash --local-dir outputs/clef/clef-flash
```

The run is one pass over the 6,682 training records in 290 steps of two packed rows, 23 records a step on average. It validates on the 732 validation records before the first step and every 58 steps, the last time on the final step.

On two H100s (`--deployment.num-train-gpus 2`), before Clef-Flash's prompt and head moved into prime-rl's strategy ([Running a model](../README.md#running-a-model)), the run took about 27 minutes, a step about 4.5 s, and 77.5 of 79.2 GiB per GPU, and the validation loss went from 0.7203 to 0.6362. It does not fit one 80 GB GPU: prime-rl holds each weight in fp32 on the GPU, about 36 GB for 9 billion parameters, with an fp32 gradient next to each trained one, and it moves AdamW's two moments from CPU RAM to the GPU for every optimizer step: up to four fp32 copies at once. On N GPUs (`--deployment.num-train-gpus N`, with `data.batch_size` a multiple of N) each holds 1/N of them. The mechanics check trains 2 layers and is expected to fit one GPU.

## Differences from Clef

Clef published no training code, so every training choice is this example's:

- The language model and the joint head train at one learning rate (1e-5, cosine schedule with warmup), with AdamW, weight decay 0.01 and gradient clipping at 1.0.
- The loss is soft cross-entropy against the soft labels, with no Brier term.
- A step is two packed rows of up to 8,192 tokens. Each record is its own attention segment, so it attends only to itself.
- The vision encoder and the LM head do not train: the records are text, and the loss never calls the LM head.

The strategy cuts a state so that the prompt fits in 8,192 tokens (`data.seq_len`), where vLLM cuts it to fit `min(16384, max_model_len)`; no state of this dataset is cut. Records with images or videos are not supported.

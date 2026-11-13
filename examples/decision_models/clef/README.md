# Clef

[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) (Cloudflare, Apache-2.0) is Qwen3.5-9B with a joint schema head. One prompt holds the state and the schema of every question. The head reads the hidden states of each question's instruction and of each of its options, attends over the whole record, and decides the record's questions together. This folder fine-tunes the released model in full on [Typed Decisions Synth](../data/README.md), under the [contract](../README.md#the-contract) of this folder's decision models.

## Files

| File | What |
|---|---|
| `encoder.py` | `ClefEncoder`: the record as Clef's own prompt, one attention segment. A question's read spans are its instruction, then one span per option, in Clef's option order (noul `true`, `false`; choice ids sorted) |
| `model.py` | `Qwen3_5Clef`: prime-rl's Qwen3.5 plus `joint_head` and `lexical_embedding`, a frozen copy of the LM head's rows for the head's lexical prior. `decision_logits` calls the head once per record, on the record's slice of the packed row |
| `joint_schema_model.py` | Clef's own code, unmodified: the prompt format (`encode_record`) and the head (`JointSchemaHead`) |
| `prepare_checkpoint.py` | The checkpoint: the release's files plus the head under `joint_head.`, the lexical copy, and the head's sizes in `config.json` |
| `train.toml`, `smoke.toml` | The full fine-tuning run, and a 30-step mechanics check composed after it |
| `THIRD_PARTY.md` | The licence of `joint_schema_model.py` |

## Run

From the repository root. Prepare the checkpoint once, and the records as [../data/README.md](../data/README.md#convert) says:

```bash
uv run hf download Cloudflare/clef-flash --local-dir outputs/clef/clef-flash
uv run python examples/decision_models/clef/prepare_checkpoint.py outputs/clef/clef-flash outputs/clef/clef-flash-prime-rl
```

Then the mechanics check, and the run. `PYTHONPATH=examples` makes `decision_models` importable in the trainer:

```bash
PYTHONPATH=examples uv run sft @ examples/decision_models/clef/train.toml @ examples/decision_models/clef/smoke.toml
PYTHONPATH=examples uv run sft @ examples/decision_models/clef/train.toml
```

The run is one pass over the 6,682 training records in 290 steps of two packed rows, 23 records a step on average. It validates on the 732 validation records before the first step and every 58 steps, the last time on the final step. The first validation measures the released model. A run without a Prime API key needs `--no-monitors.prime`.

`accuracy` scores a target whose top options tie against the first of them in Clef's option order. On 3 of the 2,540 validation questions that is another option than in prime-rl's order (noul `false`, `true`; choice ids as the record gives them), so there it is not comparable with a model read in that order.

Memory is not measured, and by arithmetic the full run does not fit one 80 GB GPU. prime-rl holds each weight in fp32 on the GPU, about 36 GB for 9 billion parameters, with an fp32 gradient next to each trained one, and it moves AdamW's two moments from CPU RAM to the GPU for every optimizer step: up to four fp32 copies at once. On N GPUs (`--deployment.num_train_gpus N`, with `data.batch_size` a multiple of N) each holds 1/N of them. The mechanics check trains 2 layers and is expected to fit one GPU.

## Differences from Clef

Clef published no training code, so every training choice is this example's:

- The language model and the joint head train at one learning rate (1e-5, untuned, cosine schedule with warmup), with AdamW, weight decay 0.01 and gradient clipping at 1.0.
- The loss is soft cross-entropy against the soft labels, with no Brier term.
- A step is two packed rows of up to 8,192 tokens. Each record is its own attention segment, so it attends only to itself.
- The vision encoder and the LM head do not train: the records are text, and the loss never calls the LM head.

The prompt, its tokens, the head and its logits are the release's, and the softmax of the logits is the distribution Clef serves. Against the release's inference code:

- The lexical prior reads `lexical_embedding`, a frozen copy of the LM head's rows, where the release reads the LM head's weight. The two are equal, since the LM head does not train.
- The head's three scalar parameters hold one element instead of none, which FSDP requires. The head's outputs are unchanged.
- A state too long for `data.seq_len` (8,192 tokens) is cut to fit, as the release cuts it at 16,384. A record whose questions alone do not fit is skipped with a warning.
- Records with images or videos are not supported.

## Third-party notice

`joint_schema_model.py` is an unmodified copy of the file in [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Apache-2.0). [THIRD_PARTY.md](THIRD_PARTY.md) has the licence. It is not formatted or linted with this repository's rules.

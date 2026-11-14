# Typed Decisions Synth

`convert_typed_decisions_synth.py` writes [Typed Decisions Synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) as [System One records](../../../docs/training.md#system-one-data): multi-question training data for the decision models of this folder.

The dataset has 7,414 synthetic cases with 25,859 questions (10,192 noul, 10,179 choice, 5,488 score) across 149 domains and workflows, in English. A case is a state (text or JSON) and 1 to 8 questions about it, each with an answer and a probability for every option. DeepSeek V4.1 Flash wrote and labelled all of it, and nobody checked it. It was made by Muhammed Nazeem for [Hmm](https://github.com/n4ze3m/hmm) and is released under the MIT licence.

## Convert

From the repository root:

```bash
uv run hf download n4ze3m/typed-decisions-synth --repo-type dataset --local-dir outputs/typed-decisions-synth
uv run python examples/decision_models/data/convert_typed_decisions_synth.py outputs/typed-decisions-synth outputs/typed-decisions-synth-system-one
```

Every record is parsed by prime-rl as it is written, so a malformed case stops the conversion, and so does a question without an answer or without probabilities.

## Output

```text
outputs/typed-decisions-synth-system-one/
├── README.md                the dataset card: it names the subset and its splits
└── default/
    ├── train.parquet        6,682 records, 23,319 questions
    ├── validation.parquet   732 records, 2,540 questions
    └── train_hard.parquet   train with no soft labels
```

Each file has the same five string columns:

| Column | From | Content |
|---|---|---|
| `id` | `state_id` | The case id |
| `state` | `state` | JSON text; a text state is a JSON string |
| `questions` | `questions` | JSON text, unchanged: question id → `{type, instructions, criteria}` |
| `label` | `gold` | JSON text, unchanged: question id → an option id, a level, or true/false |
| `soft_label` | `teacher` | JSON text: question id → option id → probability, the mean of three answers, renormalized to sum to 1. Null in `train_hard` |

A state is text or JSON as the dataset's `state_is_json` flag says: 43 states flagged as text are themselves JSON documents and stay text. The dataset's `domain` and `generator` columns are not carried over.

Every question has both labels. A question trains toward its soft label when it has one, so `train` trains on the soft labels and `train_hard` on the hard labels, one-hot. Load a split with:

```toml
[data]
type = "system_one"
name = "outputs/typed-decisions-synth-system-one"
subsets = ["default"]
splits = ["train"]  # or "validation", "train_hard"
```

The options keep the dataset's order, in which the correct option is listed first about half of the time. The longest record has 5,561 characters of state and questions.

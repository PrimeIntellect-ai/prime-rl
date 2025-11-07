How to run offline difficulty filtering on arbitrary number of nodes on the dataset.

0. Make sure you have the most recent version of the environments installed. Make sure you have a clone of `prime-environments` and are on the `mika/i3` branch (latest commit)

```bash
uv pip install -e /path/to/prime-environments/environments/i3_math
uv pip install -e /path/to/prime-environments/environments/i3_code
uv pip install -e /path/to/prime-environments/environments/i3_science
uv pip install -e /path/to/prime-environments/environments/i3_logic
```

1. For `science` and `math`, we need an LLM judge. Start a vLLM server with the `opencompass/CompassVerifier-3B` for LLM-as-a-judge.

```bash
sbatch configs/difficulty-filtering/llm_judge.sh
```

2. Start the offline evaluation using the `run.sh` script.

```bash
EVAL_CONFIG=configs/difficulty-filtering/code.toml sb -N4 configs/difficulty-filtering/run.sh
```

Notes:
- Specify the number of nodes you wanna run on using the `-N` flag.
- Specify the evaluation config file using the `EVAL_CONFIG` environment variable. We have one for `math`, `code`, `science` and `logic`
- Except for `science`, all splits still specify the `num_examples` to 1024 for debugging purposes.
- The script will automatically save the results to disk (`outputs/evals`) and to the Hugging Face dataset `PrimeIntellect/INTELLECT-3-RL-<split>-Difficulty-Filtering`


3. Use the below logic to compute the average reward for each question in the dataset and merge it with the original dataset.

*This example is for math, but should be ~the same for the other splits.*

```python
import json

from datasets import Dataset, load_dataset


def load_vf_results(path: str) -> Dataset:
    with open(path, "r") as f:
        rows = [json.loads(line) for line in f]
    return Dataset.from_list(rows)

# Math
math_instruction_prompt = (
    "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
)

math = load_dataset("PrimeIntellect/INTELLECT-3-RL", "math", split="train")
vf_math_results = load_vf_results("outputs/evals/i3-math--Qwen--Qwen3-4B-Thinking-2507/c39a71b2/results.jsonl")

vf_math_results = (
    vf_math_results.map(lambda x: {"question": x["prompt"][0]["content"]})
    .select_columns(["question", "reward"])
    .map(lambda x: {"question": x["question"].replace(math_instruction_prompt, "")})
)

# Compute per-example reward
vf_math_results_df = vf_math_results.to_pandas()
math_question_to_reward = vf_math_results_df.groupby("question").reward.mean().to_dict()

merged_math = math.map(lambda x: {"avg_reward": math_question_to_reward.get(x["question"])})
merged_math[0]
```

After, you can push this back to HF using `merged_math.push_to_hub("PrimeIntellect/INTELLECT-3-RL, "math")`.
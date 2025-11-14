"""
A script to build an SFT dataset from rollouts produced by an environment. Includes reasoning content and tool calls in OAI format, if present.

Example Usage:

First, run `uv run vf-eval` (or, alternatively, `uv run eval`) to produce rollouts. Make sure to use the `-C trajectory` flag to include the trajectory in the output.
```bash
uv run vf-eval math500 -s -C trajectory
```

Then, grab the outputs directory (e.g. `outputs/evals/math500--gpt-4.1-mini/53fadd48`) and run this script to build (and push) the SFT dataset:
```bash
uv run scripts/build_sft_dataset.py -o outputs/evals/math500--gpt-4.1-mini/53fadd48 -D mikasenghaas/math500-sft-dataset
```
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import cast

from datasets import Dataset, List, Value, load_dataset
from huggingface_hub import whoami
from openai.types.chat import ChatCompletion, ChatCompletionFunctionToolParam

TOOL_CALL_SCHEMA = {
    "function": {
        "arguments": Value("string"),
        "name": Value("string"),
        "type": Value("string"),
    },
    "id": Value("string"),
    "type": Value("string"),
}

MESSAGES_SCHEMA = List(
    {
        "content": Value("string"),
        "role": Value("string"),
        "reasoning_content": Value("string"),
        "tool_calls": List(TOOL_CALL_SCHEMA),
    }
)

TOOL_SCHEMA = Value("string")  # For now, we JSON-serialize the tool call definition


def get_prompt(result: dict) -> list[dict]:
    return cast(list[dict], result["prompt"])


def get_completion(result: dict) -> list[dict]:
    assert "completion" in result and "trajectory" in result
    completion, trajectory = result["completion"], result["trajectory"]
    responses = [
        ChatCompletion.model_validate_json(trajectory_step["response"]).model_dump() for trajectory_step in trajectory
    ]

    # Get all chat messages from chat completion responses
    oai_responses = [
        {
            k: v
            for k, v in r["choices"][0]["message"].items()
            if k in ["role", "content", "reasoning_content", "tool_calls"]
        }
        for r in responses
    ]

    # Merge with completions
    assert len([c for c in completion if c.get("role") == "assistant"]) == len(oai_responses)
    j = 0
    for i in range(len(completion)):
        if completion[i].get("role") == "assistant":
            completion[i] = oai_responses[j]
            j += 1

    return completion


def get_oai_tools(result: dict) -> list[ChatCompletionFunctionToolParam]:
    return json.loads(result.get("oai_tools", "[]"))


async def main(
    output_dir: Path,
    dataset_name: str | None,
    dataset_private: bool,
) -> None:
    # Load results and metadata
    print(f"Loading results from {output_dir}")
    results = cast(
        Dataset,
        load_dataset(
            "json",
            data_files=f"{output_dir}/results.jsonl",
            split="train",
        ),
    )
    with open(f"{output_dir}/metadata.json", "r") as f:
        metadata = json.load(f)
    env_id = metadata["env_id"]
    print(f"Loaded {len(results)} rollouts for {env_id}")

    # Parse messages (w/ reasoning content and tool calls in OAI format) and get OAI tool definitions
    print("Parsing messages...")
    prompt = [get_prompt(cast(dict, result)) for result in results]
    completion = [get_completion(cast(dict, result)) for result in results]
    oai_tools = [json.dumps(get_oai_tools(cast(dict, result))) for result in results]

    # Create SFT dataset
    print("Creating SFT dataset...")
    ds = (
        Dataset.from_dict(
            {
                "prompt": prompt,
                "completion": completion,
                "tools": oai_tools,
            }
        )
        .cast_column("prompt", MESSAGES_SCHEMA)
        .cast_column("completion", MESSAGES_SCHEMA)
        .cast_column("tools", TOOL_SCHEMA)
    )

    if dataset_name is not None:
        print(f"Pushing SFT dataset to Hugging Face Hub ({dataset_name})...")
        ds.push_to_hub(dataset_name, env_id, split="train", private=dataset_private)
        default_org = whoami().get("name", "")
        repo_name = dataset_name if "/" in dataset_name else f"{default_org}/{dataset_name}"
        print(
            f"Pushed {'private' if dataset_private else 'public'} SFT dataset for {env_id=} to HF Hub (https://huggingface.co/datasets/{repo_name})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    parser.add_argument("--dataset-name", "-D", type=str, default=None)
    parser.add_argument("--dataset-private", "-p", action="store_true")
    args = parser.parse_args()
    asyncio.run(
        main(
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_private=args.dataset_private,
        )
    )

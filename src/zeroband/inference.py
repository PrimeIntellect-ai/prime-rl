import os
import uuid
from pydantic import model_validator
from vllm import LLM, SamplingParams
from pydantic_config import BaseConfig, parse_argv
import vllm

from zeroband.models import ModelName, name_to_hf_model

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq


class Config(BaseConfig):
    name_model: ModelName = "150M"
    dataset: str = "justus27/test-vcu"
    batch_size: int = 32
    sample_per_file: int = 1024
    max_samples: int | None = None
    output_path: str = "outputs"

    @model_validator(mode="after")
    def validate_bs_and_sample_per_file(self):
        if self.sample_per_file % self.batch_size != 0:
            raise ValueError("sample_per_file must be divisible by batch_size")
        if self.max_samples is not None:
            if self.max_samples % self.batch_size != 0:
                raise ValueError("max_samples must be divisible by batch_size")
            if self.max_samples < self.sample_per_file:
                raise ValueError("max_samples must be greater than sample_per_file")
        return self


def fake_chat_template(messages):
    formatted_prompts = []

    for conversation in messages:
        prompt = ""
        for message in conversation:
            if message["role"] == "user":
                prompt += f"Human: {message['content']}\n\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n\n"
        formatted_prompts.append(prompt.strip())

    return formatted_prompts


pa_schema = pa.schema(
    [
        ("input_tokens", pa.list_(pa.int32())),
        ("output_tokens", pa.list_(pa.int32())),
        ("advantages", pa.float32()),
        ("proofs", pa.binary()),
        ("step", pa.int32()),
    ]
)


def get_parquet_table(generated_tokens: list[vllm.RequestOutput], step: int) -> pa.Table:
    # Initialize lists for each column
    input_tokens_list = []
    output_tokens_list = []
    advantages_list = []
    proofs_list = []
    steps_list = []

    # Process each RequestOutput
    for request in generated_tokens:
        # For each output in the request (handling top-n outputs)
        for output in request.outputs:
            # Input tokens are the prompt tokens
            input_tokens_list.append(request.prompt_token_ids)

            # Output tokens from the completion
            output_tokens_list.append(output.token_ids)

            # Initialize with 0 advantage as it's not part of RequestOutput
            # You might want to modify this based on your advantage calculation
            advantages_list.append(0)

            # TODO: Add toploc proof
            proofs_list.append("I am toploc proof, handcrafted by jack".encode())

            # Add step
            steps_list.append(step)

    # Create PyArrow arrays
    arrays = [
        pa.array(input_tokens_list, type=pa.list_(pa.int32())),
        pa.array(output_tokens_list, type=pa.list_(pa.int32())),
        pa.array(advantages_list, type=pa.float32()),
        pa.array(proofs_list, type=pa.binary()),
        pa.array(steps_list, type=pa.int32()),
    ]

    # Create and return table
    return pa.Table.from_arrays(arrays, schema=pa_schema)


def main(config: Config):  # -> list[dict[str, Any]]:
    prompts = ["Write me a novel" for _ in range(5)]

    llm = LLM(model=name_to_hf_model[config.name_model])
    # tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100, presence_penalty=0.1, frequency_penalty=0.1)

    # Load dataset
    dataset = load_dataset(config.dataset, split="train")

    max_samples = config.max_samples or len(dataset)

    step = 0  # step will change once we have the update model api

    # Process batches
    for i in range(0, min(len(dataset), max_samples), config.batch_size):
        # Get batch
        batch = dataset.select(range(i, min(i + config.batch_size, len(dataset))))

        # Prepare messages
        messages = [[{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": "<think>\n"}] for item in batch]

        # Get tokenized inputs
        prompts = fake_chat_template(messages)

        generated_tokens = llm.generate(prompts, sampling_params)

        table = get_parquet_table(generated_tokens, step)

        step_path = f"{config.output_path}/step_{step}"
        os.makedirs(step_path, exist_ok=True)

        pq.write_table(table, f"{step_path}/{uuid.uuid4()}.parquet")


if __name__ == "__main__":
    config = Config(**parse_argv())  # type: ignore

    main(config)

from vllm import LLM, SamplingParams
from datasets import load_dataset
import argparse
from zeroband.inference.rewards import compute_rewards
import json


def evaluate(
    model_name: str="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    dataset_name: str="justus27/math-hendrycks-genesys-format",
    max_samples: int | None=None,
    k: int = 1,
) -> None:
    # load model
    llm = LLM(model=model_name)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048, n=k)

    # load math dataset
    dataset = load_dataset(dataset_name, split="train")

    # construct prompts + metadata
    batch = dataset.select(range(len(dataset) if max_samples is None else max_samples))
    messages = [[{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": "<think>\n"}] for item in batch]
    if tokenizer.chat_template:
        prompts = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    verification_infos = [json.loads(item["verification_info"]) for item in batch]
    task_types = [item["task_type"] for item in batch]

    # generate responses and get verification info
    # TODO: filter out long prompts?
    request_outputs = llm.generate(prompts, sampling_params)
    
    # get rewards (no length reward)
    request_rewards = compute_rewards(request_outputs, verification_infos, task_types, None)

    # get passrate
    passrate = [sum(rewards) / len(rewards) for rewards in request_rewards]
    avg_passrate = sum(passrate) / len(passrate)
    print(f"Average passrate: {avg_passrate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("-d", "--dataset_name", type=str, default="justus27/math-hendrycks-genesys-format")
    parser.add_argument("-s", "--max_samples", type=int, default=None)
    parser.add_argument("-k", "--k", type=int, default=1)
    args = parser.parse_args()
    evaluate(args.model_name, args.dataset_name, args.max_samples, args.k)
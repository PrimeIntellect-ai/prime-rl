#!/usr/bin/env python3
"""
Simple test script to verify the verifiers environment integration.
"""

from types import SimpleNamespace

import verifiers as vf
from datasets import load_dataset
from openai import OpenAI
from vllm import LLM

# Import our MockOpenAI server
from zeroband.inference.openai_wrapper import MockOpenAIServer


def test_integration():
    print("=== Testing Verifiers Environment Integration ===")

    # 1. Initialize vLLM
    print("\n1. Initializing vLLM...")
    llm = LLM(
        model="willcb/Qwen2.5-0.5B-Reverse-SFT",  # Small model for testing
        max_model_len=8192,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    # 2. Start MockOpenAI server
    print("\n2. Starting MockOpenAI server...")
    mock_config = SimpleNamespace(batch_size=64, max_wait_time=0.5, model_name="willcb/Qwen2.5-0.5B-Reverse-SFT")
    server = MockOpenAIServer(llm, tokenizer, mock_config)
    server.start(port=8001)

    # 3. Create OpenAI client
    print("\n3. Creating OpenAI client...")
    client_config = server.get_client_config(port=8001)
    client = OpenAI(base_url=client_config["base_url"], api_key=client_config["api_key"])

    # 4. Initialize environment (exact same as inference)
    print("\n4. Initializing verifiers environment...")
    train_dataset = load_dataset("agentlans/wikipedia-paragraphs", split="train").map(
        lambda x: {"question": x["text"], "answer": x["text"][::-1]}
    )
    parser = vf.XMLParser(["think", "answer"], answer_field="answer")
    system_prompt = f"""Reverse the given text.

    Respond in the following format:
    {parser.get_format_str()}"""

    def lcs_reward_func(completion, answer, **kwargs) -> float:
        """
        LCS ratio of the reversed prompt and the parsed completion.
        """

        def lcs_ratio(x: str, y: str) -> float:
            """
            Return the longest common subsequence ratio of x and y.
            """
            from difflib import SequenceMatcher

            return SequenceMatcher(None, x, y).ratio()

        response = parser.parse_answer(completion) or ""
        return lcs_ratio(response, answer)

    rubric = vf.Rubric(
        funcs=[
            lcs_reward_func,
            parser.get_format_reward_func(),
        ],
        weights=[1.0, 0.2],
    )

    env = vf.SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        client=client,
        model="willcb/Qwen2.5-0.5B-Reverse-SFT",
    )

    # 5. Test generation
    print("\n5. Testing generation...")
    # Get some samples from dataset
    dataset = env.get_dataset()
    if dataset is None:
        print("ERROR: Dataset is None")
        return
    samples = dataset.select(range(20))  # Get first 2 samples

    results = env.generate(
        inputs=samples,
        client=client,
        model="willcb/Qwen2.5-0.5B-Reverse-SFT",
        sampling_args={
            "max_tokens": 4096,
            "temperature": 0.7,
        },
    )
    print("\nResults:")
    print(f"  Prompts: {len(results['prompt'])}")
    print(f"  Completions: {len(results['completion'])}")
    print(f"  Rewards: {results['reward']}")
    # get idx of max reward completion without numpy
    max_reward_idx = max(range(len(results["reward"])), key=lambda i: results["reward"][i])
    print(f"Max reward prompt: {results['prompt'][max_reward_idx]}")
    print(f"Max reward completion: {results['completion'][max_reward_idx]}")
    print(f"Max reward reward: {results['reward'][max_reward_idx]}")

    # 6. Cleanup
    print("\n6. Shutting down server...")
    server.shutdown()

    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_integration()

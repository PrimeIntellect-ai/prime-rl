import sys

from vllm import LLM, SamplingParams


def main():
    llm = LLM(model=sys.argv[1], tensor_parallel_size=int(sys.argv[2]), gpu_memory_utilization=0.5, max_model_len=4096)
    outs = llm.generate(["Reverse this text: hello world, how are you today?"] * 8, SamplingParams(max_tokens=64, temperature=1.0))
    print("GENERATED", len(outs), outs[0].outputs[0].text[:80].replace("\n", " "))


if __name__ == "__main__":
    main()

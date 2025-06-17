import json
import os

from datasets import Dataset, load_dataset

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "baselines/times_torch_L40S.json"), "r") as f:
    TORCH_BASELINES_L40S = json.load(f)

with open(os.path.join(script_dir, "baselines/times_torch_compile_L40S.json"), "r") as f:
    TORCH_COMPILE_BASELINES_L40S = json.load(f)


############################################
# CUDA Prompt
############################################
PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""


def prompt_generate_custom_cuda(arc_src: str, example_arch_src: str, example_new_arch_src: str) -> str:
    prompt = PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    prompt += PROBLEM_INSTRUCTION
    return prompt


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""

    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def prompt_generate_custom_cuda_from_prompt_template(ref_arch_src: str) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_arch = read_file(os.path.join(script_dir, "prompts/example_arch_add.py"))
    example_new_arch = read_file(os.path.join(script_dir, "prompts/example_arch_add_new.py"))

    return prompt_generate_custom_cuda(arch, example_arch, example_new_arch)


if __name__ == "__main__":
    all_data = []

    for level in ["level_1", "level_2", "level_3"]:
        data = load_dataset("ScalingIntelligence/KernelBench")[level]
        data = [d for d in data]

        for i, d in enumerate(data):
            level_id = level.replace("_", "")
            all_data.append(
                dict(
                    problem_id=f"kernelbench_{level}_{d['problem_id']}",
                    task_type="kernelbench",
                    prompt=prompt_generate_custom_cuda_from_prompt_template(d["code"]),
                    verification_info=json.dumps(
                        dict(
                            reference_arch=d["code"],
                            mean_runtime_torch=TORCH_BASELINES_L40S[level_id][d["name"] + ".py"]["mean"],
                            mean_runtime_torch_compile=TORCH_COMPILE_BASELINES_L40S[level_id][d["name"] + ".py"]["mean"],
                        )
                    ),
                    metadata=json.dumps(dict(level=d["level"], name=d["name"])),
                )
            )

    upload = Dataset.from_list(all_data)

    upload.push_to_hub("justus27/kernelbench-genesys")

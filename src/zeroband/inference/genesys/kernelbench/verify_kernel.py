import re
from functools import lru_cache

GPU = "L40S"
GPU_ARCH_MAPPING = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}


@lru_cache(maxsize=1)
def get_app():
    try:
        import modal
    except ImportError:
        raise ImportError("Modal is required for kernel verification.")

    app = modal.App("kernelbench_eval")

    cuda_version = "12.4.0"  # should be no greater than host CUDA version
    flavor = "devel"  #  includes full CUDA toolkit
    operating_sys = "ubuntu22.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"

    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
        .apt_install(
            "git",
            "gcc-10",
            "g++-10",
            "clang",  # note i skip a step
        )
        .pip_install(  # required to build flash-attn
            "anthropic",
            "numpy",
            "openai",
            "packaging",
            "pydra_config",
            "torch==2.5.0",
            "tqdm",
            "datasets",
            "transformers",
            "google-generativeai",
            "together",
            "pytest",
            "ninja",
            "utils",
        )
        .add_local_python_source("kernel_eval_utils")
    )

    @app.cls(image=image)
    class EvalFunc:
        @modal.method()
        def eval_single_sample_modal(self, ref_arch_src, custom_cuda, verbose, gpu_arch):
            from kernel_eval_utils import eval_kernel_against_ref, set_gpu_arch

            set_gpu_arch(gpu_arch)
            return eval_kernel_against_ref(
                ref_arch_src, custom_cuda, verbose=verbose, measure_performance=True, num_correct_trials=5, num_perf_trials=100
            )

    return app, EvalFunc


def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None


def assign_kernel_reward(completion: str, verification_info: dict):
    if "</think>" in completion:
        model_solution = completion.split("</think>")[1]
    else:
        return 0

    custom_cuda = extract_first_code(model_solution, ["python", "cpp"])

    if custom_cuda is None:
        return 0

    reference_arch = verification_info["reference_arch"]

    app, EvalFunc = get_app()
    with app.run():
        kernel_exec_result = EvalFunc.with_options(gpu=GPU)().eval_single_sample_modal.remote(
            reference_arch, custom_cuda, verbose=False, gpu_arch=GPU_ARCH_MAPPING[GPU]
        )

    if not kernel_exec_result.correctness:
        return 0

    baseline_time = verification_info["mean_runtime_torch"]
    runtime = kernel_exec_result.runtime

    # small reward for correct code
    if baseline_time < runtime:
        return 0.1

    # Compute reward: 0.5 if runtime == baseline, up to 1 as runtime -> 0
    reward = 0.5 + 0.5 * min(1.0, baseline_time / runtime)
    reward = min(reward, 1.0)  # Clamp to 1.0 max

    return reward

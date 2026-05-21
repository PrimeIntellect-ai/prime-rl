"""Generate all 144 sweep configs for the full RL vs SFT vs OPD comparison."""

import itertools
import os

SWEEP_DIR = "configs/sweep-v2/matrix"
os.makedirs(SWEEP_DIR, exist_ok=True)

STUDENTS = {
    "qwen06b": "Qwen/Qwen3-0.6B",
    "qwen2b": "Qwen/Qwen3.5-2B",
}

ENVS = {
    "reverse-text": {
        "train_env": '[[orchestrator.train.env]]\nid = "reverse-text"',
        "eval_env": '[[orchestrator.eval.env]]\nid = "reverse-text"',
        "max_completion_tokens_train": 256,
        "max_completion_tokens_eval": 256,
        "eval_num_examples": 200,
        "renderer": 'name = "qwen3"',
        "preserve_thinking": False,
    },
    "gsm8k": {
        "train_env": '[[orchestrator.train.env]]\nid = "math-env"\nname = "gsm8k"\nargs = { dataset_name = "openai/gsm8k", dataset_subset = "main", math_verify_max_workers = 128, math_verify_timeout = 60 }',
        "eval_env": '[[orchestrator.eval.env]]\nid = "math-env"\nname = "gsm8k-test"\nargs = { dataset_name = "openai/gsm8k", dataset_subset = "main", dataset_split = "test", math_verify_max_workers = 128, math_verify_timeout = 60 }',
        "max_completion_tokens_train": 2048,
        "max_completion_tokens_eval": 2048,
        "eval_num_examples": 64,
        "renderer": None,
        "preserve_thinking": True,
    },
    "wordle": {
        "train_env": '[[orchestrator.train.env]]\nid = "wordle"\nargs = { num_eval_examples = 100 }',
        "eval_env": '[[orchestrator.eval.env]]\nid = "wordle"\nargs = { num_eval_examples = 100 }',
        "max_completion_tokens_train": 1024,
        "max_completion_tokens_eval": 1024,
        "eval_num_examples": 100,
        "renderer": 'name = "qwen3"',
        "preserve_thinking": False,
    },
    "alphabet-sort": {
        "train_env": '[[orchestrator.train.env]]\nid = "alphabet-sort"\nargs = { min_turns = 2, max_turns = 3 }',
        "eval_env": '[[orchestrator.eval.env]]\nid = "alphabet-sort"\nargs = { min_turns = 2, max_turns = 3 }',
        "max_completion_tokens_train": 768,
        "max_completion_tokens_eval": 768,
        "eval_num_examples": 64,
        "renderer": 'name = "qwen3"',
        "preserve_thinking": False,
    },
}


# Teachers: (id_suffix, SFT config, OPD config or None)
# OPD teachers must be local vLLM — OPD with PI inference is not possible
def get_teachers(student_key):
    _ = STUDENTS[student_key]  # validate student_key exists
    if student_key == "qwen06b":
        return {
            "self": {
                "model": "Qwen/Qwen3-0.6B",
                "sft_client": "pinference",
                "opd_client": "local:8001",
            },
            "same-fam-4b": {
                "model": "qwen/qwen3-4b-instruct-2507",
                "sft_client": "pinference",
                "opd_client": "local:8001",
                "opd_model": "Qwen/Qwen3-4B-Instruct-2507",
            },
            "same-fam-8b": {
                "model": "qwen/qwen3-8b-instruct-2507",
                "sft_client": "pinference",
                "opd_client": "local:8001",
                "opd_model": "Qwen/Qwen3-8B-Instruct-2507",
            },
            "same-fam-235b": {
                "model": "qwen/qwen3-235b-a22b-instruct-2507",
                "sft_client": "pinference",
                "opd_client": None,  # Too large for local vLLM
            },
            "cross-fam": {
                "model": "openai/gpt-5-mini",
                "sft_client": "pinference",
                "opd_client": None,  # Can't use cross-family for OPD
            },
        }
    else:  # qwen2b
        return {
            "self": {
                "model": "Qwen/Qwen3.5-2B",
                "sft_client": "pinference",
                "opd_client": "local:8001",
            },
            "same-fam-4b": {
                "model": "qwen/qwen3-4b-instruct-2507",
                "sft_client": "pinference",
                "opd_client": "local:8001",
                "opd_model": "Qwen/Qwen3-4B-Instruct-2507",
            },
            "same-fam-8b": {
                "model": "qwen/qwen3-8b-instruct-2507",
                "sft_client": "pinference",
                "opd_client": "local:8001",
                "opd_model": "Qwen/Qwen3-8B-Instruct-2507",
            },
            "same-fam-235b": {
                "model": "qwen/qwen3-235b-a22b-instruct-2507",
                "sft_client": "pinference",
                "opd_client": None,  # Too large for local vLLM
            },
            "cross-fam": {
                "model": "openai/gpt-5-mini",
                "sft_client": "pinference",
                "opd_client": None,
            },
        }


SEEDS = [0, 1, 2]


def make_config(env_name, student_key, mode, teacher_key=None, seed=0):
    env = ENVS[env_name]
    student_model = STUDENTS[student_key]

    name_parts = [env_name, mode]
    if teacher_key:
        name_parts.append(teacher_key)
    name_parts.append(student_key)
    name_parts.append(f"s{seed}")
    run_name = "-".join(name_parts)

    lines = []
    lines.append("max_steps = 100")
    lines.append("seq_len = 2048")
    lines.append("")
    lines.append("[model]")
    lines.append(f'name = "{student_model}"')
    lines.append("")
    lines.append("[wandb]")
    lines.append('project = "sweep-v2-full"')
    lines.append(f'name = "{run_name}"')
    lines.append("")
    lines.append("[orchestrator]")
    lines.append(f'training_mode = "{mode}"')
    lines.append("batch_size = 128")
    lines.append("rollouts_per_example = 8")
    if seed != 0:
        lines.append(f"seed = {seed}")
    lines.append("")

    # Renderer
    if env["renderer"]:
        lines.append("[orchestrator.renderer]")
        lines.append(env["renderer"])
        if env["preserve_thinking"]:
            lines.append("preserve_all_thinking = true")
        lines.append("")
    elif env["preserve_thinking"]:
        lines.append("[orchestrator.renderer]")
        lines.append("preserve_all_thinking = true")
        lines.append("")

    # Train sampling
    lines.append("[orchestrator.train.sampling]")
    if mode == "sft":
        lines.append(f"max_completion_tokens = {max(env['max_completion_tokens_train'], 2048)}")
        lines.append('extra_body = { reasoning_effort = "minimal" }')
    else:
        lines.append(f"max_completion_tokens = {env['max_completion_tokens_train']}")
    lines.append("")

    # Train env
    lines.append(env["train_env"])
    lines.append("")

    # Eval
    lines.append("[orchestrator.eval]")
    lines.append("interval = 10")
    lines.append(f"num_examples = {env['eval_num_examples']}")
    lines.append("")
    lines.append("[orchestrator.eval.sampling]")
    lines.append(f"max_completion_tokens = {env['max_completion_tokens_eval']}")
    lines.append("")
    lines.append(env["eval_env"])
    lines.append("")

    # Teacher config (SFT and OPD only)
    if mode in ("sft", "opd") and teacher_key:
        teachers = get_teachers(student_key)
        teacher = teachers[teacher_key]

        if mode == "opd" and teacher.get("opd_client") is None:
            return None  # Skip cross-family OPD

        if mode == "sft":
            teacher_model = teacher["model"]
            lines.append("[orchestrator.teacher.model]")
            lines.append(f'name = "{teacher_model}"')
            lines.append("")
            lines.append("[orchestrator.teacher.client]")
            lines.append('base_url = ["https://api.pinference.ai/api/v1"]')
            lines.append('api_key_var = "PRIME_API_KEY"')
            lines.append("")
            lines.append("[orchestrator.teacher.client.headers_from_env]")
            lines.append('X-Prime-Team-ID = "PRIME_TEAM_ID"')
        elif mode == "opd":
            opd_model = teacher.get("opd_model", teacher["model"])
            port = teacher["opd_client"].split(":")[1]
            lines.append("[orchestrator.teacher.model]")
            lines.append(f'name = "{opd_model}"')
            lines.append("")
            lines.append("[orchestrator.teacher.client]")
            lines.append(f'base_url = ["http://localhost:{port}/v1"]')
        lines.append("")

    # Trainer
    lines.append("[trainer.optim]")
    lines.append("lr = 3e-6")
    lines.append("")
    lines.append("[ckpt]")
    lines.append("")
    lines.append("[inference]")
    lines.append("gpu_memory_utilization = 0.5")

    return run_name, "\n".join(lines)


# Generate all configs
configs = []
for env_name, student_key, seed in itertools.product(ENVS, STUDENTS, SEEDS):
    # RL (no teacher)
    result = make_config(env_name, student_key, "rl", seed=seed)
    if result:
        configs.append(result)

    # SFT and OPD with each teacher
    teachers = get_teachers(student_key)
    for teacher_key in teachers:
        for mode in ("sft", "opd"):
            result = make_config(env_name, student_key, mode, teacher_key, seed=seed)
            if result:
                configs.append(result)

# Write configs
for name, content in configs:
    path = os.path.join(SWEEP_DIR, f"{name}.toml")
    with open(path, "w") as f:
        f.write(content)

print(f"Generated {len(configs)} configs in {SWEEP_DIR}/")

# Print summary
from collections import Counter

modes = Counter()
for name, _ in configs:
    parts = name.split("-")
    # Find mode
    for p in parts:
        if p in ("rl", "sft", "opd"):
            modes[p] += 1
            break

print(f"  RL: {modes['rl']}, SFT: {modes['sft']}, OPD: {modes['opd']}")

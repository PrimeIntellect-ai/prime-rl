from reasoning_gym.factory import get_score_answer_fn

from zeroband.inference.genesys.format_utils import extract_last_json


def _extract_post_string(completion: str) -> str | None:
    """Return the portion of the completion after ``Final Answer:``."""
    parts = completion.split("</think>", 1)
    if len(parts) == 1:
        return None

    tail = parts[1].strip()
    final_response = tail.rsplit("Final Answer:", 1)
    if len(final_response) == 1:
        return None

    return final_response[1].strip()


def _extract_json_field(completion: str, field: str) -> str | None:
    """Extract ``field`` from the last JSON block following ``</think>``."""
    obj = extract_last_json(completion)
    if obj is None:
        return None
    return str(obj.get(field))


def extract_answer_arc_agi(completion: str, verification_info: dict):
    verification_info["reasoning_gym_entry"]["metadata"]["output"] = tuple(
        [tuple(b) for b in verification_info["reasoning_gym_entry"]["metadata"]["output"]]
    )
    return _extract_post_string(completion), verification_info


def extract_answer_rearc(completion: str, verification_info: dict):
    verification_info["reasoning_gym_entry"]["metadata"]["output"] = tuple(
        [tuple(b) for b in verification_info["reasoning_gym_entry"]["metadata"]["output"]]
    )
    return _extract_post_string(completion), verification_info


def extract_answer_binary_matrix(completion: str, verification_info: dict):
    return _extract_post_string(completion), verification_info


def extract_answer_maze(completion: str, verification_info: dict):
    return _extract_json_field(completion, "num_steps"), verification_info


def extract_answer_quantum_lock(completion: str, verification_info: dict):
    return _extract_json_field(completion, "sequence"), verification_info


def extract_answer_rotten_oranges(completion: str, verification_info: dict):
    return _extract_json_field(completion, "answer"), verification_info


def extract_answer_self_reference(completion: str, verification_info: dict):
    return _extract_json_field(completion, "answer"), verification_info


def extract_answer_bitwise_arithmetic(completion: str, verification_info: dict):
    return _extract_json_field(completion, "answer"), verification_info


ANSWER_PREPROCESS_FUNCTIONS = {
    "arc_agi": extract_answer_arc_agi,
    "rearc": extract_answer_rearc,
    "maze": extract_answer_maze,
    "quantum_lock": extract_answer_quantum_lock,
    "rotten_oranges": extract_answer_rotten_oranges,
    "self_reference": extract_answer_self_reference,
    "bitwise_arithmetic": extract_answer_bitwise_arithmetic,
    "binary_matrix": extract_answer_binary_matrix,
}


def verify_reasoning_gym(completion: str, verification_info: dict) -> float:
    """Score ``completion`` for the reasoning-gym task in ``verification_info``."""
    dataset = verification_info.get("reasoning_gym_dataset")
    if dataset not in ANSWER_PREPROCESS_FUNCTIONS:
        raise KeyError(f"Unsupported reasoning gym dataset: {dataset}")

    answer_preprocess_fn = ANSWER_PREPROCESS_FUNCTIONS[dataset]
    score_answer_fn = get_score_answer_fn(name=dataset)

    answer, verification_info = answer_preprocess_fn(completion, verification_info)
    if answer is None:
        return 0.0

    score = score_answer_fn(answer=answer, entry=verification_info["reasoning_gym_entry"])

    return 1.0 if score == 1 else 0.0

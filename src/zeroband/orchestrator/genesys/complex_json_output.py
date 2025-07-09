from zeroband.orchestrator.genesys.format_utils import extract_last_json


def verify_complex_json_formatting(completion: str, verification_info: dict):
    predicted_json = extract_last_json(completion)

    if not predicted_json:
        return 0

    if verification_info["ground_truth"] == predicted_json:
        return 1

    return 0

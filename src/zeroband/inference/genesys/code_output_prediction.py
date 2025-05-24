from zeroband.inference.genesys.format_utils import extract_last_json


def verify_code_output_prediction(completion: str, verification_info: dict) -> float:
    obj = extract_last_json(completion)
    if obj is None:
        return 0.0

    predicted = obj.get("code_output")
    expected = verification_info.get("code_output")

    if predicted == expected:
        return 1.0
    else:
        return 0.0

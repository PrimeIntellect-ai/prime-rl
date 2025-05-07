import json
import re
from typing import Dict
import traceback

from zeroband.inference.genesys.code_utils import check_correctness


def evaluate_code(completion: str, verification_info: Dict):
    split_response = completion.split("</think>")

    # format error
    if len(split_response) == 1:
        return 0

    code_blocks = re.findall(r"```python\n(.*?)\n```", split_response[1], re.DOTALL)

    if not code_blocks:
        return 0

    solution = code_blocks[-1]
    test_cases = json.loads(verification_info["test_cases"])

    try:
        try:
            res, _ = check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            success = all(map(lambda x: x is True, res))
            if success:
                return 1
            else:
                return 0

        except Exception:
            pass

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})

        metadata_list = []
        res_list = []
        for test_case_id, test_case in enumerate(test_cases_list):
            res, metadata = check_correctness(in_outs=test_case, generation=solution, timeout=5, debug=False)
            try:
                metadata = dict(enumerate(metadata))[0]
            except Exception:
                metadata = {}
            metadata["test_case"] = {}
            metadata["test_case"]["input"] = str(test_case["inputs"][0])
            metadata["test_case"]["output"] = str(test_case["outputs"][0])
            metadata["test_case"]["res"] = str(res)
            metadata_list.append(metadata)
            res_list.extend(res)

            if test_case_id >= 9:
                break

        success = all(map(lambda x: x is True, res_list))
    except Exception:
        traceback.print_exc(10)
        success = False
        metadata_list = None

    if success:
        return 1
    else:
        return 0

from zeroband.logger import get_logger
import re
import os
import uuid
import tempfile
import subprocess


def verify_kod_code(completion, verification_info):
    logger = get_logger()

    split_response = completion.split("</think>")

    # format error
    if len(split_response) == 1:
        return -1

    code_blocks = re.findall(r"```python\n(.*?)\n```", split_response[1], re.DOTALL)

    if not code_blocks:
        return -1

    code_response = code_blocks[-1]

    # Create temporary directory with random UUID
    temp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    os.makedirs(temp_dir)

    try:
        solution_path = os.path.join(temp_dir, "solution.py")
        with open(solution_path, "w") as f:
            f.write(code_response)

        # Write test code
        test_path = os.path.join(temp_dir, "test_solution.py")
        with open(test_path, "w") as f:
            f.write(verification_info["test_code"])

        result = subprocess.run(["pytest", test_path], capture_output=True, text=True)

        return 1 if result.returncode == 0 else -1

    except Exception as e:
        logger.warning(f"Error during kod_code verification: {str(e)}")
        return -1

    finally:
        import shutil

        shutil.rmtree(temp_dir)

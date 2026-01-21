import re
from pathlib import Path

from tests.conftest import ProcessResult


def strip_escape_codes(text: str) -> str:
    """Helper to strip escape codes from text"""
    return re.sub(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def check_no_error(process: ProcessResult, output_dir: Path) -> None:
    """Helper to assert that a process did not error"""
    if process.returncode != 0:
        print("=== Inference Outputs ===")
        with open(output_dir / "logs" / "inference.stdout", "r") as f:
            print(*f.readlines()[-100:], sep="\n")
        print("=== Orchestrator Outputs ===")
        with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
            print(*f.readlines()[-1000:], sep="\n")
    assert process.returncode == 0, f"Process has non-zero return code ({process})"


def check_number_goes_up_or_down(
    lines: list[str],
    start_step: int = 0,
    end_step: int = -1,
    pattern: str = r"Reward:\s*(\d+\.\d{4})",
    go_up: bool = True,
):
    """Helper to assert that a number in lines goes up from a specified start to end step"""
    step_lines = [line for line in lines if "Step" in line]
    assert len(step_lines) > 0, f"No step lines found in output ({lines})"
    try:
        start_step_line = step_lines[start_step]
    except IndexError:
        start_step_line = ""
    try:
        end_step_line = step_lines[end_step]
    except IndexError:
        end_step_line = ""
    assert start_step_line, f"Could not find start step {start_step} in output ({lines})"
    assert end_step_line, f"Could not find end step {end_step} in output ({lines})"
    start_step_match = re.search(pattern, start_step_line)
    end_step_match = re.search(pattern, end_step_line)
    assert start_step_match is not None, (
        f"Could not find number for start step {start_step} in line {start_step_line} ({lines})"
    )
    assert end_step_match is not None, (
        f"Could not find number for end step {end_step} in line {end_step_line} ({lines})"
    )
    start_step_number = float(start_step_match.group(1))
    end_step_number = float(end_step_match.group(1))
    if go_up:
        assert start_step_number < end_step_number, (
            f"Number did not go up. Found start_number={start_step_number} <= end_number={end_step_number} "
            f"(start line: {start_step_line}, end line: {end_step_line}) ({lines})"
        )
    else:
        assert start_step_number > end_step_number, (
            f"Number did not go down. Found start_number={start_step_number} >= end_number={end_step_number} "
            f"(start line: {start_step_line}, end line: {end_step_line}) ({lines})"
        )


def check_number_in_range(
    lines: list[str],
    step: int = -1,
    min_threshold: float | None = 0.0,
    max_threshold: float | None = None,
    pattern: str = r"Reward:\s*(\d+\.\d{4})",
):
    """Helper to assert that a number in step logs is within a threshold"""
    step_lines = [line for line in lines if "SUCCESS" in line and "Step" in line and re.search(pattern, line)]
    assert len(step_lines) > 0, f"No step lines found in output ({lines})"

    # Search for the specific step number in the lines
    if step == -1:
        step_line = step_lines[-1]
    else:
        step_pattern = rf"Step {step}\b"
        matching_lines = [line for line in step_lines if re.search(step_pattern, line)]
        assert len(matching_lines) > 0, f"Could not find step {step} in output ({step_lines})"
        step_line = matching_lines[0]

    step_reward = re.search(pattern, step_line)
    assert step_reward is not None, f"Could not find reward for step {step}. Line: {step_line} ({lines})"
    step_reward = float(step_reward.group(1))
    if min_threshold is not None:
        assert step_reward >= min_threshold, (
            f"Reward did not reach minimum threshold. Found reward={step_reward} < {min_threshold} "
            f"(line: {step_line}) ({lines})"
        )
    if max_threshold is not None:
        assert step_reward <= max_threshold, (
            f"Reward did not reach maximum threshold. Found reward={step_reward} > {max_threshold} "
            f"(line: {step_line}) ({lines})"
        )

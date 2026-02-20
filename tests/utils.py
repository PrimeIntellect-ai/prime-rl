import re
from pathlib import Path

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from tests.conftest import ProcessResult


def prom_sample_key(name: str, **labels: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Build a stable key for a Prometheus sample name and labels."""
    return name, tuple(sorted(labels.items()))


def prom_collect_samples(registry: CollectorRegistry) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Parse all Prometheus samples from a registry into a keyed value map."""
    samples = {}
    for family in text_string_to_metric_families(generate_latest(registry).decode()):
        for sample in family.samples:
            samples[prom_sample_key(sample.name, **sample.labels)] = sample.value
    return samples


def prom_sample_value(samples: dict[tuple[str, tuple[tuple[str, str], ...]], float], name: str, **labels: str) -> float:
    """Return one Prometheus sample value and assert that it exists."""
    key = prom_sample_key(name, **labels)
    assert key in samples
    return samples[key]


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
    step_lines = [line for line in lines if "SUCCESS" in line and "Step" in line and re.search(pattern, line)]
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


def check_metric_in_range(
    lines: list[str],
    metric_name: str,
    pattern: str,
    step: int = -1,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
):
    """Helper to assert that a metric in step logs is within a threshold"""
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

    metric_match = re.search(pattern, step_line)
    assert metric_match is not None, f"Could not find {metric_name} for step {step}. Line: {step_line} ({lines})"
    metric_value = float(metric_match.group(1))
    if min_threshold is not None:
        assert metric_value >= min_threshold, (
            f"{metric_name} did not reach minimum threshold. Found {metric_name}={metric_value} < {min_threshold} "
            f"(line: {step_line}) ({lines})"
        )
    if max_threshold is not None:
        assert metric_value <= max_threshold, (
            f"{metric_name} exceeded maximum threshold. Found {metric_name}={metric_value} > {max_threshold} "
            f"(line: {step_line}) ({lines})"
        )


def check_reward_in_range(
    lines: list[str],
    step: int = -1,
    min_threshold: float | None = 0.0,
    max_threshold: float | None = None,
):
    """Helper to assert that reward in step logs is within a threshold"""
    check_metric_in_range(
        lines,
        metric_name="Reward",
        pattern=r"Reward:\s*(\d+\.\d{4})",
        step=step,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )


def check_avg_reward_in_range(
    lines: list[str],
    last_n_steps: int,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
):
    """Helper to assert that the average reward over the last N steps is within a threshold"""
    pattern = r"Reward:\s*(\d+\.\d{4})"
    step_lines = [line for line in lines if "SUCCESS" in line and "Step" in line and re.search(pattern, line)]
    assert len(step_lines) >= last_n_steps, (
        f"Not enough step lines found. Expected at least {last_n_steps}, got {len(step_lines)}"
    )

    recent_lines = step_lines[-last_n_steps:]
    rewards = []
    for line in recent_lines:
        match = re.search(pattern, line)
        assert match is not None, f"Could not find Reward in line: {line}"
        rewards.append(float(match.group(1)))

    avg_reward = sum(rewards) / len(rewards)
    if min_threshold is not None:
        assert avg_reward >= min_threshold, (
            f"Average reward over last {last_n_steps} steps did not reach minimum threshold. "
            f"Found avg_reward={avg_reward:.4f} < {min_threshold} (rewards={rewards})"
        )
    if max_threshold is not None:
        assert avg_reward <= max_threshold, (
            f"Average reward over last {last_n_steps} steps exceeded maximum threshold. "
            f"Found avg_reward={avg_reward:.4f} > {max_threshold} (rewards={rewards})"
        )


def check_mismatch_kl_in_range(
    lines: list[str],
    step: int = -1,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
):
    """Helper to assert that mismatch KL in step logs is within a threshold"""
    check_metric_in_range(
        lines,
        metric_name="Mismatch KL",
        pattern=r"Mismatch KL:\s*(\d+\.\d{4})",
        step=step,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )

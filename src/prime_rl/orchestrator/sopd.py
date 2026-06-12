"""Semantic on-policy distillation (SOPD).

The teacher is the *current policy itself*, rescored through the student
inference pool with one privileged input: a feedback packet assembled from the
diagnostics the environment computed for this exact rollout (per-assertion
verdicts, reward, stop condition, errors). The resulting per-token logprobs
ship to the trainer as ``teacher_logprobs`` and ride the RL loss via
``trainer.loss.teacher_tau``.
"""

from prime_rl.configs.orchestrator import SopdConfig
from prime_rl.orchestrator.types import TrainRollout
from prime_rl.transport import TrainingSample


def synthesize_feedback(rollout: TrainRollout, max_chars: int) -> str:
    """Assemble the environment's diagnostics for one rollout into a text packet.

    Reads only what the environment already computed: the scalar reward, the
    flattened rubric metrics, and any per-assertion results surfaced via the
    env's ``state_columns`` (e.g. AutomationBench's ``_assertion_results``).
    """
    raw = rollout.raw
    lines: list[str] = [
        "Environment feedback on the attempt below. It was computed after the "
        "attempt finished and was never visible to the assistant.",
        f"reward: {rollout.reward:.3f}",
    ]

    task_completed = raw.get("task_completed_correctly")
    if task_completed is not None:
        lines.append(f"task completed correctly: {'yes' if float(task_completed) == 1.0 else 'no'}")

    assertion_results = raw.get("_assertion_results") or []
    if assertion_results:
        lines.append("success criteria for this task (graded against the final state):")
        for result in assertion_results:
            status = "EXCLUDED" if result.get("excluded") else ("PASS" if result.get("passed") else "FAIL")
            params = ", ".join(f"{key}={value!r}" for key, value in (result.get("params") or {}).items())
            lines.append(f"- [{status}] {result.get('type')}({params})")

    if rollout.is_truncated:
        lines.append("the attempt was truncated before finishing")
    stop_condition = raw.get("stop_condition")
    if stop_condition:
        lines.append(f"stop condition: {stop_condition}")
    if rollout.error is not None:
        lines.append(f"rollout error: {rollout.error.get('error')}")

    return "\n".join(lines)[:max_chars]


def build_sopd_contexts(
    rollouts: list[TrainRollout],
    samples: list[TrainingSample],
    tokenizer,
    config: SopdConfig,
) -> list[list[int]]:
    """Per-sample privileged context token ids, aligned with ``samples``.

    Every sample of a rollout shares that rollout's feedback packet. With
    ``include_diagnostics = False`` (uninformed-teacher ablation) all contexts
    are empty and the teacher sees exactly what the student saw.
    """
    if not config.include_diagnostics:
        return [[] for _ in samples]

    context_ids_by_sample: dict[int, list[int]] = {}
    for rollout in rollouts:
        if not rollout.samples:
            continue
        feedback = synthesize_feedback(rollout, config.max_feedback_chars)
        context_ids = tokenizer.encode(config.feedback_wrapper.format(feedback=feedback), add_special_tokens=False)
        for sample in rollout.samples:
            context_ids_by_sample[id(sample)] = context_ids

    return [context_ids_by_sample[id(sample)] for sample in samples]

#!/usr/bin/env python3
"""Create a Docent analysis plan for Prime RL transcript analysis.

Run with:
    uv run --no-project --with docent-python python scripts/docent/create_prime_analysis_plan.py <collection_id>
"""

from __future__ import annotations

import argparse

FAILURE_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {
            "type": "string",
            "enum": ["success", "failure", "errored", "unclear"],
        },
        "failure_mode": {
            "type": "string",
            "enum": [
                "none",
                "task_misunderstanding",
                "reasoning_error",
                "tool_use_error",
                "format_error",
                "incomplete_solution",
                "degenerate_generation",
                "verification_or_reward_issue",
                "environment_or_infra_error",
                "other",
            ],
        },
        "severity": {
            "type": "number",
            "description": "1 is minor, 5 is decisive for the failed rollout.",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "summary": {
            "type": "string",
            "citations": True,
            "description": "One concise paragraph explaining what happened.",
        },
        "evidence": {
            "type": "string",
            "citations": True,
            "description": "Specific transcript evidence supporting the label.",
        },
        "suggested_fix": {
            "type": "string",
            "description": "A concrete change to try in the environment, prompt, policy, or analysis setup.",
        },
    },
    "required": [
        "outcome",
        "failure_mode",
        "severity",
        "confidence",
        "summary",
        "evidence",
        "suggested_fix",
    ],
    "additionalProperties": False,
}


VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "label_supported": {"type": "boolean"},
        "corrected_failure_mode": {
            "type": "string",
            "enum": [
                "none",
                "task_misunderstanding",
                "reasoning_error",
                "tool_use_error",
                "format_error",
                "incomplete_solution",
                "degenerate_generation",
                "verification_or_reward_issue",
                "environment_or_infra_error",
                "other",
            ],
        },
        "verification_notes": {
            "type": "string",
            "citations": True,
        },
    },
    "required": ["label_supported", "corrected_failure_mode", "verification_notes"],
    "additionalProperties": False,
}


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _where_clause(*, env_name: str | None, reward_below: float | None) -> str:
    clauses = [
        "metadata_json->>'source' = 'prime-rl'",
    ]
    if env_name:
        clauses.append(f"metadata_json->>'env_name' = {_sql_literal(env_name)}")
    if reward_below is not None:
        clauses.append("metadata_json->'scores' ? 'reward'")
        clauses.append(f"CAST(metadata_json->'scores'->>'reward' AS DOUBLE PRECISION) < {reward_below}")
    return " AND ".join(clauses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_id", help="Docent collection containing Prime RL AgentRuns.")
    parser.add_argument("--env-name", help="Restrict the plan to one Prime RL env_name.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum runs to classify in the first pass.")
    parser.add_argument(
        "--reward-below",
        type=float,
        default=1.0,
        help="Classify rollouts with reward below this value. Pass --include-successes to disable.",
    )
    parser.add_argument(
        "--include-successes",
        action="store_true",
        help="Classify all ingested Prime RL rollouts, not just reward-below candidates.",
    )
    parser.add_argument("--model", default="openai/gpt-5-mini", help="Docent model string for readings.")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Submit and approve Docent jobs immediately instead of leaving the plan for review.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.include_successes:
        args.reward_below = None
    return args


def main() -> None:
    from docent import Docent

    args = parse_args()
    client = Docent()
    where = _where_clause(env_name=args.env_name, reward_below=args.reward_below)

    candidate_runs = client.query(
        args.collection_id,
        f"""
SELECT id AS run,
       metadata_json->>'env_name' AS env_name,
       metadata_json->>'example_id' AS example_id,
       metadata_json->'scores'->>'reward' AS reward
FROM agent_runs
WHERE {where}
ORDER BY id
LIMIT {args.limit}
""",
        name="prime-rl/candidate-runs/v1",
    )

    failure_reading = client.read(
        prompt_template=[
            "You are auditing a Prime RL rollout transcript. The AgentRun may contain one transcript per "
            "trajectory step; each step transcript shows the prompt/context the policy saw and its completion.\n\n",
            "Run metadata: env=",
            candidate_runs.env_name,
            ", example_id=",
            candidate_runs.example_id,
            ", reward=",
            candidate_runs.reward,
            "\n\nTranscript:\n",
            candidate_runs.run.as_type("agent_run"),
            "\n\nClassify the outcome and failure mode. Use direct transcript citations for summary and evidence. "
            "If the reward seems inconsistent with the transcript, choose verification_or_reward_issue.",
        ],
        output_schema=FAILURE_SCHEMA,
        model=args.model,
        collection_id=args.collection_id,
        name="prime-rl/failure-classification/v1",
    )
    failure_reading_ref = _sql_literal(f"{failure_reading}")

    client.query(
        args.collection_id,
        f"""
SELECT
    rr.output->>'outcome' AS outcome,
    rr.output->>'failure_mode' AS failure_mode,
    rr.output->>'confidence' AS confidence,
    COUNT(*) AS n,
    AVG(CAST(rr.output->>'severity' AS DOUBLE PRECISION)) AS avg_severity
FROM reading_results rr
JOIN reading_result_links rrl ON rrl.result_id = rr.id
WHERE rrl.reading_id = {failure_reading_ref}
GROUP BY outcome, failure_mode, confidence
ORDER BY n DESC
""",
        name="prime-rl/failure-mode-aggregate/v1",
    )

    suspicious_results = client.query(
        args.collection_id,
        f"""
SELECT rr.id AS reading_result
FROM reading_results rr
JOIN reading_result_links rrl ON rrl.result_id = rr.id
WHERE rrl.reading_id = {failure_reading_ref}
  AND rr.output->>'outcome' = 'failure'
  AND rr.output->>'confidence' = 'high'
ORDER BY CAST(rr.output->>'severity' AS DOUBLE PRECISION) DESC
LIMIT 50
""",
        name="prime-rl/high-confidence-failures/v1",
    )

    client.read(
        prompt_template=[
            "Verify this Prime RL failure classification. Check whether the cited evidence actually supports "
            "the label, and correct the failure mode if needed.\n\n",
            suspicious_results.reading_result.as_type("reading_result"),
        ],
        output_schema=VERIFIER_SCHEMA,
        model=args.model,
        collection_id=args.collection_id,
        name="prime-rl/failure-classification-verifier/v1",
    )

    client.flush(auto_approve=args.auto_approve)
    print("Created Docent analysis plan:")
    print(f"  collection_id: {args.collection_id}")
    print("  candidate query: prime-rl/candidate-runs/v1")
    print("  reading: prime-rl/failure-classification/v1")
    print("  aggregate query: prime-rl/failure-mode-aggregate/v1")
    print("  verifier reading: prime-rl/failure-classification-verifier/v1")


if __name__ == "__main__":
    main()

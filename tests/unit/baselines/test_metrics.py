import pytest

from prime_rl.baselines.metrics import pass_at_k_unbiased, summarize_records


def test_pass_at_k_unbiased():
    assert pass_at_k_unbiased(5, 0, 1) == 0.0
    assert pass_at_k_unbiased(5, 5, 3) == 1.0
    assert pass_at_k_unbiased(3, 1, 5) is None
    assert pass_at_k_unbiased(5, 1, 1) == pytest.approx(0.2)


def test_summarize_records_grouped_success_counts():
    records = [
        {"example_id": "a", "trial_index": 0, "correct": False, "input_tokens": 1, "output_tokens": 2},
        {"example_id": "a", "trial_index": 1, "correct": True, "input_tokens": 1, "output_tokens": 2},
        {"example_id": "a", "trial_index": 2, "correct": False, "input_tokens": 1, "output_tokens": 2},
        {"example_id": "b", "trial_index": 0, "correct": True, "input_tokens": 1, "output_tokens": 2},
        {"example_id": "b", "trial_index": 1, "correct": True, "input_tokens": 1, "output_tokens": 2},
        {"example_id": "b", "trial_index": 2, "correct": False, "input_tokens": 1, "output_tokens": 2},
    ]

    summary = summarize_records(records, ks=(1, 3))

    assert summary["num_examples"] == 2
    assert summary["num_rollouts"] == 6
    assert summary["successes"] == 3
    assert summary["mean_sample_accuracy"] == 0.5
    assert summary["single_shot_accuracy"] == 0.5
    assert summary["pass"]["1"]["successes_at_k_histogram"] == {"0": 1, "1": 1}
    assert summary["pass"]["1"]["pass_at_k"] == pytest.approx(0.5)
    assert summary["pass"]["1"]["prefix_pass_at_k"] == 0.5
    assert summary["pass"]["3"]["pass_at_k"] == 1.0
    assert summary["pass"]["3"]["prefix_pass_at_k"] == 1.0
    assert summary["pass"]["3"]["success_rate_at_k_mean"] == 0.5
    assert summary["pass"]["3"]["all_pass_at_k"] == 0.0
    assert summary["pass"]["3"]["successes_at_k_histogram"]["1"] == 1
    assert summary["pass"]["3"]["successes_at_k_histogram"]["2"] == 1
    assert summary["tokens"]["total_tokens"] == 18


def test_pass_at_k_uses_unbiased_estimator_not_ordered_prefix():
    records = [
        {"example_id": "a", "trial_index": 0, "correct": False},
        {"example_id": "a", "trial_index": 1, "correct": False},
        {"example_id": "a", "trial_index": 2, "correct": True},
        {"example_id": "a", "trial_index": 3, "correct": False},
    ]

    summary = summarize_records(records, ks=(1, 2))

    assert summary["pass"]["1"]["pass_at_k"] == pytest.approx(0.25)
    assert summary["pass"]["1"]["prefix_pass_at_k"] == 0.0
    assert summary["pass"]["2"]["pass_at_k"] == pytest.approx(0.5)
    assert summary["pass"]["2"]["prefix_pass_at_k"] == 0.0


def test_summarize_records_posterior_success_count_distribution_is_answer_class_based():
    records = [
        {
            "example_id": "q1",
            "trial_index": 0,
            "correct": False,
            "parsed_answer": "alpha",
            "posterior_correct": 0.7,
        },
        {
            "example_id": "q1",
            "trial_index": 1,
            "correct": False,
            "parsed_answer": "alpha",
            "posterior_correct": 0.7,
        },
        {
            "example_id": "q1",
            "trial_index": 2,
            "correct": False,
            "parsed_answer": "beta",
            "posterior_correct": 0.2,
        },
        {
            "example_id": "q2",
            "trial_index": 0,
            "correct": False,
            "parsed_answer": "gamma",
            "posterior_correct": 0.1,
        },
        {
            "example_id": "q2",
            "trial_index": 1,
            "correct": False,
            "parsed_answer": "delta",
            "posterior_correct": 0.1,
        },
        {
            "example_id": "q2",
            "trial_index": 2,
            "correct": False,
            "parsed_answer": "delta",
            "posterior_correct": 0.1,
        },
    ]

    summary = summarize_records(records, ks=(3,))

    posterior = summary["pass"]["3"]["posterior"]
    assert posterior["pass_at_k"] == pytest.approx(0.3605263158)
    assert posterior["successes_at_k_mean"] == pytest.approx(0.6885964912)
    assert posterior["success_rate_at_k_mean"] == pytest.approx(0.2295321637)
    distribution = posterior["successes_at_k_distribution"]
    assert distribution["0"] == pytest.approx(0.6394736842)
    assert distribution["1"] == pytest.approx(0.0324561404)
    assert distribution["2"] == pytest.approx(0.3280701754)
    assert distribution["3"] == pytest.approx(0.0)

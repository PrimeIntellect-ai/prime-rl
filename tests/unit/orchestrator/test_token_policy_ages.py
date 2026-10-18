from types import SimpleNamespace

from verifiers.v1.types import WeightVersionSpan

from prime_rl.orchestrator.utils import token_policy_ages


def test_token_policy_ages_counts_labelled_tokens_only():
    spans = [
        WeightVersionSpan(version="default", start=0, end=4),
        WeightVersionSpan(version="7", start=4, end=10),
        WeightVersionSpan(version="9", start=10, end=12),
    ]
    trace = SimpleNamespace(nodes=[SimpleNamespace(weight_versions=None), SimpleNamespace(weight_versions=spans)])

    # Batch 11 trains on v10: 6 tokens of age 3 and 2 tokens of age 1.
    assert token_policy_ages(trace, 11) == (20, 8, 3)

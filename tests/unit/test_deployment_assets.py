from pathlib import Path


def test_chart_does_not_duplicate_prime_runtime_image():
    assert not (Path(__file__).parents[2] / "Dockerfile.dynamo").exists()

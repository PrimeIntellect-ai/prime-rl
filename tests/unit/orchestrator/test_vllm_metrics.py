from prime_rl.orchestrator.vllm_metrics import parse_prometheus_text_sums


def test_parse_prometheus_text_parses_counters_and_labels():
    text = """
# HELP vllm:generated_tokens_total Total generated tokens
# TYPE vllm:generated_tokens_total counter
vllm:generated_tokens_total{model="m",worker="0"} 10
vllm:generated_tokens_total{model="m",worker="1"} 7
vllm:prompt_tokens_total 3
vllm:gpu_kv_cache_usage_percent 42
"""
    sums, series_counts = parse_prometheus_text_sums(text)

    assert sums["vllm:generated_tokens_total"] == 17
    assert series_counts["vllm:generated_tokens_total"] == 2
    assert sums["vllm:prompt_tokens_total"] == 3
    assert sums["vllm:gpu_kv_cache_usage_percent"] == 42


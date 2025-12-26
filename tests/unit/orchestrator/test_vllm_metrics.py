from prime_rl.orchestrator.vllm_metrics import parse_prometheus_text


def test_parse_prometheus_text_parses_counters_and_labels():
    text = """
# HELP vllm:generated_tokens_total Total generated tokens
# TYPE vllm:generated_tokens_total counter
vllm:generated_tokens_total{model="m",worker="0"} 10
vllm:generated_tokens_total{model="m",worker="1"} 7
vllm:prompt_tokens_total 3
vllm:gpu_kv_cache_usage_percent 42
"""
    m = parse_prometheus_text(text)

    assert "vllm:generated_tokens_total" in m
    assert sum(v for _, v in m["vllm:generated_tokens_total"]) == 17
    assert m["vllm:prompt_tokens_total"][0][1] == 3
    assert m["vllm:gpu_kv_cache_usage_percent"][0][1] == 42


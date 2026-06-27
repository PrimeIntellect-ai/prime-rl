from pathlib import Path

import pytest

from scripts.summarize_vllm_throughput import main, summarize


def _write_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                (
                    "(APIServer pid=1) INFO 06-24 22:20:09 [loggers.py:271] "
                    "Engine 000: Avg prompt throughput: 33.2 tokens/s, "
                    "Avg generation throughput: 16768.0 tokens/s, Running: 256 reqs, "
                    "Waiting: 4343 reqs, GPU KV cache usage: 33.4%, "
                    "Prefix cache hit rate: 5.3%"
                ),
                (
                    "(APIServer pid=1) INFO 06-24 22:20:19 [loggers.py:271] "
                    "Engine 000: Avg prompt throughput: 0.0 tokens/s, "
                    "Avg generation throughput: 10881.9 tokens/s, Running: 256 reqs, "
                    "Waiting: 3624 reqs, GPU KV cache usage: 28.7%, "
                    "Prefix cache hit rate: 5.3%"
                ),
                (
                    "(APIServer pid=1) INFO 06-24 22:20:29 [loggers.py:271] "
                    "Engine 000: Avg prompt throughput: 116.1 tokens/s, "
                    "Avg generation throughput: 15088.6 tokens/s, Running: 256 reqs, "
                    "Waiting: 4339 reqs, GPU KV cache usage: 38.7%, "
                    "Prefix cache hit rate: 5.3%"
                ),
            ]
        )
    )


def test_summarize_defaults_to_exact_prompt_zero(tmp_path: Path) -> None:
    log_path = tmp_path / "node_0.log"
    _write_log(log_path)

    summary = summarize(
        path=log_path,
        label="native",
        first_n=4,
        running=256,
        waiting=None,
        min_waiting=0,
        min_kv_cache_pct=25.0,
        max_kv_cache_pct=35.0,
        prompt_tokens_s=0.0,
        min_prompt_tokens_s=None,
        max_prompt_tokens_s=None,
    )

    assert summary.matching_points == 1
    assert summary.first_n_mean_generation_tokens_s == 10881.9
    assert summary.matching[0].stamp == "06-24 22:20:19"


def test_summarize_prompt_range_is_explicit(tmp_path: Path) -> None:
    log_path = tmp_path / "node_0.log"
    _write_log(log_path)

    summary = summarize(
        path=log_path,
        label="native",
        first_n=4,
        running=256,
        waiting=None,
        min_waiting=4000,
        min_kv_cache_pct=25.0,
        max_kv_cache_pct=35.0,
        prompt_tokens_s=0.0,
        min_prompt_tokens_s=None,
        max_prompt_tokens_s=50.0,
    )

    assert summary.matching_points == 1
    assert summary.first_n_mean_generation_tokens_s == 16768.0
    assert summary.matching[0].prompt_tokens_s == 33.2


def test_main_fails_underpowered_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "node_0.log"
    _write_log(log_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_vllm_throughput.py",
            "--min-waiting",
            "4000",
            "--min-kv-cache-pct",
            "25.0",
            "--max-kv-cache-pct",
            "35.0",
            "--min-matching-points",
            "2",
            f"native={log_path}",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2

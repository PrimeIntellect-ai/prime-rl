from prime_rl.utils.health import Rule, RunHealth


def test_run_health_rules_fire_against_recent_history_and_clear():
    health = RunHealth(
        (
            Rule("collapse", r"entropy", "drop", factor=0.2, floor=1e-3),
            Rule("spike", r"err", "rise", factor=2.0, floor=0.1),
        )
    )
    for _ in range(10):
        assert health.check({"entropy": 1.0, "err": 0.05})["health/warnings"] == 0
    assert health.check({"entropy": 0.1, "err": 0.2}) == {
        "health/collapse": 1.0,
        "health/spike": 0.0,
        "health/warnings": 1.0,
    }
    out = health.check({"entropy": 0.1, "err": 0.3})
    assert out == {"health/collapse": 1.0, "health/spike": 1.0, "health/warnings": 2.0}
    assert health.check({"entropy": 1.0, "err": 0.05})["health/warnings"] == 0

from prime_rl.configs.orchestrator import CircuitBreakerConfig
from prime_rl.orchestrator.breaker import CircuitBreaker


def test_circuit_breaker_opens_probes_and_recovers():
    breaker = CircuitBreaker(CircuitBreakerConfig(failures=3, cooldown=10, max_cooldown=15))
    for _ in range(2):
        assert breaker.record(False, "SandboxError: 503", now=0) is None
    assert breaker.record(False, "SandboxError: 503", now=0) == "open"
    assert not breaker.allows(now=5)

    # One probe after the cooldown; a failed probe reopens with a doubled, capped cooldown.
    assert breaker.allows(now=10)
    breaker.on_dispatch()
    assert not breaker.allows(now=10)
    assert breaker.record(False, "SandboxError: 503", now=10) == "open"
    assert breaker.cooldown == 15
    assert not breaker.allows(now=24)

    # An unresponsive env server holds the probe back until it answers again.
    breaker.server_down = True
    assert not breaker.allows(now=30)
    breaker.server_down = False
    assert breaker.allows(now=30)
    breaker.on_dispatch()
    assert breaker.record(True, None, now=31) == "closed"
    assert breaker.allows(now=31) and breaker.cooldown == 10 and breaker.failures == 0

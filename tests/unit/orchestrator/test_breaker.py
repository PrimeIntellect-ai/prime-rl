import uuid

from prime_rl.configs.orchestrator import CircuitBreakerConfig
from prime_rl.orchestrator.breaker import CircuitBreaker


def test_circuit_breaker_opens_probes_and_recovers():
    breaker = CircuitBreaker(CircuitBreakerConfig(failures=3, cooldown=10, max_cooldown=15))
    groups = [uuid.uuid4() for _ in range(3)]
    # Every member of one broken group fails: that counts once.
    for _ in range(8):
        assert breaker.record(False, "SandboxError: 503", 0, groups[0], 0) is None
    assert breaker.record(False, "SandboxError: 503", 0, groups[1], 0) is None
    assert breaker.record(False, "SandboxError: 503", 1, groups[2], 0) == "open"
    assert not breaker.allows(now=5)

    # One probe after the cooldown; a failed probe reopens with a doubled, capped cooldown.
    assert breaker.allows(now=11)
    breaker.on_dispatch()
    assert not breaker.allows(now=11)
    # Episodes started before the breaker opened neither count as the probe nor close it.
    assert breaker.record(False, "SandboxError: 503", 12, groups[0], 0) is None
    assert breaker.record(True, None, 12, groups[0], 0) is None
    assert breaker.record(False, "SandboxError: 503", 12, uuid.uuid4(), 11) == "open"
    assert breaker.cooldown == 15
    assert not breaker.allows(now=26)

    # An unresponsive env server holds the probe back until it answers again.
    breaker.server_down = True
    assert not breaker.allows(now=32)
    breaker.server_down = False
    assert breaker.allows(now=32)
    breaker.on_dispatch()
    assert breaker.record(True, None, 33, uuid.uuid4(), 32) == "closed"
    assert breaker.allows(now=33) and breaker.cooldown == 10 and not breaker.failed_groups

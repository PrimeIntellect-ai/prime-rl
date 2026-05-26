"""Async-native orchestrator v2.

Replaces the step-driven outer loop of `prime_rl.orchestrator` with an async
pipeline driven by a shared train+eval dispatcher. Opt-in via
``orchestrator.experimental.use_orch_v2 = true``; the legacy orchestrator
remains the default while we parity-test against it.
"""

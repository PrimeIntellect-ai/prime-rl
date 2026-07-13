"""Test-time training (TTT) service — per-rollout LoRA updates at compaction boundaries.

The rollout side (`verifiers.v1.ttt`) POSTs the abandoned branch (exact token ids + loss
mask) to this service, which takes gradient step(s) on the rollout's adapter, saves a
versioned PEFT checkpoint, and (re)loads it into the vLLM deployment.
"""

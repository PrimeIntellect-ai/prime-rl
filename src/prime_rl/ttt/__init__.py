"""Test-time training (TTT) service — per-rollout LoRA updates at compaction boundaries.

The fourth prime-rl process type: the rollout side (`verifiers.v1.ttt`) detects a context
rewrite and POSTs the abandoned branch (exact token ids + loss mask) to this service, which
takes gradient step(s) on the rollout's adapter, saves a versioned PEFT checkpoint, and
(re)loads the adapter into the vLLM deployment. See `trainer.py` for the training core and
`server.py` for the HTTP surface.
"""

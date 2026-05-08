# Concision Gemma ES

This example runs synchronous ES-LoRA on the `concision-gemma` Verifiers environment with
`OpenKing/Gemma-270m-it-non-gated`, a non-gated HF-format mirror of Gemma 3 270M IT.

The concision reward is length-only:

```text
reward = -abs(len(completion) - len(answer))
```

It is useful as a small, fast environment for ES trainer smoke runs and timing breakdowns.

```bash
PYTHONPATH=environments/concision_gemma uv run inference @ examples/concision_gemma/inference.toml
PYTHONPATH=environments/concision_gemma uv run es @ examples/concision_gemma/es.toml
```

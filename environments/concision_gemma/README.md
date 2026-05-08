# concision-gemma

`concision-gemma` is a single-turn Verifiers environment for the concision objective used in the ES/GRPO comparison setup.

- Train split: 2 short prompt/answer pairs
- Eval split: 8 held-out prompt/answer pairs
- Reward: `-abs(len(completion) - len(answer))`
- Metrics: normalized concision reward and completion length

The environment intentionally scores length matching only. It does not score answer correctness.

```bash
PYTHONPATH=environments/concision_gemma uv run python -c "import verifiers as vf; print(vf.load_environment('concision-gemma'))"
```

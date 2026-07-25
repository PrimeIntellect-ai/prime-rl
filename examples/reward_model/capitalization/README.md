# Bradley–Terry reward-model sanity check

This example verifies that the reward-model trainer can learn an intentionally trivial preference: capitalized audit labels are chosen over otherwise identical fully lowercase responses.

Generate the deterministic local JSONL files and train Qwen3-0.6B:

```bash
uv run python examples/reward_model/capitalization/make_data.py
uv run reward-model @ examples/reward_model/capitalization/reward_model.toml
```

Preference datasets use three fields. Each value may be a string or an OpenAI-style message list:

```json
{"prompt": [{"role": "user", "content": "State the label."}], "chosen": [{"role": "assistant", "content": "PASS"}], "rejected": [{"role": "assistant", "content": "pass"}]}
```

The trainer renders the chosen and rejected conversations independently, obtains one scalar reward per sequence, and minimizes `-log sigmoid(reward_chosen - reward_rejected)`. Validation reports loss, pairwise accuracy, and mean reward margin.

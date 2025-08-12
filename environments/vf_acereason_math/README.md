# vf-acereason-math

### Overview
- **Environment ID**: `vf-acereason-math`
- **Short description**: Single-turn math word problems from AceReason-Math with boxed numeric answers and CoT; graded by math-verify reward.
- **Tags**: math, single-turn, think, boxed-answer

### Datasets
- **Primary dataset(s)**: `nvidia/AceReason-Math` (HF) prompts mapped to boxed-answer questions
- **Source links**: Hugging Face Datasets
- **Split sizes**: Uses `train` split

### Task
- **Type**: single-turn
- **Parser**: None (reward uses raw completion text); format may follow think + boxed answer convention
- **Rubric overview**: Uses `math_verify_reward_function` to parse and check boxed answer correctness against ground truth

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-acereason-math
```

Configure model and sampling:

```bash
uv run vf-eval vf-acereason-math \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"solve_rate_field": null, "min_solve_rate": null, "max_solve_rate": null}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Reports are written under `./environments/vf_acereason_math/reports/` and auto-embedded below.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `solve_rate_field` | str or null | `null` | Field name in dataset to filter by solve rate (if present) |
| `min_solve_rate` | float or null | `null` | Minimum solve rate filter |
| `max_solve_rate` | float or null | `null` | Maximum solve rate filter |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if math-verify confirms the boxed answer equals target, else 0.0 |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval vf-acereason-math -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->

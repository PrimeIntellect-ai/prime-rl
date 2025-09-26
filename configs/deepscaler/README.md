# DeepScaler

This is a reproduction of the [DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) experiments. They use RL to train `Deepseek-R1-Distill-Qwen-1.5B` to 43.1% Pass@1 accuracy on AIME2024 (+14.3% improvement over the base model), surpassing the performance of OpenAI’s o1-preview with just 1.5B parameters

## Evals

They evaluate on a series of math benchmarks, including Math500, AIME 2024, AMC 2023, Minerva Math and Olympiad Math. We will focus on `math500` and `aime2024`, as these are already implemented as evaluation benchmarks on the Environment Hub.

```bash
bash scripts/tmux.sh
```

```bash
# Run this in the `Inference` pane
uv run inference --model.name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

```bash
uv run eval \
  --model.name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --environment-ids math500,aime2024 \
  --rollouts-per-example 4,64
```
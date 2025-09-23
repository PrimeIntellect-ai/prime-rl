# Evals

You can eval any [verifiers](https://github.com/willccbb/verifiers) environment against API models, local models and checkpoints from an SFT or RL training using the `eval` entrypoint.

> We recommned using the `vf-eval` entrypoint for evaluating a *single* environment against API models or local models. This is often useful whne bulding an environment. However, if want to evaluate multiple environments in parallel and/ or evaluate a training checkpoint, the PRIME-RL `eval` entrypoint is likely more convenient.

We demonstrate evals by evaluating two common benchmarks [`gpqa`](https://app.primeintellect.ai/dashboard/environments/primeintellect/gpqa) and [`math500`](https://app.primeintellect.ai/dashboard/environments/primeintellect/math500).

To check all available configuration options, run `uv run eval --help`.

### Local Models

To evaluate any HF model, start an inference server with the desired model before running the `eval` command. For example, to evaluate against the `math500` and `aime2025` environments, run the following commands:

```bash
uv run inference --model.name <model-name>
```

```bash
uv run eval \
  --model.name <model-name> \
  --environment-ids math500,aime2025
```

### Checkpoints

To evaluate a SFT or RL checkpoint, start an inference server with the model being the base model that you started training from and specify the directory containing the weight checkpoints with `--weights-dir`. 

```bash
uv run inference --model.name <model-name>
```

```bash
uv run eval \
  --model.name <model-name> \
  --environment-ids math500,aime2025 \
  --weights-dir outputs/weights
```

By default, this will evaluate the base model and all step checkpoints found in the weights directory. To skip evaling the base model, set `--no-eval-base` and to evaluate only specific steps, set `--steps` as a comma-separated list of integers representing the steps to evaluate. For example,

```bash
uv run eval \
  --model.name <model-name> \
  --environment-ids math500,aime2025 \
  --weights-dir outputs/weights \
  --steps 10,20,30,40,50 \
  --no-eval-base
```


### API Models

To evaluate API models, you need to set the API base URL and API key. We will exemplify this with the OpenAI API, but the same principles apply to other inference providers.

First, set the API key as an environment variable.

```bash
export OPENAI_API_KEY=...
```

Then, start evaluation by setting the base URL, the name of the environment variable containing the API key, and the model identifier that is exposed by the API.

```bash
uv run eval \
  --client.base-url https://api.openai.com/v1 \
  --client.api-key-var OPENAI_API_KEY \
  --model.name <model-name> \
  --environment-ids math500,aime2025
```

# Entrypoints

## RL

The main usecase of PRIME-RL is RL training. Three main abstractions facilitate RL training: the **orchestrator**, the **trainer**, and the **inference** service.

![Architecture](assets/architecture.png)

### Orchestrator

The orchestrator is a lightweight CPU process that handles the core data and scheduling logic, serving as an intermediary between the trainer and inference service with bidirectional relays. In one direction, it collects rollouts from the inference server, assembles them into packed batches, and dispatches them to the trainer; in the other direction, it relays updated model weights from the trainer to the inference service. The orchestrator utilizes `verifiers` environments to abstract multi-turn rollout generation and scoring, leveraging async OpenAI-compatible inference clients.

### Trainer

The trainer is responsible for producing an updated policy model given rollouts and advantages. We use FSDP2 as the backend with compatibility for any HuggingFace (HF) model. For some models we also provide custom implementations, mostly for performance reasons. FSDP shards model parameters, gradients, and optimizer states, allowing training large models with data parallelism and minimal GPU memory footprint. We support a variety of popular training objectives, such as GRPO, GSPO, OPO, and RLOO. The trainer is inspired by [`torchtitan`](https://github.com/pytorch/torchtitan) and relies on native PyTorch features to implement advanced parallelism techniques, such as tensor, context or expert parallelism.

### Inference

The inference service in its simplest form is a standard OpenAI-compatible server with a vLLM backend. The API specification is extended with two custom endpoints to enable updating the server with the latest policy: `update_weights` is used to reload model weights from a HF-compatible checkpoint on disk, and `reload_weights is used to reset the weights to the base model in between experiments. Otherwise, we rely on vLLM's optimized kernels, parallelism strategies, and scheduling for fast rollout generation. Given the disaggregated nature of the service architecture, it can be directly extended to include multiple engines with a shared request pool, allowing operation across multiple clusters and straightforward integration of alternative inference engines (e.g. SGLang, Tokasaurus). We also heavily rely on native data parallelism in vLLM (also available in SGLang) for orchestrating the fleet of nodes dedicated to inference. 

### RL

For doing RL training all components need to be started. One can do this manually,

```bash
uv run inference ...
```

```bash
uv run orchestrator ...
```

```bash
uv run trainer ...
```

Or, alternatively on a single node, use the `rl` entrypoint to start all components.

```bash
uv run rl \
    --trainer @ path/to/train.toml \
    --orchestrator @ path/to/orch.toml \
    --inference @ path/to/infer.toml \
    ...
```

For more details on multi-node deployment options, see the [deployment](deployment.md) documentation and see the [examples](examples) for concrete training configurations. To see all available configuration options, run `uv run rl --help`.

## SFT

We provide a fairly straight-forward SFT trainer which is capable of fine-tuning any conversational model on multi-turn conversation with tool calling. It shares a lot of components with the RL trainer, such as the modeling code, parallelism techniques, checkpoint format, logger, etc. which ensures a seemless post-training workflow.

To start an SFT training, you need to prepare a dataset in [prompt-completion format](https://huggingface.co/docs/trl/en/dataset_formats#prompt-completion) (we do not support any other format). Single-turn fine-tuning should be compatible with the chat templates of most models. However, to properly handle loss masking, we require that the tokenizer's chat template satisfies a prefix property: the tokenization of any conversation prefix must be a prefix of the tokenization of the full conversation. For instance, tokenizing message 1 should yield a token sequence that forms a prefix of tokenizing messages 1 and 2, which in turn should be a prefix of tokenizing messages 1, 2, 3, and so forth. An example of a chat template that *does not* satisfy this property is Qwen3's chat template, as it strips away past think sections.

On a single GPU, start the training with the `sft` entrypoint

```bash
uv run sft ...
```

If you have access to multiple GPUs, use [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html) with `--nproc-per-node` to start the training. 

```bash
uv run torchrun --nproc-per-node 8 src/prime_rl/trainer/sft/train.py ...
```

For more details on multi-node deployment options, see the [deployment](deployment.md) documentation and see the [examples](examples) for concrete training configurations. To see all available configuration options, run `uv run sft --help`.

## Evals

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
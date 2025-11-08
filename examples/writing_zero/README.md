# Writing

This example explores a multi-stage process of SFT/RFT in order to recreate the results of the [Writing-Zero](https://www.alphaxiv.org/abs/2506.00103) paper. First, we will train a generative reward model (`GenRM`) which is an LLM that will be used to generate rewards during the Bootstrapped Relative Policy Optimization(`BRPO`). Here are the steps:

1. Use [synthesized data](https://huggingface.co/datasets/dmnsh/w0_sft) to run SFT on base model ([PrimeIntellect/Qwen3-4B](https://huggingface.co/PrimeIntellect/Qwen3-4B))
2. Run GRPO on the previous model using [LitBench](https://app.primeintellect.ai/dashboard/environments/dmnsh001/litbench) environment to finalize the training of our GenRM.
3. Lastly, we train a base model ([PrimeIntellect/Qwen3-4B](https://huggingface.co/PrimeIntellect/Qwen3-4B)) on [`w0-brpo`](https://app.primeintellect.ai/dashboard/environments/dmnsh001/w0-brpo) environment with minor tweaks to the existing prime-rl stack.

For a more thorough explanation about the background, visit this [blog](https://damoonsh.github.io/w/2025/10/01/w0.html).

> The commands in this example were designed to be run on 2 GPUs each with more than 120GB (one trainer and one inference GPU). It is possible to run on less or more GPUs using different deployment strategies. If you run on a different setup, you may need to adjust the start commands.

## GenRM

A base model will first get fune-tuned on synthetic data then a simple GRPO will be ran the fine-tuned model. Given the runtimes, do one step then save the model on huggingface then go to the next step.


### Setup

```bash
uv run wandb login
uv run hf auth login
prime env install dmnsh001/LitBench
prime env install dmnsh001/w0-brpo
```

### SFT

[PrimeIntellect/Qwen3-4B](https://huggingface.co/PrimeIntellect/Qwen3-4B) is fine-tuned on [synthesized data](https://huggingface.co/datasets/dmnsh/w0_sft).

```bash
# In the `Trainer` pane
uv run sft @ examples/writing_zero/GenRM/sft/train.toml \
  --wandb.project ... \
  --wandb.name ... \
  --weights
```

To train on multiple GPUs, run

```bash
# In the `Trainer` pane
uv run torchrun \
  --nproc-per-node ... \
  src/prime_rl/trainer/sft/train.py @ examples/writing_zero/GenRM/sft/train.toml \
  --wandb.project ... \
  --wandb.name ... \
  --weights
```

Make sure to push the model to huggingface:

```bash
uv run hf upload <user>/<model-name> outputs/weights/<step-num>
```

- Enter the correct username, pick your model name and step that you would like to push

### RL

For GRPO, we will load the model from previous step and run the GRPO on it using [LitBench](https://app.primeintellect.ai/dashboard/environments/dmnsh001/litbench) environment:

```bash
# Run this in the `Trainer` pane
uv run rl \
  --trainer @ examples/writing_zero/GenRM/rl/train.toml \
  --orchestrator @ examples/writing_zero/GenRM/rl/orch.toml \
  --inference @ examples/writing_zero/GenRM/rl/infer.toml \
  --model.name ... \
  --wandb.project ... \
  --wandb.name ...
```

Push this model to huggingface as well.

Now our GenRM is ready, save it on huggingface since we will load it for BRPO training next.

## BRPO

Before running BRPO, we will need to make a manual change to `prime-rl` codebase to accommodate BRPO. Go into `src/prime_rl/utils/vf.py` within the `generate_group` function add the argument```python interleave_scoring = False``` to **env.generate**. The function should look like this:

```python
async def generate_group(
    client: AsyncOpenAI,
    env: Environment,
    model_name: str,
    problem: dict,
    rollouts_per_example: int,
    sampling_args: dict,
    semaphore: asyncio.Semaphore | None,
) -> GenerateOutputs:
    """Asynchronously generate and score rollouts for one problem."""
    return await env.generate(
        inputs=Dataset.from_list([problem] * rollouts_per_example),
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        semaphore=semaphore,
        use_tqdm=False,
        interleave_scoring=False # <------- This is new
    )
```

Then re-install the library to ensure changes take effect:

```bash
uv pip install -e .
```

**Note**: Future changes to the codebase might render this change useless or even erroneous. The logic is that we need all rollouts for the prompt to be generated before running the rubric, recreate this logic in updated codebase if needed. For more information read this [blog](https://damoonsh.github.io/w/2025/10/01/w0.html).


Now that the codebase is ready, we need to run two inferences, one for the model getting trained the other one for GenRM model that we previously trained. Please note that given the specs of your environment, you may have to alter `gpu_memory_utilization` in each .toml file for both to be running at the same time.

**GenRM**:
```bash
uv run inference @ examples/writing_zero/BRPO/infer.toml
```
**Model** to be trained:
```bash
uv run inference @ examples/writing_zero/BRPO/rl/infer.toml
```

Then we are ready for the actual training: 
```bash
uv run rl \
  --trainer @ examples/writing_zero/BRPO/rl/train.toml \
  --orchestrator @ examples/writing_zero/BRPO/rl/orch.toml \
  --wandb.project .... \
  --wandb.name ....
```


## Evals

Foe eval, two environments can be used: [RewardBench](https://app.primeintellect.ai/dashboard/environments/primeintellect/reward-bench) and [WritingBench](https://app.primeintellect.ai/dashboard/environments/primeintellect/writing-bench). First make sure to install them both:

```bash
prime env install primeintellect/reward-bench
prime env install primeintellect/writing-bench
```

Start the server for the BRPO model:

```bash
uv run inference --model.name <user-name>/brpo-model-name>
```

The environments are highly adjustable based on the passed arguments, here are example eval args:


[WritingBench](https://app.primeintellect.ai/dashboard/environments/primeintellect/writing-bench) needs a judge model, in the cmd below grok-4-fast hosted on **OpenRouter** is being used (`OPENROUTER_API_KEY` needs to be passed).

```bash
uv run vf-eval writing_bench \
  -b 'http://localhost:8000/v1' -m 'dmnsh/Qwen3-4b-W0-BRPO' \
  -s -n 300 -r 3 -c 256 \
  -a '{
    "judge_model": "x-ai/grok-4-fast",
    "judge_base_url": "https://openrouter.ai/api/v1",
    "judge_api_key_var": "OPENROUTER_API_KEY"
  }'
```
[RewardBench](https://app.primeintellect.ai/dashboard/environments/primeintellect/reward-bench) is rather large since it has versions 1, 2, and multilingual. Here is an example for running it with custom arguments:

```bash
uv run vf-eval reward_bench \
  -b 'http://localhost:8000/v1' \
  -m 'dmnsh/Qwen3-4b-W0-BRPO-v2' \
  -n 300 -r 3 \
  -a '{"version": "2", "exclude": ["Safety", "Math", "Ties"]}'
```

The results for those evals can be summarised in table below:

| Benchmark\Model      | PrimeIntellect/Qwen3-4B | dmnsh/Qwen3-4b-W0-BRPO | Change (%) |
|------------------:|:-----------------------:|:----------------------:|:----------:|
| RewardBench2       | 0.478 | 0.498  | +4.18%    |
|  Writing Bench  | 6.864 | 7.012  | +2.16%    |

Experimentation for different training arguments and more diverse synthesized data is encouraged but the results shows that with minimum training, the BRPO algorithm works. Here is the link to already trained [GenRM](https://huggingface.co/dmnsh/Qwen3-4b-W0-GenRM) and [BRPO-ed-model](https://huggingface.co/dmnsh/Qwen3-4b-W0-BRPO).
# Algorithms

This page covers the RL training signal: algorithms, model runtime keys, loss channels, filters, and how traces become trainer samples.

## Configuration

RL runs select one or more algorithms:

```toml
[orchestrator]

[[orchestrator.algorithms]]
id = "grpo"

[[orchestrator.train.env]]
id = "reverse-text"

[[orchestrator.train.env]]
id = "terminal-env"

[[orchestrator.train.env.algorithms]]
id = "grpo"

[[orchestrator.train.env.algorithms]]
id = "echo"
```

Top-level `orchestrator.algorithms` is the default for every train env. An env can override it with its own `algorithms` tables. Builtin ids run in the prime-rl process; env-owned ids are sent to the env server, which resolves the algorithm in the environment package.

Algorithm config is flat. There is no nested `config` field:

```toml
[[orchestrator.algorithms]]
id = "opd"
model = "reference"
```

## Authoring

Algorithms are classes over `verifiers` traces:

```python
import verifiers.v1 as vf


class LengthGRPOConfig(vf.AlgorithmConfig):
    id: str = "my_env.length_grpo"
    penalty: float = 0.01


class LengthGRPO(vf.Algorithm[LengthGRPOConfig]):
    async def setup(self, models: dict[str, vf.ModelRuntime]) -> None:
        pass

    def loss(self) -> str:
        return "rl"

    async def advantage(self, traces: list[vf.Trace]) -> list[vf.Trace]:
        rewards = [trace.reward - self.config.penalty * trace.completion_len for trace in traces]
        baseline = sum(rewards) / len(rewards)
        for trace, reward in zip(traces, rewards, strict=True):
            value = reward - baseline
            for branch in trace.branches:
                branch.mask = [sampled and not trace.has_error for sampled in branch.sampled_mask]
                branch.advantages = [float(value) if keep else 0.0 for keep in branch.mask]
        return traces
```

Export one algorithm class from the module:

```python
__all__ = ["LengthGRPO"]
```

The contract is intentionally direct:

- `setup(models)` receives live `vf.ModelRuntime` objects keyed by model id.
- `loss()` returns the trainer loss channel string, such as `"rl"` or `"ce"`.
- `advantage(traces)` mutates `branch.advantages: list[float]` and `branch.mask: list[bool]` in place.
- Both branch lists must align exactly with `branch.token_ids`.
- Algorithms receive a completed rollout group as `list[vf.Trace]`; per-rollout behavior is just a loop over that list.

Algorithms that need model logprobs use the runtime clients:

```python
class ReferenceKLConfig(vf.AlgorithmConfig):
    id: str = "my_env.reference_kl"
    model: str = "reference"


class ReferenceKL(vf.Algorithm[ReferenceKLConfig]):
    def __init__(self, config: ReferenceKLConfig) -> None:
        super().__init__(config)
        self.runtime: vf.ModelRuntime | None = None

    async def setup(self, models: dict[str, vf.ModelRuntime]) -> None:
        runtime = models[self.config.model]
        if not isinstance(runtime.client, vf.TrainClient):
            raise TypeError(f"models[{self.config.model!r}] must be token-capable")
        self.runtime = runtime

    async def advantage(self, traces: list[vf.Trace]) -> list[vf.Trace]:
        if self.runtime is None:
            raise RuntimeError("setup() must run before advantage()")
        for trace in traces:
            for branch in trace.branches:
                reference_logprobs = await self.runtime.client.prefill_logprobs(self.runtime.model, branch.token_ids)
                branch.mask = [sampled and not trace.has_error for sampled in branch.sampled_mask]
                branch.advantages = [
                    float(reference_logprob - rollout_logprob) if keep else 0.0
                    for reference_logprob, rollout_logprob, keep in zip(
                        reference_logprobs,
                        branch.logprobs,
                        branch.mask,
                        strict=True,
                    )
                ]
        return traces
```

The trace carries rollout data. It does not carry live clients or client configs. The orchestrator sends model runtime configs to the env server, and the env server materializes cached `vf.ModelRuntime` objects for env-owned algorithms.

## Builtins

Builtins live in `prime_rl.orchestrator.algorithms` and are selected by registry id:

| ID | Loss | Behavior |
|---|---|---|
| `grpo` | `rl` | Group mean baseline: `reward - mean(group rewards)`. |
| `max_rl` | `rl` | Mean-normalized group baseline. |
| `rl` | `rl` | Raw trace reward on sampled tokens. |
| `sft` | `ce` | CE weight `1.0` on sampled tokens. |
| `echo` | `ce` | CE weight `0.1` on unsampled tool-message tokens. |
| `opd` | `rl` | Scores each branch under the configured model key; writes reference-vs-rollout logprob deltas as RL advantages. |
| `opsd` | `rl` | Scores a demonstration-conditioned branch under the configured model key. |

`opd` and `opsd` do not use a special trainer loss. They fold fixed reference-model logprob information into the advantage stream, then train through the normal `rl` loss.

## Models And Actor

The trained model is configured at `orchestrator.model` and is always available to algorithms as `models["policy"]`.

```toml
[model]
name = "PrimeIntellect/Qwen3-0.6B"
```

Train rollouts are generated by `orchestrator.actor`, which defaults to `"policy"`:

```toml
[orchestrator]
actor = "policy"
```

Additional model endpoints are keyed under `orchestrator.models`. The keys are user-defined; `policy` is reserved for `orchestrator.model`.

```toml
[orchestrator]
actor = "reference"

[[orchestrator.algorithms]]
id = "sft"

[orchestrator.models.reference]
name = "Qwen/Qwen3-32B"
tokens = true

[orchestrator.models.reference.client]
base_url = ["http://localhost:8001/v1"]
```

Token-capable models set `tokens = true` and must expose renderer-backed generation, token ids, sampled logprobs, and prefill logprobs. They can be actors and can be used by algorithms such as `opd`.

Generic OpenAI-compatible endpoints leave `tokens = false`. They are available to env code as model clients, but they are not actors for RL training because trainer samples require token ids and sampled logprobs.

## Loss Channels

Trainer samples carry a list of per-token loss channels:

```python
TrainingAdvantage(
    loss="rl",
    values=[0.0, 0.0, 1.2, 1.2],
    mask=[False, False, True, True],
)
```

The packer keeps one sample and carries all needed channels through the same packed forward/backward pass. Missing channels in a packed sample are backfilled with zeros and `False` masks.

The trainer currently has two built-in losses:

- `rl`: the configured RL loss from `[trainer.loss]`. The default uses importance ratios between trainer logprobs and rollout logprobs, plus the configured KL stability term.
- `ce`: masked negative log-likelihood. `values` are CE weights, so `sft` writes `1.0` and `echo` writes fractional weights.

Each loss channel is normalized by its own global token count. Channels may overlap on the same token; their gradients add.

Custom `[trainer.loss] type = "custom"` replaces only the `rl` loss function. The `ce` loss is fixed.

## Filters

Filters run between algorithm execution and trainer shipping. The default pre-batch filters monitor only; the default post-batch filters enforce.

```toml
[[post_batch_filters]]
type = "zero_advantage"
enforce = true
```

`zero_advantage` inspects the rollout's RL advantage values. `gibberish` and `repetition` inspect sampled token ids and logprobs.

## Trace-To-Sample Shape

Each v1 trace can have one or more branches. A branch is a root-to-leaf path through the trace graph and becomes one `TrainingSample` when it has token ids and at least one sampled token.

For each branch:

- `branch.token_ids` becomes `TrainingSample.token_ids`.
- `branch.sampled_mask` becomes the sample mask.
- `branch.logprobs` becomes rollout/inference logprobs.
- Algorithms write branch-level `advantages` and `mask`; the orchestrator routes those into `TrainingAdvantage` channels.

This is why algorithms are branch-aware but not task-type-aware: task-specific data can live in `trace.task` or `trace.info`, while training signals always align to branch tokens.

## Multi-Turn And Renderers

Token-capable rollouts use the `verifiers` train client, which renders prompts through `renderers` and calls a vLLM-compatible `/inference/v1/generate` endpoint. This path records exact token ids, sampled logprobs, message spans, multimodal sidecars, and routed experts when available.

Renderer pools are owned by the env server and cached for the server lifetime. A model runtime exposes the actual `RendererPool` as `models[key].renderer` when the client is renderer-backed. Algorithms that need to render alternate prompts can use that object directly.

Prefill is not a separate algorithm concept. It is just `TrainClient.prefill_logprobs(model, token_ids)` on token-capable clients.

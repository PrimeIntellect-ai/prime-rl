from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prime_rl.configs.orchestrator import TrainEnvConfig, TrainSamplingConfig
from prime_rl.configs.shared import ClientConfig, HeartbeatConfig, SlurmConfig, TrainerLogConfig, WandbConfig
from prime_rl.configs.trainer import GCConfig, LoRAConfig, ModelConfig, TokenizerConfig
from prime_rl.utils.config import BaseConfig


class ESAlgorithmConfig(BaseConfig):
    """Configures the synchronous evolution-strategy update."""

    population_size: Annotated[int, Field(ge=1, description="Number of candidate perturbations per ES step.")] = 32

    candidate_chunk_size: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Number of candidates to materialize and evaluate at once. Larger chunks improve inference "
                "batching but increase LoRA cache and filesystem pressure."
            ),
        ),
    ] = 32

    sigma: Annotated[float, Field(gt=0, description="Standard deviation of the parameter-space perturbations.")] = 1e-3

    lr: Annotated[float, Field(ge=0, description="Learning rate applied to the ES gradient estimate.")] = 1e-4

    mirrored: Annotated[
        bool,
        Field(
            description=(
                "Use mirrored +/- perturbation pairs. When enabled, population_size must be even and the "
                "candidate score is computed from reward_plus - reward_minus."
            )
        ),
    ] = False

    reward_normalization: Annotated[
        Literal["zscore", "centered", "none"],
        Field(description="Normalization applied to one-sided candidate rewards before reconstructing the update."),
    ] = "zscore"

    seed: Annotated[int, Field(description="Base random seed for perturbation and dataset sampling.")] = 42

    keep_candidate_adapters: Annotated[
        bool,
        Field(description="Keep per-candidate adapter directories instead of deleting them after each chunk."),
    ] = False

    @model_validator(mode="after")
    def validate_mirrored_population(self):
        if self.mirrored and self.population_size % 2 != 0:
            raise ValueError("population_size must be even when mirrored=True")
        return self


class ESTrainConfig(BaseConfig):
    """Configures training rollout environments for synchronous ES."""

    env: list[TrainEnvConfig] = [TrainEnvConfig()]

    sampling: TrainSamplingConfig = TrainSamplingConfig()

    examples_per_env: Annotated[
        int,
        Field(
            ge=1,
            description="Number of examples sampled per training environment for each candidate evaluation.",
        ),
    ] = 16

    rollouts_per_example: Annotated[
        int,
        Field(ge=1, description="Number of rollouts per example for each candidate."),
    ] = 1

    num_workers: Annotated[
        int | Literal["auto"],
        Field(description="Default number of worker processes for spawned env servers."),
    ] = "auto"

    max_retries: Annotated[
        int,
        Field(ge=0, description="Default number of retries for failed rollouts."),
    ] = 0

    max_concurrent_rollouts_per_rank: Annotated[
        int,
        Field(
            ge=1,
            description="Maximum number of rollout coroutines a trainer rank runs concurrently.",
        ),
    ] = 256

    @model_validator(mode="after")
    def resolve_env_defaults(self):
        group_sampling = self.sampling.model_dump()
        for env in self.env:
            if "sampling" not in env.model_fields_set:
                env.sampling = TrainSamplingConfig(**group_sampling)
            else:
                merged = group_sampling | env.sampling.model_dump(exclude_unset=True)
                env.sampling = TrainSamplingConfig(**merged)
            if "num_workers" not in env.model_fields_set:
                env.num_workers = self.num_workers
            if "max_retries" not in env.model_fields_set:
                env.max_retries = self.max_retries
        return self

    @model_validator(mode="after")
    def validate_unique_env_names(self):
        env_names = [env.resolved_name for env in self.env]
        duplicates = [n for n in env_names if env_names.count(n) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate training environment names: {set(duplicates)}. Each env must have a unique name."
            )
        return self


class ESCheckpointConfig(BaseConfig):
    """Configures ES state and mean-adapter checkpoints."""

    interval: Annotated[int | None, Field(ge=1, description="Save ES state every N steps.")] = None

    resume_step: Annotated[
        int | None,
        Field(ge=-1, description="Step to resume from. If -1, resume from the latest ES checkpoint."),
    ] = None

    keep_last: Annotated[int | None, Field(ge=1, description="Keep at most this many recent ES checkpoints.")] = None


class ESDeploymentConfig(BaseModel):
    """Configures local synchronous ES launch."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["single_node"] = "single_node"

    num_gpus: Annotated[int, Field(ge=1, description="Number of trainer ranks to launch with torchrun.")] = 1

    gpus_per_node: Annotated[int, Field(ge=1, description="Number of GPUs available on the local node.")] = 8

    @model_validator(mode="after")
    def validate_gpu_count(self):
        if self.num_gpus > self.gpus_per_node:
            raise ValueError(f"num_gpus ({self.num_gpus}) exceeds gpus_per_node ({self.gpus_per_node}).")
        return self


class ESConfig(BaseConfig):
    """Configures the synchronous ES-LoRA trainer."""

    model: ModelConfig = ModelConfig(lora=LoRAConfig())

    tokenizer: TokenizerConfig = TokenizerConfig()

    client: ClientConfig = ClientConfig()

    train: ESTrainConfig = ESTrainConfig()

    algorithm: ESAlgorithmConfig = ESAlgorithmConfig()

    ckpt: ESCheckpointConfig | None = ESCheckpointConfig(interval=10)

    log: TrainerLogConfig = TrainerLogConfig()

    wandb: WandbConfig | None = None

    output_dir: Annotated[
        Path,
        Field(description="Directory to write ES state, adapters, logs, and metrics."),
    ] = Path("outputs/es")

    clean_output_dir: Annotated[
        bool,
        Field(description="Delete output_dir before starting. Refuses to overwrite by default."),
    ] = False

    matmul_precision: Annotated[
        Literal["highest", "high", "medium"],
        Field(
            description=(
                "Precision for float32 matrix multiplications. Use 'highest' on ROCm/AMD GPUs if reduced "
                "precision matmuls corrupt large-vocabulary softmaxes; use 'high' for TF32 on NVIDIA GPUs."
            ),
        ),
    ] = "high"

    max_steps: Annotated[int | None, Field(description="Maximum ES steps to run. If None, run indefinitely.")] = None

    dist_timeout_seconds: Annotated[
        int,
        Field(ge=1, description="Timeout in seconds for torch distributed operations."),
    ] = 600

    gc: Annotated[
        GCConfig | None,
        Field(description="Garbage collection config. Set to null to use Python's default GC behavior."),
    ] = GCConfig()

    heartbeat: Annotated[
        HeartbeatConfig | None, Field(description="Heartbeat config for monitoring training progress.")
    ] = None

    deployment: ESDeploymentConfig = ESDeploymentConfig()

    slurm: Annotated[
        SlurmConfig | None,
        Field(description="Reserved for future SLURM ES launch support. Local torchrun is implemented first."),
    ] = None

    dry_run: Annotated[bool, Field(description="Only validate and dump resolved configs and exit early.")] = False

    @model_validator(mode="after")
    def validate_lora_enabled(self):
        if self.model.lora is None:
            raise ValueError("Synchronous ES currently requires model.lora to be configured.")
        return self

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def validate_slurm_not_supported_yet(self):
        if self.slurm is not None:
            raise ValueError("ES SLURM launch is not implemented yet; run with deployment.type='single_node'.")
        return self

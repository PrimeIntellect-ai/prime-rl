from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from prime_rl.trainer.config import (
    AdamWConfig,
    BenchConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfigType,
    SchedulerConfigType,
    TokenizerConfig,
)
from prime_rl.transport.config import FileSystemTransportConfig, TransportConfigType
from prime_rl.utils.config import HeartbeatConfig, LogConfig, MetricsServerConfig, WandbConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class LossConfig(BaseConfig):
    """Base config for loss."""

    ratio_type: Annotated[Literal["token", "sequence"], Field(description="Type of importance ratio to use.")] = "token"

    token_mask_high: Annotated[
        float, Field(ge=0, description="The high threshold for token importance ratio to mask.")
    ] = 8.0
    token_mask_low: Annotated[
        float, Field(ge=0, description="The low threshold for token importance ratio to mask.")
    ] = 0.125
    sequence_clip_high: Annotated[
        float, Field(ge=0, description="The high threshold for sequence importance ratio to clip.")
    ] = 10.0
    geo_mask_high: Annotated[float, Field(ge=0, description="The high threshold for geo importance ratio to mask.")] = (
        10.0
    )
    geo_mask_low: Annotated[float, Field(ge=0, description="The low threshold for geo importance ratio to mask.")] = 0.1
    sequence_mask_low: Annotated[
        float,
        Field(
            ge=0,
            description="If set, masks entire sequences when any generated token has an importance ratio below this value.",
        ),
    ] = 0.0
    sequence_mask_high: Annotated[
        float,
        Field(
            ge=0,
            description="If set, masks entire sequences when any generated token has an importance ratio above this value.",
        ),
    ] = 100.0

    adv_tau: Annotated[float, Field(ge=0, description="The tau for advantages.")] = 1.0
    teacher_tau: Annotated[float, Field(ge=0, description="The tau for teacher logprobs.")] = 0.0
    kl_tau: Annotated[float, Field(ge=0, description="The tau for KL divergence.")] = 0.0

    @model_validator(mode="after")
    def validate_mask_bounds(self):
        if self.token_mask_low >= self.token_mask_high:
            raise ValueError(
                f"token_mask_low ({self.token_mask_low}) must be less than token_mask_high ({self.token_mask_high})"
            )
        if self.geo_mask_low >= self.geo_mask_high:
            raise ValueError(
                f"geo_mask_low ({self.geo_mask_low}) must be less than geo_mask_high ({self.geo_mask_high})"
            )
        if self.sequence_mask_low >= self.sequence_mask_high:
            raise ValueError(
                f"sequence_mask_low ({self.sequence_mask_low}) must be less than sequence_mask_high ({self.sequence_mask_high})"
            )
        return self


class FakeDataLoaderConfig(BaseConfig):
    """Configures a fake data loader sampling random micro batches for debugging."""

    batch_size: Annotated[int, Field(ge=1)] = 2
    generate_samples: Annotated[
        bool, Field(description="Whether to generate separate samples and pack them into a single micro batch.")
    ] = False


class DataLoaderConfig(BaseConfig):
    """Configures the data loader used for training."""

    fake: Annotated[FakeDataLoaderConfig | None, Field(description="Whether to use a fake data loader.")] = None


class BaseWeightBroadcastConfig(BaseModel):
    """Configures the base weight broadcast."""

    pass


class FileSystemWeightBroadcastConfig(BaseWeightBroadcastConfig):
    """Configures the weight broadcast."""

    type: Literal["filesystem"] = "filesystem"
    save_sharded: Annotated[bool, Field(description="Whether to save the weight checkpoint in sharded format.")] = True
    save_format: Annotated[
        Literal["safetensors", "torch"], Field(description="The format to save the weight checkpoint in.")
    ] = "safetensors"


class NCCLWeightBroadcastConfig(BaseWeightBroadcastConfig):
    """Configures the NCCL broadcast."""

    type: Literal["nccl"] = "nccl"
    host: Annotated[str, Field(description="The host to use for the NCCL broadcast.")] = "localhost"
    port: Annotated[int, Field(description="The port to use for the NCCL broadcast.")] = 29501
    timeout: Annotated[int, Field(description="The timeout in seconds to use for the NCCL broadcast.")] = 1200
    # TODO: Should not be configurable, but auto-inferred
    inference_world_size: Annotated[int, Field(description="The number of GPUs used for inference.")] = 1


WeightBroadcastConfigType: TypeAlias = FileSystemWeightBroadcastConfig | NCCLWeightBroadcastConfig


class RLTrainerConfig(BaseSettings):
    """Configures the RL trainer"""

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The tokenizer configuration
    tokenizer: TokenizerConfig = TokenizerConfig()

    # The data configuration
    data: DataLoaderConfig = DataLoaderConfig()

    # The loss configuration
    loss: LossConfig = LossConfig()

    # The optimizer configuration
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()

    # The learning rate scheduler configuration
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = ConstantSchedulerConfig()

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    weight_broadcast: Annotated[WeightBroadcastConfigType, Field(discriminator="type")] = (
        FileSystemWeightBroadcastConfig()
    )

    rollout_transport: Annotated[TransportConfigType, Field(discriminator="type")] = FileSystemTransportConfig()

    # The logging configuration
    log: LogConfig = LogConfig()

    # The wandb configuration
    wandb: WandbConfig | None = None

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints, weights, rollouts and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs")

    max_steps: Annotated[
        int | None,
        Field(
            description="Maximum number of steps to run training for. If None, will run indefinitely.",
        ),
    ] = None

    max_async_level: Annotated[
        int,
        Field(
            ge=0,
            description="Maximum number of steps that inference can be ahead of training. Determines how 'off-policy' the inference engines can be. Higher values yield better throughput through async execution, but may yield lower performance. If 0, will be fully synchronous.",
        ),
    ] = 1

    memory_profiler_path: Annotated[Path | None, Field(description="Path to write memory profile to.")] = None

    bench: Annotated[
        BenchConfig | None,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 4 and use fake data.",
        ),
    ] = None

    trace_path: Annotated[Path | None, Field(description="Path to write pytorch profiler trace to.")] = None

    dist_timeout_seconds: Annotated[
        int,
        Field(
            description="Timeout in seconds for torch distributed ops. Defaults to 600 seconds.",
        ),
    ] = 600

    heartbeat: Annotated[
        HeartbeatConfig | None, Field(description="The heartbeat config for monitoring training progress.")
    ] = None

    metrics_server: Annotated[
        MetricsServerConfig | None,
        Field(description="Prometheus metrics server config. If set, exposes /metrics endpoint for scraping."),
    ] = None

    max_concurrent_runs: Annotated[
        int,
        Field(
            ge=1,
            description="The maximum number of concurrent runs to allow. If 1, then only one run will be allowed at a time.",
        ),
    ] = 1

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench is not None:
            self.max_steps = 4  # 1 Warmup + 3 Benchmark
            if not self.data.fake:
                self.data.fake = FakeDataLoaderConfig()
            if self.ckpt:  # Do not checkpoint
                self.ckpt = None
        return self

    @model_validator(mode="after")
    def dont_do_massive_traces(self):
        if self.trace_path:
            if self.max_steps is None:
                raise ValueError("Must specify max_steps when tracing")
            if self.max_steps >= 10:
                raise ValueError(
                    "Tracing more than 10 steps is not recommended as your trace will be massive. Remove this line if you really want to trace more steps."
                )
        return self

    @model_validator(mode="after")
    def validate_lora_adapter_saving(self):
        if self.ckpt and self.ckpt.weights and self.ckpt.weights.save_adapter_separately:
            lora_enabled = self.model and self.model.lora
            if not lora_enabled:
                raise ValueError(
                    "save_adapter_separately=True requires LoRA to be enabled. "
                    "Set model.lora or disable save_adapter_separately."
                )
        return self

    @model_validator(mode="after")
    def validate_weight_broadcast_type(self):
        if self.weight_broadcast.type == "nccl" and self.max_async_level != 1:
            raise ValueError("NCCL weight broadcast only works with async level 1")
        return self

    @model_validator(mode="after")
    def validate_opt_and_fsdp_offload(self):
        if self.optim.type == "muon" and self.model.fsdp_cpu_offload:
            raise ValueError("Muon optimizer does not support FSDP CPU offload")
        return self

    @model_validator(mode="after")
    def validate_optim_cpu_offload_single_run(self):
        if self.model.optim_cpu_offload and self.max_concurrent_runs > 1:
            raise ValueError("Optimizer CPU offload is not supported with max_concurrent_runs > 1")
        return self

    @model_validator(mode="after")
    def validate_lora_broadcast(self):
        if self.model.lora is not None and self.weight_broadcast.type == "nccl":
            # TODO: Support this
            raise ValueError("NCCL weight broadcast does not support LoRA yet.")
        return self

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def auto_setup_fused_lm_head_chunk_size(self):
        if self.model.fused_lm_head_chunk_size == "auto":
            self.model.fused_lm_head_chunk_size = 2048

        return self

    @model_validator(mode="after")
    def ep_only_with_custom_impl(self):
        if self.model.ep > 1 and self.model.impl not in ("custom", "auto"):
            raise ValueError("EP is only supported with the custom implementation or auto mode")

        return self

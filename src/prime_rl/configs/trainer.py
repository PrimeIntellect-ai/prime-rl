import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from prime_rl.configs.shared import (
    BaseModelConfig,
    FileSystemTransportConfig,
    HeartbeatConfig,
    LogConfig,
    MetricsServerConfig,
    TransportConfig,
    WandbConfig,
)
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings

# -- Shared trainer configs (used by both SFT and RL trainers) --

AttnImplementation: TypeAlias = Literal["sdpa", "flash_attention_2", "flash_attention_3", "fa4"]

# User-facing name -> internal name. Users set `flash_attention_4` in configs,
# which gets rewritten to `fa4` before pydantic validation.
# We use `fa4` internally because `flash_attention_*` triggers transformers
# to attempt installing a kernel from hub.
_ATTN_ALIASES = {"flash_attention_4": "fa4"}


class ActivationCheckpointConfig(BaseConfig):
    """Configures activation checkpointing."""

    freq: Annotated[
        int,
        Field(
            ge=1,
            description="Applies activation checkpointing to every `freq` layers. Defaults to 1, which will is full activation checkpointing.",
        ),
    ] = 1


class ActivationOffloadingConfig(BaseConfig):
    """Configures the activation offloading."""

    pin_memory: Annotated[bool, Field(description="Whether to pin the offloaded activations to CPU memory.")] = True

    max_inflight_activations: Annotated[
        int,
        Field(
            ge=1,
            description="The maximum number of activations to keep in while offloading further. (More activations means smoother overlap, but more gpu memory usage)",
        ),
    ] = 5


class CompileConfig(BaseConfig):
    """Configures model compilation."""

    fullgraph: Annotated[
        bool,
        Field(description="Whether to compile the transformer blocks with fullgraph."),
    ] = False


class BenchConfig(BaseConfig):
    """Configures benchmark mode."""

    output_json: Annotated[
        Path | None,
        Field(description="Path to write benchmark results as JSON. If not set, only prints to console."),
    ] = None


class LoRAConfig(BaseConfig):
    """Configuration for LoRA (Low-Rank Adaptation)."""

    rank: Annotated[
        int,
        Field(
            ge=1,
            description="Rank of the low-rank decomposition matrices.",
        ),
    ] = 16

    alpha: Annotated[
        float,
        Field(
            ge=0,
            description="LoRA scaling parameter.",
        ),
    ] = 32.0

    dropout: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="LoRA dropout rate.",
        ),
    ] = 0.0

    target_modules: Annotated[
        list[str],
        Field(
            description="Module names or regex patterns for modules to apply LoRA to. Simple names (e.g., 'q_proj') match any component in the module path. Regex patterns match anywhere in the name.",
        ),
    ] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "experts",
    ]

    modules_to_save: Annotated[
        list[str],
        Field(
            description="Module names or regex patterns for modules to keep fully trainable (not freeze). Simple names match any component in the module path. Regex patterns match anywhere in the name.",
        ),
    ] = []


class DebugModelConfig(BaseConfig):
    """Debugging feature around model and distributed training."""

    num_layers: Annotated[
        int | None,
        Field(description="The number of layers in the model."),
    ] = None

    random_init: Annotated[
        bool,
        Field(
            description="Whether to random initialize the model.",
        ),
    ] = False


class ModelConfig(BaseModelConfig):
    """Configures the model for training."""

    seq_len: Annotated[int, Field(description="The sequence length to use for the model.")] = 2048

    attn: Annotated[
        AttnImplementation,
        Field(
            description="The attention implementation to use. When CP is enabled, ring attention uses the matching kernel family (FA2 for flash_attention_2, FA3 for flash_attention_3).",
        ),
    ] = "flash_attention_2"

    compile: Annotated[
        CompileConfig | None,
        Field(
            description="Whether to compile the model using `torch.compile`.",
        ),
    ] = None

    ac: Annotated[
        ActivationCheckpointConfig | None,
        Field(
            description="Whether to apply activation checkpointing to the model. If None, will not apply activation checkpointing.",
        ),
    ] = None

    ac_offloading: Annotated[
        ActivationOffloadingConfig | None,
        Field(
            description="Whether to apply activation offloading to the model. If None, will not apply activation offloading.",
        ),
    ] = None

    fsdp_cpu_offload: Annotated[
        bool,
        Field(
            description="Whether to enable FSDP CPU offloading for parameters, gradients, and optimizer states. When enabled, uses pinned memory for efficient CPU-GPU transfers.",
        ),
    ] = False

    optim_cpu_offload: Annotated[
        bool,
        Field(
            description="Whether to enable optimizer state CPU offloading. Unlike fsdp_cpu_offload, this only moves optimizer states (momentum, variance) to CPU, keeping weights on GPU. This avoids the H2D all-gather overhead while still saving GPU memory.",
        ),
    ] = False

    reshard_after_forward: Annotated[
        bool, Field(description="Whether to reshard the model after each forward pass.")
    ] = True

    dp_replicate: Annotated[
        int,
        Field(
            description="The data parallel dim where model weights are replicated.",
        ),
    ] = 1

    ep: Annotated[
        int,
        Field(
            description="The expert parallelism to use if the model has MoE layers. If 1, then no EP will be used.",
        ),
    ] = 1

    tp: Annotated[
        int,
        Field(
            description="The tensor parallelism size to use. If 1, then no TP will be used.",
        ),
    ] = 1

    cp: Annotated[
        int,
        Field(
            description="The context parallelism size to use. If 1, then no CP will be used.",
        ),
    ] = 1

    impl: Annotated[
        Literal["hf", "custom", "auto"],
        Field(
            description=(
                "Model implementation to use. 'auto' (default) selects 'custom' if supported by the model, "
                "otherwise 'hf'."
            ),
        ),
    ] = "auto"

    optimization_dtype: Annotated[
        Literal["bfloat16", "float32"],
        Field(
            description="The dtype to use for the model optimization.",
        ),
    ] = "float32"

    reduce_dtype: Annotated[
        Literal["bfloat16", "float32"],
        Field(
            description="The dtype to use for the model reduce.",
        ),
    ] = "float32"

    moe_use_grouped_mm: Annotated[
        bool,
        Field(
            description="Whether to use grouped mm for the MoE layers. Require compute capability >= 9.0",
        ),
    ] = True

    freeze_moe_router: Annotated[
        bool,
        Field(
            description="Whether to freeze the MoE router parameters during training.",
        ),
    ] = False

    lora: Annotated[
        LoRAConfig | None,
        Field(
            description="Whether to apply LoRA to the model. If None, will not apply LoRA.",
        ),
    ] = None

    debug: Annotated[
        DebugModelConfig,
        Field(
            description="Debugging feature around model and distributed training.",
        ),
    ] = DebugModelConfig()

    fused_lm_head_chunk_size: Annotated[
        int | Literal["auto", "disabled"],
        Field(
            description=(
                "The chunk size to use for the fused LM head. "
                "Three behaviors: "
                "(1) int >= 512: explicitly set chunk size for fused LM head; "
                "(2) 'auto': auto-enable (RL training auto-sets to 8192); "
                "(3) 'disabled': explicitly disable fused LM head (use vanilla). "
                "Explicitly setting an integer value for this feature isn't supported for SFT training."
            ),
        ),
    ] = "disabled"

    @model_validator(mode="before")
    @classmethod
    def _normalize_attn_alias(cls, data):
        """Rewrite user-facing `flash_attention_4` to internal `fa4` before validation."""
        if isinstance(data, dict) and data.get("attn") in _ATTN_ALIASES:
            data["attn"] = _ATTN_ALIASES[data["attn"]]
        return data

    @model_validator(mode="after")
    def trust_remote_code_only_with_hf(self):
        """Trust remote code only if the model is from HF."""
        if self.trust_remote_code:
            if self.impl not in ("hf", "auto"):
                raise ValueError("Trust remote code is only supported with the HF implementation or auto mode.")
        return self

    @model_validator(mode="after")
    def cp_only_with_flash_attn(self):
        if self.cp > 1 and self.attn not in ["flash_attention_2", "flash_attention_3"]:
            raise ValueError("CP is only supported with flash attention 2 or flash attention 3")
        if self.cp > 1 and self.attn == "flash_attention_3" and self.impl != "custom":
            raise ValueError(
                "CP with flash_attention_3 requires model.impl='custom' "
                "(the FA3 ring-attention kernel is only implemented for the custom model path)"
            )
        return self

    @model_validator(mode="after")
    def ac_offloading_requires_ac(self):
        """Automatically enable activation checkpointing when activation offloading is enabled."""
        if self.ac_offloading is not None and self.ac is None:
            self.ac = ActivationCheckpointConfig()
        return self

    @model_validator(mode="after")
    def cpu_offload_mutual_exclusion(self):
        if self.fsdp_cpu_offload and self.optim_cpu_offload:
            raise ValueError("Cannot enable both fsdp_cpu_offload and optim_cpu_offload. Use one or the other.")
        return self

    @model_validator(mode="after")
    def fused_lm_head_chunk_size_is_valid(self):
        if isinstance(self.fused_lm_head_chunk_size, int):
            low = 512
            warn_threshold = 8192
            if self.fused_lm_head_chunk_size < low:
                raise ValueError(
                    f"Fused LM head chunk size must be at least {low}, got {self.fused_lm_head_chunk_size}"
                )
            if self.fused_lm_head_chunk_size < warn_threshold:
                warnings.warn(
                    f"Fused LM head chunk size is set to {self.fused_lm_head_chunk_size}, which is less than the recommended threshold of {warn_threshold}. This may cause some runs to diverge due to numerical instability in floating point arithmetic."
                )

        return self

    @model_validator(mode="after")
    def flash_attention_4_only_with_custom_impl(self):
        if self.attn == "fa4" and self.impl != "custom":
            raise ValueError("Flash attention 4 is only supported with the custom implementation")
        return self


class TokenizerConfig(BaseConfig):
    """Configuration for the tokenizer."""

    name: Annotated[
        str | None,
        Field(description="The name or path of the tokenizer to use. If None, will use the model's default tokenizer."),
    ] = None

    trust_remote_code: Annotated[
        bool | None,
        Field(
            description="Whether to trust remote code for tokenizer initialization. If None, will use the model's default trust remote code setting.",
        ),
    ] = None

    chat_template: Annotated[
        str | None,
        Field(
            description="The chat template to use for the tokenizer. If None, will use the tokenizer's default chat template."
        ),
    ] = None


class ConstantSchedulerConfig(BaseModel):
    """Configuration for constant learning rate scheduler."""

    type: Literal["constant"] = "constant"


class LinearSchedulerConfig(BaseModel):
    """Configuration for linear learning rate scheduler."""

    type: Literal["linear"] = "linear"

    warmup_steps: Annotated[int, Field(ge=0, description="Number of warmup steps for the learning rate scheduler.")] = (
        10
    )

    decay_steps: Annotated[
        int,
        Field(
            ge=0,
            description="Number of steps to decay the learning rate during the final portion of training.",
        ),
    ] = 10

    min_lr: Annotated[float, Field(ge=0, description="Minimum learning rate to converge to.")] = 0.0


class CosineSchedulerConfig(BaseModel):
    """Configuration for cosine learning rate scheduler."""

    type: Literal["cosine"] = "cosine"

    warmup_steps: Annotated[int, Field(ge=0, description="Number of warmup steps for the learning rate scheduler.")] = (
        10
    )

    min_lr: Annotated[float, Field(ge=0, description="Minimum learning rate to converge to.")] = 0.0


SchedulerConfig: TypeAlias = Annotated[
    ConstantSchedulerConfig | LinearSchedulerConfig | CosineSchedulerConfig, Field(discriminator="type")
]


class BaseOptimizerConfig(BaseModel):
    lr: Annotated[float, Field(ge=0)] = 1e-6
    weight_decay: Annotated[float, Field(ge=0)] = 0.01
    max_norm: Annotated[float, Field(ge=0, description="Maximum gradient norm to clip.")] = 1.0


class SGDConfig(BaseOptimizerConfig):
    type: Literal["sgd"] = "sgd"
    nesterov: bool = True
    momentum: float = 0.9


class AdamWConfig(BaseOptimizerConfig):
    type: Literal["adamw"] = "adamw"

    betas1: Annotated[float, Field(ge=0)] = 0.9
    betas2: Annotated[float, Field(ge=0)] = 0.999


class MuonConfig(BaseOptimizerConfig):
    type: Literal["muon"] = "muon"

    mu: Annotated[float, Field(ge=0, description="Momentum factor for the Muon algorithm.")] = 0.95
    betas1: Annotated[
        float, Field(ge=0, description="Beta1 for the AdamW/Lion sub-optimizer used on non-Muon params.")
    ] = 0.9
    betas2: Annotated[
        float, Field(ge=0, description="Beta2 for the AdamW/Lion sub-optimizer used on non-Muon params.")
    ] = 0.95


OptimizerConfig: TypeAlias = Annotated[SGDConfig | AdamWConfig | MuonConfig, Field(discriminator="type")]


class WeightCheckpointConfig(BaseConfig):
    """Configures saving HF-compatible weight checkpoints."""

    save_sharded: Annotated[
        bool,
        Field(
            description="Whether to save the weight checkpoint in sharded format.",
        ),
    ] = True

    save_format: Annotated[
        Literal["safetensors", "torch"],
        Field(
            description="The format to save the weight checkpoint in.",
        ),
    ] = "safetensors"

    save_adapter_separately: Annotated[
        bool,
        Field(
            description="Whether to save LoRA adapters separately before merging into full model weights.",
        ),
    ] = False


class CheckpointConfig(BaseConfig):
    """Configures checkpointing the full model, optimizer and training state for resuming training."""

    interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Interval at which to save the training checkpoint. If None, will only checkpoint at the end of training.",
        ),
    ] = None

    weights: WeightCheckpointConfig | None = WeightCheckpointConfig()

    resume_step: Annotated[
        int | None,
        Field(
            ge=-1,
            description="Step to resume training from. If None, will start from scratch. If -1, will restart from latest checkpoint available.",
        ),
    ] = None

    keep_last: Annotated[
        int | None,
        Field(
            ge=1,
            description="Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency.",
        ),
    ] = None

    keep_interval: Annotated[
        int | None,
        Field(
            ge=1,
            description="Keep checkpoints at every N steps permanently (e.g., keep_interval=100 keeps step 100, 200, ...). If None, no interval-based keeping.",
        ),
    ] = None

    skip_progress: Annotated[
        bool,
        Field(
            description="Whether to skip loading the progress from checkpoint.",
        ),
    ] = False

    skip_scheduler: Annotated[
        bool,
        Field(
            description="Whether to skip loading the scheduler from checkpoint.",
        ),
    ] = False

    skip_dataloader: Annotated[
        bool,
        Field(
            description="Whether to skip loading the dataloader from checkpoint.",
        ),
    ] = False


# -- RL trainer-specific configs --


class DefaultLossConfig(BaseModel):
    """Config for the default loss."""

    type: Literal["default"] = "default"
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


class SFTLossConfig(BaseModel):
    """Config for SFT-style masked negative log-likelihood loss."""

    type: Literal["sft"] = "sft"


class CustomLossConfig(BaseModel):
    """Config for a custom external loss function."""

    type: Literal["custom"] = "custom"
    import_path: Annotated[str, Field(description="Import path to the loss function (e.g., 'my_module.my_loss')")]
    kwargs: Annotated[dict[str, Any], Field(default_factory=dict, description="Kwargs to pass to the loss function")]


LossConfig: TypeAlias = Annotated[DefaultLossConfig | SFTLossConfig | CustomLossConfig, Field(discriminator="type")]


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


WeightBroadcastConfig: TypeAlias = Annotated[
    FileSystemWeightBroadcastConfig | NCCLWeightBroadcastConfig, Field(discriminator="type")
]


class TrainerConfig(BaseSettings):
    """Configures the RL trainer"""

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The tokenizer configuration
    tokenizer: TokenizerConfig = TokenizerConfig()

    # The data configuration
    data: DataLoaderConfig = DataLoaderConfig()

    # The loss configuration
    loss: LossConfig = DefaultLossConfig()

    # The optimizer configuration
    optim: OptimizerConfig = AdamWConfig()

    # The learning rate scheduler configuration
    scheduler: SchedulerConfig = ConstantSchedulerConfig()

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    weight_broadcast: WeightBroadcastConfig = FileSystemWeightBroadcastConfig()

    rollout_transport: TransportConfig = FileSystemTransportConfig()

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

    enable_router_replay: Annotated[
        bool,
        Field(
            description="Whether to enable router replay. If True, will return routed experts in the batch. This is only supported if `enable_return_routed_experts=True` in the inference config or pass `--enable-return-routed-experts` to vLLM server. This is only supported for custom models.",
        ),
    ] = False

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
            self.model.fused_lm_head_chunk_size = 8192

        return self

    @model_validator(mode="after")
    def ep_only_with_custom_impl(self):
        if self.model.ep > 1 and self.model.impl not in ("custom", "auto"):
            raise ValueError("EP is only supported with the custom implementation or auto mode")

        return self

    @model_validator(mode="after")
    def router_replay_only_with_custom_impl(self):
        if self.enable_router_replay and self.model.impl not in ("custom", "auto"):
            raise ValueError("Router replay is only supported with the custom implementation or auto mode")

        return self

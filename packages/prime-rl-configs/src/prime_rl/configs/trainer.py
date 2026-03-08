import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.configs.shared import (
    BaseModelConfig,
    FileSystemTransportConfig,
    HeartbeatConfig,
    MetricsServerConfig,
    TrainerLogConfig,
    TransportConfig,
    WandbConfig,
)
from prime_rl.utils.config import BaseConfig

# -- Shared trainer configs (used by both SFT and RL trainers) --

AttnImplementation: TypeAlias = Literal["eager", "sdpa", "flash_attention_2", "flash_attention_3", "fa4"]
EPCommBackend: TypeAlias = Literal["torch", "deepep"]

# User-facing name -> internal name. Users set `flash_attention_4` in configs,
# which gets rewritten to `fa4` before pydantic validation.
# We use `fa4` internally because `flash_attention_*` triggers transformers
# to attempt installing a kernel from hub.
_ATTN_ALIASES = {"flash_attention_4": "fa4"}


class GCConfig(BaseConfig):
    interval: int = Field(50, ge=1)
    """Run garbage collection every N training steps. Disables Python's automatic GC so every rank collects together and one slow rank can't stall the others."""


class ActivationCheckpointConfig(BaseConfig):
    mode: Literal["full", "selective"] = "full"
    """``full`` checkpoints whole transformer blocks; ``selective`` checkpoints only the subcomponents listed in ``targets`` inside supported custom decoder layers."""

    freq: int = Field(1, ge=1)
    """Apply activation checkpointing to every N layers."""

    targets: list[str] = ["norm"]
    """Selective checkpoint targets. ``norm`` checkpoints every norm module inside selected layers. ``attn_proj`` checkpoints projection-side attention work outside the kernel (input/output projections, attention-local norms, RoPE, gating, model-specific MLA projection helpers). ``mlp`` checkpoints the entire dense MLP forward (not for MoE). ``mla_up_proj`` checkpoints MLA Q/KV up-projection where supported. ``routed_experts`` checkpoints routed expert compute in MoE layers (including LatentMoE). ``linear_attn`` checkpoints non-softmax token mixers (NemotronH Mamba, Qwen3.5-MoE GatedDeltaNet, AFMoE sliding-window attention)."""

    @model_validator(mode="after")
    def validate_selective_targets(self):
        self.targets = list(dict.fromkeys(self.targets))
        if self.mode == "selective" and not self.targets:
            raise ValueError("Selective activation checkpointing requires at least one target.")
        return self


class ActivationOffloadingConfig(BaseConfig):
    pin_memory: bool = True
    """Pin offloaded activations to CPU memory."""

    max_inflight_activations: int = Field(5, ge=1)
    """Max activations kept in flight while offloading. More activations smooth overlap at the cost of GPU memory."""


class CompileConfig(BaseConfig):
    fullgraph: bool = False
    """Compile transformer blocks with ``fullgraph=True``."""


class BenchConfig(BaseConfig):
    output_json: Path | None = None
    """Path to write benchmark results as JSON. If unset, results are only printed to the console."""


class LoRAConfig(BaseConfig):
    rank: int = Field(16, ge=1)
    """Rank of the low-rank decomposition matrices."""

    alpha: float = Field(32.0, ge=0)
    """LoRA scaling parameter."""

    dropout: float = Field(0.0, ge=0, le=1)
    """LoRA dropout rate."""

    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "experts",
        "fc1_latent_proj",
        "fc2_latent_proj",
    ]
    """Module names or regex patterns to apply LoRA to. Simple names (e.g. ``q_proj``) match any component in the module path; regex patterns match anywhere in the name. Names unknown to the current model are silently ignored, so defaults cover multiple architectures. NemotronH note: ``experts`` matches NonGatedGroupedExperts inside LatentMoE; ``fc1_latent_proj``/``fc2_latent_proj`` adapt the latent up/down projections. Add ``in_proj``/``out_proj`` to also LoRA Mamba."""

    modules_to_save: list[str] = []
    """Module names or regex patterns to keep fully trainable (not freeze). Same matching rules as ``target_modules``."""


class DebugModelConfig(BaseConfig):
    num_layers: int | None = None
    """Override the number of transformer layers (truncates the model)."""

    random_init: bool = False
    """Randomly initialize the model instead of loading weights."""

    force_balanced_routing: bool = False
    """Replace MoE token-choice routing with a round-robin assignment so every expert sees an equal share. Intended for fake-data smoke tests where untrained routing would otherwise OOM under severe imbalance. Gating scores are still gathered from the override indices so the forward pass stays consistent."""


class ModelConfig(BaseModelConfig):
    seq_len: int = 2048
    """Sequence length the model is trained on."""

    attn: AttnImplementation = "flash_attention_2"
    """Attention implementation. With CP enabled, ring attention uses the matching kernel family (FA2/FA3/FA4)."""

    compile: CompileConfig | None = None
    """Compile the model with ``torch.compile``."""

    ac: ActivationCheckpointConfig | None = None
    """Activation checkpointing configuration. If None, activation checkpointing is disabled."""

    ac_offloading: ActivationOffloadingConfig | None = None
    """Activation offloading configuration. If None, activation offloading is disabled."""

    fsdp_cpu_offload: bool = False
    """Enable FSDP CPU offloading for parameters, gradients, and optimizer states. Uses pinned memory for efficient CPU↔GPU transfers."""

    optim_cpu_offload: bool = False
    """Offload only optimizer states (momentum, variance) to CPU, keeping weights on GPU. Avoids the H2D all-gather overhead of FSDP CPU offload while still saving GPU memory."""

    reshard_after_forward: bool = True
    """Reshard the model after each forward pass."""

    dp_replicate: int = 1
    """Data parallel dim where model weights are replicated."""

    ep: int = 1
    """Expert parallelism degree for MoE layers. 1 disables EP."""

    ep_comm_backend: EPCommBackend = "torch"
    """Communication backend for expert parallelism. ``torch`` uses TorchTitan all-to-all collectives; ``deepep`` uses DeepEP custom kernels."""

    deepep_num_sms: int = Field(20, ge=1)
    """SMs allocated for DeepEP intranode dispatch/combine kernels. Also determines internode RDMA channel count (``num_channels = num_sms / 2``). Lower values leave more SMs for compute; higher values speed up dispatch/combine. The optimal value depends on EP degree and hardware. Only used when ``ep_comm_backend='deepep'``."""

    deepep_token_chunk_size: int | None = Field(None, ge=1)
    """Token chunk size for DeepEP MoE pipelining. When set, DeepEP dispatch for chunk i+1 is launched while experts compute chunk i. Only used when ``ep_comm_backend='deepep'``."""

    cp: int = 1
    """Context parallelism degree. 1 disables CP."""

    cp_style: Literal["ring", "ulysses"] = "ring"
    """CP communication style. ``ring`` uses ring-attention all-gather/reduce-scatter (requires custom kernels per attention type). ``ulysses`` uses all-to-all to redistribute Q/K/V from sequence-sharded to head-sharded, runs vanilla attention locally on the full sequence, then all-to-all back — works out-of-the-box with any attention kernel (softmax FA, linear attention, mamba, etc.)."""

    impl: Literal["hf", "custom", "auto"] = "auto"
    """Model implementation. ``auto`` selects ``custom`` if supported by the model, otherwise ``hf``."""

    optimization_dtype: Literal["bfloat16", "float32"] = "float32"
    """dtype for model optimization."""

    reduce_dtype: Literal["bfloat16", "float32"] = "float32"
    """dtype for gradient/parameter reductions."""

    moe_use_grouped_mm: bool = True
    """Use grouped mm for MoE layers. Requires compute capability ≥ 9.0."""

    fp8: bool = False
    """FP8 training via DeepGEMM. Replaces ``nn.Linear`` with FP8 blockwise linear and uses FP8 grouped GEMM for MoE experts. Requires SM90 (Hopper) GPUs and ``model.impl='custom'``."""

    freeze_moe_router: bool = False
    """Freeze MoE router parameters during training."""

    lora: LoRAConfig | None = None
    """LoRA configuration. If None, LoRA is disabled."""

    debug: DebugModelConfig = DebugModelConfig()
    """Debugging knobs for the model and distributed training."""

    fused_lm_head_token_chunk_size: int | Literal["auto", "disabled"] = "disabled"
    """Flattened token chunk size for the fused LM head. ``int >= 1`` sets the tokens per LM-head chunk explicitly; ``auto`` auto-enables (RL training picks 8192); ``disabled`` uses the vanilla LM head. Integer values aren't supported for SFT training."""

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
        if self.cp > 1 and self.attn not in ["flash_attention_2", "flash_attention_3", "fa4"]:
            raise ValueError("CP is only supported with flash attention 2, flash attention 3, or fa4")
        if self.cp > 1 and self.attn in ("flash_attention_3", "fa4") and self.impl != "custom":
            # Both ring and ulysses route FA3/FA4 through our custom FlashAttention class:
            # ring patches `_compute_attention` with the ring kernel, ulysses patches it with
            # the all-to-all wrapper around the FA3/FA4 kernel. The HF path patches
            # `_flash_attention_forward` which only wraps FA2.
            raise ValueError(
                f"CP with {self.attn} requires model.impl='custom' "
                "(FA3/FA4 paths are only implemented for the custom model attention class)"
            )
        return self

    @model_validator(mode="after")
    def ac_offloading_requires_ac(self):
        """Automatically enable activation checkpointing when activation offloading is enabled."""
        if self.ac_offloading is not None and self.ac is None:
            self.ac = ActivationCheckpointConfig()
        return self

    @model_validator(mode="after")
    def selective_ac_only_with_custom_impl(self):
        if self.ac is not None and self.ac.mode == "selective" and self.impl not in ("custom", "auto"):
            raise ValueError("Selective activation checkpointing requires model.impl='custom' or 'auto'")
        return self

    @model_validator(mode="after")
    def cpu_offload_mutual_exclusion(self):
        if self.fsdp_cpu_offload and self.optim_cpu_offload:
            raise ValueError("Cannot enable both fsdp_cpu_offload and optim_cpu_offload. Use one or the other.")
        return self

    @model_validator(mode="after")
    def flash_attention_4_only_with_custom_impl(self):
        if self.attn == "fa4" and self.impl != "custom":
            raise ValueError("Flash attention 4 is only supported with the custom implementation")
        return self

    @model_validator(mode="after")
    def fp8_only_with_custom_impl(self):
        if self.fp8 and self.impl not in ("custom", "auto"):
            raise ValueError("FP8 training is only supported with model.impl='custom' or 'auto'.")
        return self

    @model_validator(mode="after")
    def validate_ep_comm_backend(self):
        if self.ep_comm_backend == "torch":
            return self

        if self.ep <= 1:
            raise ValueError(f"model.ep_comm_backend='{self.ep_comm_backend}' requires model.ep > 1.")

        return self


class TokenizerConfig(BaseConfig):
    name: str | None = None
    """Tokenizer name or path. If None, the model's default tokenizer is used."""

    trust_remote_code: bool | None = None
    """Trust remote code when initializing the tokenizer. If None, inherits the model's ``trust_remote_code`` setting."""

    chat_template: str | None = None
    """Chat template for the tokenizer. Either a Jinja2 template string or a path to a template file. If None, the tokenizer's default chat template is used."""


class ConstantSchedulerConfig(BaseConfig):
    type: Literal["constant"] = "constant"


class LinearSchedulerConfig(BaseConfig):
    type: Literal["linear"] = "linear"

    warmup_steps: int = Field(10, ge=0)
    """Warmup steps for the learning rate scheduler."""

    decay_steps: int = Field(10, ge=0)
    """Steps to decay the learning rate during the final portion of training."""

    min_lr: float = Field(0.0, ge=0)
    """Minimum learning rate to converge to."""


class CosineSchedulerConfig(BaseConfig):
    type: Literal["cosine"] = "cosine"

    warmup_steps: int = Field(10, ge=0)
    """Warmup steps for the learning rate scheduler."""

    min_lr: float = Field(0.0, ge=0)
    """Minimum learning rate to converge to."""


SchedulerConfig: TypeAlias = Annotated[
    ConstantSchedulerConfig | LinearSchedulerConfig | CosineSchedulerConfig, Field(discriminator="type")
]


class BaseOptimizerConfig(BaseConfig):
    lr: float = Field(1e-6, ge=0)
    """Peak learning rate."""

    weight_decay: float = Field(0.01, ge=0)
    """L2 weight-decay coefficient."""

    max_norm: float | None = Field(1.0, ge=0)
    """Maximum gradient norm to clip to. If None, gradient clipping is disabled."""


class SGDConfig(BaseOptimizerConfig):
    type: Literal["sgd"] = "sgd"

    nesterov: bool = True
    """Use Nesterov momentum."""

    momentum: float = 0.9
    """SGD momentum factor."""


class AdamWConfig(BaseOptimizerConfig):
    type: Literal["adamw"] = "adamw"

    betas1: float = Field(0.9, ge=0)
    """Adam first-moment (β1) decay."""

    betas2: float = Field(0.999, ge=0)
    """Adam second-moment (β2) decay."""


class MuonConfig(BaseOptimizerConfig):
    type: Literal["muon"] = "muon"

    mu: float = Field(0.95, ge=0)
    """Momentum factor for the Muon algorithm."""

    betas1: float = Field(0.9, ge=0)
    """β1 for the AdamW/Lion sub-optimizer used on non-Muon params."""

    betas2: float = Field(0.95, ge=0)
    """β2 for the AdamW/Lion sub-optimizer used on non-Muon params."""


class SignSGDConfig(BaseOptimizerConfig):
    type: Literal["sign_sgd"] = "sign_sgd"


OptimizerConfig: TypeAlias = Annotated[
    SGDConfig | AdamWConfig | MuonConfig | SignSGDConfig, Field(discriminator="type")
]


class WeightCheckpointConfig(BaseConfig):
    save_sharded: bool = True
    """Save the weight checkpoint in sharded format."""

    save_format: Literal["safetensors", "torch"] = "safetensors"
    """Weight checkpoint serialization format."""

    save_adapter_separately: bool = False
    """Save LoRA adapters separately before merging into full model weights."""


class CheckpointConfig(BaseConfig):
    output_dir: Path | None = None
    """Override directory for checkpoints and weights. If set, checkpoints and weight snapshots are written here instead of under the trainer ``output_dir`` — useful for writing large checkpoints to a separate storage volume."""

    interval: int | None = Field(None, ge=1)
    """Interval at which to save the training checkpoint. If None, only checkpoints at the end of training."""

    weights: WeightCheckpointConfig | None = WeightCheckpointConfig()
    """Weight-checkpoint sub-configuration. If None, no HF-compatible weight checkpoints are written."""

    skip_gather_master_weights: bool = False
    """Skip gathering and saving HF-compatible weight checkpoints. Useful for large models where the gather is expensive and only DCP checkpoints are needed."""

    weights_only: bool = False
    """Save only weight checkpoints (no optimizer/scheduler state). Much faster and smaller than full checkpoints, but cannot resume training."""

    resume_step: int | None = Field(None, ge=-1)
    """Step to resume training from. None starts from scratch; ``-1`` restarts from the latest checkpoint available."""

    keep_last: int | None = Field(None, ge=1)
    """Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency."""

    keep_interval: int | None = Field(None, ge=1)
    """Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping."""

    skip_progress: bool = False
    """Skip loading the progress from checkpoint."""

    skip_scheduler: bool = False
    """Skip loading the scheduler from checkpoint."""

    skip_dataloader: bool = False
    """Skip loading the dataloader from checkpoint."""

    skip_optimizer: bool = False
    """Skip loading the optimizer state from checkpoint."""


class DefaultLossConfig(BaseConfig):
    type: Literal["default"] = "default"

    dppo_mask_low: float = Field(0.2, ge=0)
    """Lower DPPO masking threshold."""

    dppo_mask_high: float = Field(0.2, ge=0)
    """Upper DPPO masking threshold."""

    adv_tau: float = Field(1.0, ge=0)
    """Temperature for the advantage term."""

    kl_tau: float = Field(1e-3, ge=0)
    """Temperature for the KL term."""


class CustomLossConfig(BaseConfig):
    type: Literal["custom"] = "custom"

    import_path: str
    """Import path to the loss function (e.g. ``my_module.my_loss``)."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the loss function."""


LossConfig: TypeAlias = Annotated[DefaultLossConfig | CustomLossConfig, Field(discriminator="type")]


class FakeDataLoaderConfig(BaseConfig):
    batch_size: int = Field(2, ge=1)
    """Batch size of the fake data loader."""

    generate_samples: bool = False
    """Generate separate samples and pack them into a single micro-batch instead of using random tensors."""


class DataLoaderConfig(BaseConfig):
    fake: FakeDataLoaderConfig | None = None
    """Use a fake data loader sampling random micro-batches (for debugging)."""


class BaseWeightBroadcastConfig(BaseConfig):
    pass


class FileSystemWeightBroadcastConfig(BaseWeightBroadcastConfig):
    type: Literal["filesystem"] = "filesystem"

    save_sharded: bool = True
    """Save the weight checkpoint in sharded format."""

    save_format: Literal["safetensors", "torch"] = "safetensors"
    """Weight checkpoint serialization format."""


class NCCLWeightBroadcastConfig(BaseWeightBroadcastConfig):
    type: Literal["nccl"] = "nccl"

    host: str = "localhost"
    """Host for the NCCL broadcast rendezvous."""

    port: int = 29501
    """Port for the NCCL broadcast rendezvous."""

    timeout: int = 1200
    """Timeout in seconds for the NCCL broadcast."""

    # TODO: Should not be configurable, but auto-inferred
    inference_world_size: int = 1
    """Number of GPUs used for inference."""

    quantize_in_weight_transfer: bool = False
    """Use kernel-format FP8 quantized NCCL transfer for weight updates. When disabled, uses default HF checkpoint-format transfer."""


WeightBroadcastConfig: TypeAlias = Annotated[
    FileSystemWeightBroadcastConfig | NCCLWeightBroadcastConfig, Field(discriminator="type")
]


class TrainerExperimentalConfig(BaseConfig):
    pass


class TrainerConfig(BaseConfig):
    model: ModelConfig = ModelConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    data: DataLoaderConfig = DataLoaderConfig()

    loss: LossConfig = DefaultLossConfig()
    """Loss config for rl-mode batches. opd and sft batches dispatch to their own loss fns unconditionally and do not read this."""

    optim: OptimizerConfig = AdamWConfig()

    scheduler: SchedulerConfig = ConstantSchedulerConfig()

    ckpt: CheckpointConfig | None = None
    """Full training-state checkpoint configuration (model + optimizer + scheduler). If None, no resume-capable checkpoints are written."""

    weight_broadcast: WeightBroadcastConfig = FileSystemWeightBroadcastConfig()
    """Transport used to broadcast updated weights from trainer to inference."""

    rollout_transport: TransportConfig = FileSystemTransportConfig()
    """Transport used to ship rollouts from orchestrator to trainer."""

    log: TrainerLogConfig = TrainerLogConfig()

    wandb: WandbConfig | None = None

    output_dir: Path = Path("outputs")
    """Directory to write outputs to — checkpoints, weights, rollouts, and logs are written as subdirectories. Should be a persistent directory with enough disk space and unique per experiment running on a single node."""

    matmul_precision: Literal["highest", "high", "medium"] = "high"
    """Precision for float32 matrix multiplications. ``highest`` is full FP32 (required on ROCm/AMD GPUs to avoid catastrophic precision loss in softmax over large vocabularies). ``high`` enables TF32 on NVIDIA GPUs for a speedup with minor precision tradeoff. See ``torch.set_float32_matmul_precision``."""

    max_steps: int | None = None
    """Maximum number of training steps. If None, runs indefinitely."""

    max_async_level: int = Field(1, ge=0)
    """Maximum steps inference can be ahead of training (how off-policy inference can be). Higher values yield better throughput via async execution at the cost of policy lag; ``0`` is fully synchronous."""

    enable_router_replay: bool = False
    """Return routed experts in the batch so the trainer can replay routing. Requires ``enable_return_routed_experts=true`` on the vLLM server (or ``--enable-return-routed-experts``) and is only supported for custom models."""

    memory_profiler_path: Path | None = None
    """Path to write the memory profile to."""

    bench: BenchConfig | None = None
    """Benchmark-mode configuration. When set, ``max_steps`` is forced to 4 and fake data is used."""

    gc: GCConfig | None = GCConfig()
    """Garbage collection config. Disables automatic GC and runs deterministic collections every N steps to avoid stragglers. Set to null to use Python's default GC behavior."""

    trace_path: Path | None = None
    """Path to write the PyTorch profiler trace to."""

    dist_timeout_seconds: int = 600
    """Timeout in seconds for torch distributed ops."""

    heartbeat: HeartbeatConfig | None = None
    """BetterStack heartbeat configuration for monitoring training progress."""

    metrics_server: MetricsServerConfig | None = None
    """Prometheus metrics server configuration. If set, exposes a ``/metrics`` endpoint for scraping."""

    max_concurrent_runs: int = Field(1, ge=1)
    """Maximum number of concurrent runs to allow. If 1, only one run may run at a time."""

    pack_full_step: bool = False
    """When True, wait for all active runs to have data before packing a training step."""

    experimental: TrainerExperimentalConfig = TrainerExperimentalConfig()

    @model_validator(mode="after")
    def deepep_disables_grad_clipping(self):
        if self.model.ep_comm_backend == "deepep" and self.optim.max_norm is not None:
            warnings.warn(
                "Gradient clipping is not compatible with DeepEP. "
                "Automatically setting optim.max_norm to None (disabled).",
                stacklevel=1,
            )
            self.optim.max_norm = None
        return self

    @model_validator(mode="after")
    def vlms_require_bfloat16(self):
        if self.model.vlm is not None and (
            self.model.optimization_dtype != "bfloat16" or self.model.reduce_dtype != "bfloat16"
        ):
            raise ValueError(
                "VLM models must use optimization_dtype='bfloat16' and reduce_dtype='bfloat16' to match vLLM inference."
            )
        return self

    @model_validator(mode="after")
    def vlm_freeze_incompatible_with_lora(self):
        if self.model.vlm is not None and not self.model.vlm.freeze_vision_encoder and self.model.lora is not None:
            raise ValueError(
                "freeze_vision_encoder=false is incompatible with LoRA. "
                "LoRA freezes all non-adapter parameters including the vision encoder."
            )
        return self

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
    def auto_setup_fused_lm_head_token_chunk_size(self):
        if self.model.fused_lm_head_token_chunk_size == "auto":
            self.model.fused_lm_head_token_chunk_size = 8192

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

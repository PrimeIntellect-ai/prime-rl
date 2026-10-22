import math
import re
import uuid
import warnings
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from prime_rl.configs.monitors import MonitorsConfig, TrainMonitorsConfig
from prime_rl.configs.shared import (
    BaseModelConfig,
    ControlConfig,
    EnvVars,
    FileSystemWeightBroadcastConfig,
    HeartbeatConfig,
    ResumeConfig,
    RunConfig,
    SFTDeploymentConfig,
    SingleNodeDeploymentConfig,
    SlurmConfig,
    TrainerLogConfig,
    TransportConfig,
    WeightBroadcastConfig,
    ZMQTransportConfig,
)
from prime_rl.utils.config import BaseConfig, default_output_dir, find_package_resource

# -- Shared trainer configs (used by both SFT and RL trainers) --

AttnImplementation: TypeAlias = Literal["flash_attention_2", "flash_attention_3", "flash_attention_4", "auto"]
DSABackend: TypeAlias = Literal["tilelang", "cudnn_flashmla"]


class GCConfig(BaseConfig):
    interval: int = Field(50, ge=1)
    """Run garbage collection every N training steps. Disables Python's automatic GC so every rank collects together and one slow rank can't stall the others."""


class ActivationCheckpointConfig(BaseConfig):
    mode: Literal["full", "selective"] = "full"
    """Both modes checkpoint whole transformer blocks. ``selective`` additionally retains selected operations."""

    freq: int = Field(1, ge=1)
    """Apply activation checkpointing to every N layers."""

    targets: list[str] | None = None
    """Operator names or namespaces retained in selective mode. ``None`` uses the default targets; an explicit list replaces them."""


class ActivationOffloadingConfig(BaseConfig):
    """Offload tensors saved for backward to pinned CPU memory during the forward pass."""

    max_inflight_activations: int = Field(5, ge=1)
    """Max activations kept on GPU while their offload copies are in flight on the side stream. Higher values give more copy/compute overlap at the cost of GPU memory."""


OffloadMode: TypeAlias = Literal["none", "optimizer", "full"]


class CompileConfig(BaseConfig):
    fullgraph: bool = False
    """Compile transformer blocks with ``fullgraph=True``."""

    mode: Literal["reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs", "lite"] | None = None
    """``torch.compile`` mode. ``reduce-overhead`` records CUDA graphs to cut kernel launch overhead; ``max-autotune`` modes trade longer compile times for tuned kernels (``max-autotune`` also records CUDA graphs, ``max-autotune-no-cudagraphs`` does not). CUDA-graphed layers re-record on new input shapes. ``None`` uses PyTorch's default mode."""


class FusionsConfig(BaseConfig):
    enabled: list[Literal["gate_up", "qkv"]] = ["gate_up", "qkv"]
    """Runtime parameter fusions. ``gate_up`` runs each MoE expert's gate and up projections as one grouped GEMM; ``qkv`` runs attention's q, k and v projections as one GEMM. Only modules that support a fusion are packed, checkpoints keep the canonical parameter names and shapes, and fusions are skipped when LoRA is enabled. Set to ``[]`` to disable."""

    shard_fused_on_dim1: bool = False
    """Experimental. Shard fused 2-D weights along dim 1 under FSDP so that weight loading and checkpointing are zero-copy: the checkpoint reads and writes the fused weights and their optimizer state in place, instead of assembling a full copy of every fused weight on each rank first. Requires the hidden size to be divisible by the FSDP shard mesh size."""


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
    """Module names or regex patterns to apply LoRA to. Simple names (e.g. ``q_proj``) match any component in the module path; regex patterns match anywhere in the name. Names unknown to the current model are silently ignored, so defaults cover multiple architectures. NemotronH note: ``experts`` matches the ReLU² grouped experts; ``fc1_latent_proj``/``fc2_latent_proj`` adapt the latent projections. Add ``in_proj``/``out_proj`` to also LoRA Mamba."""


class DebugModelConfig(BaseConfig):
    num_layers: int | None = None
    """Override the number of transformer layers (truncates the model)."""

    random_init: bool = False
    """Randomly initialize the model instead of loading weights."""

    force_balanced_routing: bool = False
    """Replace MoE token-choice routing with a round-robin assignment so every expert sees an equal share. Intended for fake-data smoke tests where untrained routing would otherwise OOM under severe imbalance. Gating scores are still gathered from the override indices so the forward pass stays consistent."""


MXFP8Recipe: TypeAlias = Literal["mxfp8_rceil", "mxfp8_rceil_wgrad_with_hp"]

_DEFAULT_FP8_IGNORE_PATTERNS: list[str] = [
    "lm_head",
    "router",
    # Use escaped dots — re.search treats `.` as any-char, so the previous
    # "mlp.gate." pattern was also matching dense MLP `mlp.gate_proj` (the
    # trailing `.` was matching `_`). That left the dense MLP gate projection
    # in BF16 on the trainer while inference quantized it to FP8, causing
    # hidden-state drift before the MoE router.
    r"mlp\.gate\.",
    r"shared_expert\.output_gate",  # Qwen3.5 MoE: nn.Linear(hidden, 1, bias=False)
    "eh_proj",
    "weights_proj",
    "in_proj_a",
    "in_proj_b",
]


class FP8Config(BaseConfig):
    type: Literal["fp8"] = "fp8"
    ignore_patterns: list[str] = _DEFAULT_FP8_IGNORE_PATTERNS
    """Dense linear module names excluded from DeepGEMM FP8 replacement."""

    extra_ignore_patterns: list[str] = []
    """Patterns appended to `ignore_patterns`, to exclude model-specific modules while keeping the defaults."""


class MXFP8Config(BaseConfig):
    type: Literal["mxfp8"] = "mxfp8"
    recipe: MXFP8Recipe = "mxfp8_rceil"
    """MXFP8 recipe for dense linear modules."""

    ignore_patterns: list[str] = _DEFAULT_FP8_IGNORE_PATTERNS
    """Dense linear module names excluded from torchao MXFP8 replacement."""

    extra_ignore_patterns: list[str] = []
    """Patterns appended to `ignore_patterns`, to exclude model-specific modules while keeping the defaults."""


QuantizationConfig: TypeAlias = Annotated[FP8Config | MXFP8Config, Field(discriminator="type")]


class MoEComputeConfigBase(BaseConfig):
    apply_to: str | list[Annotated[int, Field(ge=0, strict=True)]] = "all"
    """Model layers to use this backend for: ``"all"``, a percentage such as ``"85%"``,
    or zero-based layer indices such as ``[0, 1, 2]``. Percentages select the first fraction
    of model layers, rounded down. Other expert groups use BF16 compute and transport.
    """

    @field_validator("apply_to")
    @classmethod
    def validate_apply_to(cls, value: str | list[int]) -> str | list[int]:
        if isinstance(value, str) and value != "all":
            if re.fullmatch(r"\d+(?:\.\d+)?%", value) is None or float(value[:-1]) > 100:
                raise ValueError('apply_to must be "all", a percentage from "0%" to "100%", or a list of layer indices')
        return value

    def resolve_layers(self, num_layers: int) -> set[int]:
        if isinstance(self.apply_to, list):
            invalid = [index for index in self.apply_to if index >= num_layers]
            if invalid:
                raise ValueError(
                    f"apply_to layer indices {invalid} are out of range for a model with {num_layers} layers"
                )
            return set(self.apply_to)
        count = num_layers if self.apply_to == "all" else int(num_layers * float(self.apply_to[:-1]) / 100)
        return set(range(count))


class BF16MoEComputeConfig(MoEComputeConfigBase):
    """Run routed experts in bfloat16."""

    type: Literal["bf16"] = "bf16"
    backend: Literal["torch", "sonicmoe"] = "torch"
    """Expert compute implementation."""


class DeepGemmFP8MoEComputeConfig(MoEComputeConfigBase):
    """Run routed-expert grouped GEMMs with DeepGEMM FP8 kernels."""

    type: Literal["deepgemm_fp8"] = "deepgemm_fp8"


class MXFP8MoEComputeConfig(MoEComputeConfigBase):
    """Run routed-expert grouped GEMMs with Prime's vendored MXFP8 implementation."""

    type: Literal["mxfp8"] = "mxfp8"
    recipe: MXFP8Recipe = "mxfp8_rceil"
    """MXFP8 expert-compute recipe."""


MoEComputeConfig: TypeAlias = Annotated[
    BF16MoEComputeConfig | DeepGemmFP8MoEComputeConfig | MXFP8MoEComputeConfig,
    Field(discriminator="type"),
]


class TorchMoEDispatchConfig(BaseConfig):
    """Dispatch and combine routed tokens with torch all-to-all collectives."""

    type: Literal["torch"] = "torch"
    transport: Literal["bf16", "mxfp8"] = "bf16"
    """Wire format for routed activations and their reverse-path gradients."""


class DeepEPMoEDispatchConfig(BaseConfig):
    """Dispatch and combine routed tokens with DeepEP."""

    type: Literal["deepep"] = "deepep"
    num_sms: int = Field(20, ge=1)
    """SMs allocated to DeepEP communication kernels."""

    token_chunk_size: int | None = Field(None, ge=1)
    """Optional chunk size used to pipeline dispatch with local expert compute."""


class MegaMoeMoEDispatchConfig(BaseConfig):
    type: Literal["mega_moe"] = "mega_moe"
    precision: Literal["bf16", "mxfp8", "fp8", "fp8_bwd"] = "bf16"
    """Expert GEMM precision. ``mxfp8``: forward and backward in MXFP8 (SM100). ``fp8``: forward and backward
    in blockwise FP8 (SM90). ``fp8_bwd``: BF16 forward, blockwise FP8 backward (SM90)."""

    max_tokens_per_rank: int | None = Field(None, ge=1)
    """Tokens per rank per MoE call that the symmetric buffer is sized for. Defaults to the micro batch
    (``model.seq_len`` for RL, ``data.micro_batch_size * data.seq_len`` for SFT) and cannot be smaller."""

    num_reserved_sms: int = Field(16, ge=1)
    """SMs left free for concurrent NCCL kernels. ``NCCL_MAX_CTAS`` must be set to at most this value."""


def mega_moe_max_tokens_per_rank(max_tokens_per_rank: int | None, micro_batch_tokens: int) -> int:
    """A rank with more tokens than its buffer fails alone while its peers wait in the kernel, so the
    buffer must hold the full micro batch."""
    if max_tokens_per_rank is None:
        return micro_batch_tokens
    if max_tokens_per_rank < micro_batch_tokens:
        raise ValueError(
            f"model.moe.dispatch.max_tokens_per_rank ({max_tokens_per_rank}) must be at least the micro batch "
            f"({micro_batch_tokens} tokens)."
        )
    return max_tokens_per_rank


MoEDispatchConfig: TypeAlias = Annotated[
    TorchMoEDispatchConfig | DeepEPMoEDispatchConfig | MegaMoeMoEDispatchConfig,
    Field(discriminator="type"),
]


class MoERuntimeConfig(BaseConfig):
    """Independent routed-expert compute and token-dispatch choices."""

    compute: MoEComputeConfig = BF16MoEComputeConfig()
    dispatch: MoEDispatchConfig = TorchMoEDispatchConfig()


class InitConfig(BaseConfig):
    """Random weight initialization: every weight matrix and embedding from N(0, std). Norm weights start at 1,
    biases at 0."""

    std: float = Field(0.02, gt=0)
    """Standard deviation."""

    seed: int = 0
    """Seed for the initialization. The values depend only on the seed, not on the number of GPUs."""


class ModelConfig(BaseModelConfig):
    conversion_dir: Path | None = None
    """Directory for the auto-converted weights (written to a `prime`/`hf` subdirectory). If not set, we write into the model snapshot directory."""

    seq_len: int = 2048
    """Sequence length the model is trained on."""

    attn: AttnImplementation = "auto"
    """Attention implementation. ``auto`` selects FA3 on Hopper (SM90) and FA4 on Blackwell (SM100+)."""

    compile: CompileConfig | None = CompileConfig()
    """Compile the model with ``torch.compile``."""

    fusions: FusionsConfig = FusionsConfig()
    """Runtime parameter fusions, on by default."""

    ac: ActivationCheckpointConfig | None = ActivationCheckpointConfig()
    """Activation checkpointing configuration. If None, activation checkpointing is disabled."""

    ac_offloading: ActivationOffloadingConfig | None = ActivationOffloadingConfig()
    """Activation offloading configuration. If None, activation offloading is disabled."""

    offload: OffloadMode = "optimizer"
    """CPU offloading. ``optimizer`` keeps optimizer states (momentum, variance) in CPU RAM and weights on GPU. ``full`` keeps FP32 masters, optimizer states, and accumulated gradients in CPU RAM and runs each optimizer chunk on CPU as soon as its last gradient arrives, overlapped with backward; it supports AdamW and SignSGD only, disables gradient clipping, and rounds each gradient to BF16 once before the FP32 CPU update. ``none`` keeps everything on GPU."""

    reshard_after_forward: bool = True
    """Reshard the model after each forward pass."""

    dp_replicate: int = 1
    """Data parallel dim where model weights are replicated."""

    ep: int | Literal["auto"] = "auto"
    """Expert parallelism degree for MoE layers. 1 disables EP. ``auto`` resolves for MoE models to the largest degree up to 8 that divides both ``fsdp_island_size = world_size // dp_replicate`` and the expert count and is a multiple of ``cp``, and to 1 for non-MoE models. Set an explicit integer to override."""

    moe: MoERuntimeConfig = MoERuntimeConfig()
    """Routed-expert compute and token-dispatch runtime."""

    cp: int = 1
    """Context parallelism degree. 1 disables CP."""

    optimization_dtype: Literal["bfloat16", "float32"] = "float32"
    """dtype for model optimization."""

    reduce_dtype: Literal["bfloat16", "float32"] = "float32"
    """dtype for gradient/parameter reductions."""

    reduce_scatter_max_input_buffers: int = Field(1, ge=1)
    """Maximum number of FSDP gradient reduce-scatter input buffers kept in flight. Raising it trades memory (about one layer's gradients per extra buffer) for overlap when reduce-scatters are exposed, e.g. MoE without EP or full activation checkpointing."""

    quantization: QuantizationConfig | None = None

    dsa_backend: DSABackend = "cudnn_flashmla"
    """Kernels for the sparse attention of ``glm_moe_dsa`` (DSA sparse MLA, e.g. GLM-5) and ``deepseek_v4``. ``cudnn_flashmla`` (default) runs the FlashMLA sparse prefill forward with the cuDNN frontend CuTe-DSL backward; it needs an SM90, SM100 or SM103 GPU, ``flash-mla`` (installed with the ``gpu`` extra on x86_64), 64 or 128 attention heads and head_dim 512, and setup fails otherwise. ``tilelang`` runs the TileLang forward and backward."""

    dsv4_kv_precision: Literal["bf16", "fp8"] = "fp8"
    """Precision DeepSeek V4 attention reads keys and values at. ``fp8`` rounds them as vLLM's ``fp8_ds_mla`` KV cache stores them, with a straight-through gradient."""

    freeze_moe_router: bool | Literal["auto"] = "auto"
    """Freeze MoE router gate parameters. ``auto`` (default): ``True`` for RL, ``False`` for SFT; a no-op on non-MoE models. Frozen routers get no optimizer state, so resuming a checkpoint saved with a frozen router with ``freeze_moe_router=false`` fails on missing DCP keys."""

    lora: LoRAConfig | None = None
    """LoRA configuration. If None, LoRA is disabled."""

    debug: DebugModelConfig = DebugModelConfig()
    """Debugging knobs for the model and distributed training."""

    fused_lm_head_token_chunk_size: int | Literal["disabled"] = 8192
    """Flattened token chunk size for the fused LM head. ``int >= 1`` sets the tokens per LM-head chunk explicitly; ``disabled`` uses the vanilla LM head. In SFT the fused head computes the summed cross-entropy and its gradients chunk by chunk, holding one chunk's full-vocab logits at a time."""

    @model_validator(mode="after")
    def validate_cp(self):
        if self.cp > 1 and self.attn not in ["flash_attention_2", "flash_attention_3", "flash_attention_4", "auto"]:
            raise ValueError("CP is only supported with flash attention 2, 3, or 4")
        return self

    @model_validator(mode="after")
    def ac_offloading_requires_ac(self):
        """Activation offloading requires activation checkpointing, so ``ac = None`` disables both."""
        if self.ac is None:
            self.ac_offloading = None
        return self

    @model_validator(mode="after")
    def validate_moe_runtime(self):
        if self.ep == 1:
            return self

        compute = self.moe.compute
        dispatch = self.moe.dispatch
        if isinstance(dispatch, DeepEPMoEDispatchConfig):
            if isinstance(compute, MXFP8MoEComputeConfig):
                raise ValueError("MXFP8 expert compute does not support DeepEP dispatch.")
        elif isinstance(dispatch, MegaMoeMoEDispatchConfig):
            pass  # Compute is fused into dispatch; `model.moe.compute` is ignored for these layers.
        elif dispatch.transport == "mxfp8":
            if not isinstance(compute, MXFP8MoEComputeConfig):
                raise ValueError("MXFP8 transport requires model.moe.compute.type='mxfp8'.")
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


StepsOrFraction: TypeAlias = Annotated[int, Field(ge=0)] | Annotated[float, Field(gt=0, lt=1)]


def resolve_steps(value: int | float, max_steps: int | None) -> int:
    """Steps as given, or a fraction of ``max_steps`` rounded down."""
    return math.floor(value * max_steps) if isinstance(value, float) else value


class LinearSchedulerConfig(BaseConfig):
    type: Literal["linear"] = "linear"

    warmup_steps: StepsOrFraction = 10
    """Warmup steps for the learning rate scheduler, or a fraction of ``max_steps``."""

    decay_steps: StepsOrFraction = 10
    """Steps to decay the learning rate during the final portion of training, or a fraction of ``max_steps``."""

    decay_shape: Literal["linear", "cosine", "1-sqrt"] = "linear"
    """Shape of the decay from the peak to ``min_lr``."""

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


def validate_scheduler(scheduler: SchedulerConfig, max_steps: int | None) -> None:
    """Check scheduler phases against max_steps so misconfigurations fail at config time."""
    if isinstance(scheduler, LinearSchedulerConfig):
        if scheduler.warmup_steps == 0 and scheduler.decay_steps == 0:
            raise ValueError(
                "Linear scheduler requires warmup_steps > 0 or decay_steps > 0 (use the constant scheduler instead)"
            )
        if scheduler.decay_steps > 0:
            if max_steps is None:
                raise ValueError("Must specify max_steps when using a linear scheduler with decay_steps > 0")
            warmup_steps = resolve_steps(scheduler.warmup_steps, max_steps)
            decay_steps = resolve_steps(scheduler.decay_steps, max_steps)
            if warmup_steps + decay_steps > max_steps:
                raise ValueError(
                    f"warmup_steps ({warmup_steps}) + decay_steps ({decay_steps}) "
                    f"must not exceed max_steps ({max_steps})"
                )
        elif isinstance(scheduler.warmup_steps, float) and max_steps is None:
            raise ValueError("Must specify max_steps when warmup_steps is a fraction")
    if isinstance(scheduler, CosineSchedulerConfig):
        if max_steps is None:
            raise ValueError("Must specify max_steps when using a cosine scheduler")
        if scheduler.warmup_steps >= max_steps:
            raise ValueError(f"warmup_steps ({scheduler.warmup_steps}) must be less than max_steps ({max_steps})")


class BaseOptimizerConfig(BaseConfig):
    lr: float = Field(1e-6, ge=0)
    """Peak learning rate."""

    weight_decay: Annotated[float, Field(ge=0)] | Literal["auto"] = "auto"
    """L2 weight-decay coefficient. ``"auto"`` (default) resolves to ``0.0`` for RL and ``0.01`` for SFT."""

    max_norm: float | None = Field(1.0, ge=0)
    """Maximum gradient norm to clip to. If None, gradient clipping is disabled."""


class AdamWConfig(BaseOptimizerConfig):
    type: Literal["adamw"] = "adamw"

    betas1: float = Field(0.9, ge=0)
    """Adam first-moment (β1) decay."""

    betas2: float = Field(0.999, ge=0)
    """Adam second-moment (β2) decay."""

    eps: float = Field(1e-8, gt=0)
    """Adam epsilon, added to the denominator of the update."""


class MuonConfig(BaseOptimizerConfig):
    type: Literal["muon"] = "muon"

    mu: float = Field(0.95, ge=0)
    """Momentum factor for the Muon algorithm."""

    nesterov: bool = True
    """Nesterov momentum for the Muon updates."""

    betas1: float = Field(0.9, ge=0)
    """β1 for the AdamW/Lion sub-optimizer used on non-Muon params."""

    betas2: float = Field(0.95, ge=0)
    """β2 for the AdamW/Lion sub-optimizer used on non-Muon params."""

    eps: float = Field(1e-8, gt=0)
    """Epsilon for the AdamW sub-optimizer used on non-Muon params."""


class SignSGDConfig(BaseOptimizerConfig):
    type: Literal["sign_sgd"] = "sign_sgd"


OptimizerConfig: TypeAlias = Annotated[AdamWConfig | MuonConfig | SignSGDConfig, Field(discriminator="type")]


class CheckpointConfig(BaseConfig):
    output_dir: Path | None = None
    """Override directory for checkpoints. If set, checkpoints are written here instead of under the trainer ``output_dir`` — useful for writing large checkpoints to a separate storage volume."""

    interval: int | None = Field(None, ge=1)
    """Interval at which to save the training checkpoint. If None, only checkpoints at the end of training."""

    keep_last: int | None = Field(None, ge=1)
    """Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency."""

    keep_interval: int | None = Field(None, ge=1)
    """Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping."""

    skip_scheduler: bool = False
    """Skip loading the scheduler from checkpoint."""

    skip_dataloader: bool = False
    """Skip loading the dataloader from checkpoint."""

    skip_optimizer: bool = False
    """Skip loading the optimizer state from checkpoint."""


class IPOLossConfig(BaseConfig):
    type: Literal["ipo"] = "ipo"
    eps: float = Field(0.3, ge=0)
    """Maximum absolute probability change before a token is masked."""

    max_importance_ratio: float = Field(1e4, ge=1, allow_inf_nan=False)
    """Cap the importance weight of accepted tokens while preserving its policy gradient."""

    adv_tau: float = Field(1.0, ge=0)
    """Temperature for the advantage term."""


class IcePopLossConfig(BaseConfig):
    type: Literal["icepop"] = "icepop"

    ratio_low: float = Field(0.2, gt=0)
    """Lower accepted trainer-to-inference probability ratio."""

    ratio_high: float = Field(5.0, gt=0)
    """Upper accepted trainer-to-inference probability ratio."""

    adv_tau: float = Field(1.0, ge=0)
    """Temperature for the advantage term."""

    score_centering: bool = False
    """Center the ratio-masked importance-weighted score under the sampler."""

    score_centering_topk: int = Field(128, ge=1)
    """Captured head size for an untruncated sampler with a proportional tail.
    Probabilities retain full-vocabulary normalization; sampling is not truncated."""

    @model_validator(mode="after")
    def validate_ratio_bounds(self):
        if self.ratio_low > self.ratio_high:
            raise ValueError("ratio_low must not exceed ratio_high")
        return self


class PPOLossConfig(BaseConfig):
    type: Literal["ppo"] = "ppo"

    ratio_low: float = Field(0.8, gt=0, le=1, allow_inf_nan=False)
    """Lower ratio bound for the clipped surrogate."""

    ratio_high: float = Field(1.2, ge=1, allow_inf_nan=False)
    """Upper ratio bound for the clipped surrogate."""

    max_importance_ratio: float = Field(1e4, ge=1, allow_inf_nan=False)
    """Cap the unbounded side of the surrogate while preserving its gradient."""

    adv_tau: float = Field(1.0, ge=0)
    """Temperature for the advantage term."""

    @model_validator(mode="after")
    def validate_max_importance_ratio(self):
        if self.max_importance_ratio < self.ratio_high:
            raise ValueError("max_importance_ratio must be at least ratio_high")
        return self


class CISPOLossConfig(BaseConfig):
    type: Literal["cispo"] = "cispo"

    ratio_low: float = Field(0.0, ge=0, le=1, allow_inf_nan=False)
    """Lower bound for the detached importance weight; zero disables lower clipping."""

    ratio_high: float = Field(5.0, ge=1, allow_inf_nan=False)
    """Upper bound for the detached importance weight."""

    adv_tau: float = Field(1.0, ge=0)
    """Temperature for the advantage term."""


class CustomLossConfig(BaseConfig):
    type: Literal["custom"] = "custom"

    import_path: str
    """Import path to the loss function (e.g. ``my_module.my_loss``)."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the loss function."""


LossConfig: TypeAlias = Annotated[
    IPOLossConfig | IcePopLossConfig | PPOLossConfig | CISPOLossConfig | CustomLossConfig,
    Field(discriminator="type"),
]


class FakeDataLoaderConfig(BaseConfig):
    batch_size: int = Field(2, ge=1)
    """Batch size of the fake data loader."""

    generate_samples: bool = False
    """Generate separate samples and pack them into a single micro-batch instead of using random tensors."""


class DataLoaderConfig(BaseConfig):
    fake: FakeDataLoaderConfig | None = None
    """Use a fake data loader sampling random micro-batches (for debugging)."""

    num_workers: int = Field(1, ge=0)
    """Number of background threads that prepare micro batches ahead of the training loop, analogous to torch DataLoader's ``num_workers`` but with threads instead of processes. 0 prepares them on the main thread."""


class BaseTrainerConfig(BaseConfig):
    """Fields and validators shared by the RL trainer (``TrainerConfig``) and SFT (``SFTConfig``)."""

    auto_freeze_moe_router: ClassVar[bool]
    auto_weight_decay: ClassVar[float]

    model: ModelConfig = ModelConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    optim: OptimizerConfig = AdamWConfig()

    scheduler: SchedulerConfig = ConstantSchedulerConfig()

    ckpt: CheckpointConfig | None = None

    log: TrainerLogConfig = TrainerLogConfig()

    matmul_precision: Literal["highest", "high", "medium"] = "high"
    """Precision for float32 matrix multiplications. ``highest`` is full FP32 (required on ROCm/AMD GPUs to avoid catastrophic precision loss in softmax over large vocabularies). ``high`` enables TF32 on NVIDIA GPUs for a speedup with minor precision tradeoff. See ``torch.set_float32_matmul_precision``."""

    max_steps: int | None = None
    """Maximum number of training steps. If None, runs indefinitely."""

    memory_profiler_path: Path | None = None
    """Path to write the memory profile to."""

    gc: GCConfig | None = GCConfig()
    """Garbage collection config. Disables automatic GC and runs deterministic collections every N steps to avoid stragglers. Set to null to use Python's default GC behavior."""

    trace_path: Path | None = None
    """Path to write the PyTorch profiler trace to."""

    dist_timeout_seconds: int = 900
    """Timeout in seconds for torch distributed ops: a hung collective (e.g. a dead rank) fails the run after this long."""

    heartbeat: HeartbeatConfig | None = None
    """BetterStack heartbeat configuration for monitoring training progress."""

    control: ControlConfig | None = None
    """Serve the run control API on the master rank (pause the run after the next
    step). If None, the run takes no control requests."""

    env_vars: EnvVars = {}
    """Extra environment variables for the trainer process(es). Merged on top of the launcher defaults."""

    @model_validator(mode="after")
    def resolve_auto_defaults(self):
        if self.model.freeze_moe_router == "auto":
            self.model.freeze_moe_router = self.auto_freeze_moe_router
        if self.optim.weight_decay == "auto":
            self.optim.weight_decay = self.auto_weight_decay
        return self

    @model_validator(mode="after")
    def deepep_disables_grad_clipping(self):
        if self.model.ep != 1 and self.model.moe.dispatch.type == "deepep" and self.optim.max_norm is not None:
            warnings.warn(
                "Gradient clipping is not compatible with DeepEP. "
                "Automatically setting optim.max_norm to None (disabled).",
                stacklevel=1,
            )
            self.optim.max_norm = None
        return self

    @model_validator(mode="after")
    def full_optimizer_offload_requires_supported_optimizer(self):
        if self.model.offload == "full" and self.optim.type not in ("adamw", "sign_sgd"):
            raise ValueError("Full optimizer offload only supports AdamW and SignSGD")
        return self

    @model_validator(mode="after")
    def full_optimizer_offload_disables_grad_clipping(self):
        if self.model.offload == "full" and self.optim.max_norm is not None:
            warnings.warn(
                "Gradient clipping prevents optimizer-in-backward overlap with CPU optimizer offload. "
                "Automatically setting optim.max_norm to None (disabled).",
                stacklevel=1,
            )
            self.optim.max_norm = None
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
    def validate_scheduler_steps(self):
        validate_scheduler(self.scheduler, self.max_steps)
        return self

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self


class TrainerLaunchConfig(BaseTrainerConfig):
    """Fields and validators of the trainers that launch on their own (SFT and pretraining): run identity, output
    directory, monitors, resume, deployment and SLURM."""

    resume: ResumeConfig | None = None
    """Resume the run from a checkpoint (point at it with the previous run's ``run.name``). Without ``[ckpt]`` the run loads the checkpoint but saves no new ones. If None, does not resume."""

    monitors: TrainMonitorsConfig = TrainMonitorsConfig()
    """Metric monitors (``monitors.wandb``, ``monitors.file``, ``monitors.prime``)."""

    run: RunConfig = Field(default_factory=RunConfig)
    """Run metadata. ``run.name`` names the run directory under ``output_dir``."""

    output_dir: Path = Field(default_factory=default_output_dir)
    """Directory that groups related runs. Each run writes its artifacts (checkpoints, logs, ...) to ``output_dir / run.name``. Should be a persistent directory with enough disk space. Defaults to ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""

    clean: bool = False
    """Delete the run directory (``output_dir / run.name``) before starting training. Required to overwrite a run directory that contains artifacts from a previous run when not resuming."""

    deployment: SFTDeploymentConfig = SingleNodeDeploymentConfig()

    slurm: SlurmConfig | None = None
    """SLURM configuration. When set, the run is submitted as a SLURM job instead of running locally."""

    dry_run: bool = False
    """Only validate and dump resolved configs, then exit early."""

    @property
    def run_dir(self) -> Path:
        assert self.run.dir is not None  # resolved at construction
        return self.output_dir / self.run.dir

    def default_run_name(self) -> str:
        return f"{self.model.name.split('/')[-1]}--{uuid.uuid4().hex[:8]}".lower()

    @model_validator(mode="before")
    @classmethod
    def normalize_deployment(cls, data):
        if not isinstance(data, dict):
            return data
        deployment = data.get("deployment")
        if isinstance(deployment, dict) and deployment.get("type") == "multi_node":
            for key in ("num_train_gpus", "num_infer_gpus"):
                deployment.pop(key, None)
        return data

    @model_validator(mode="after")
    def auto_setup_run_identity(self):
        """Default the run name (``default_run_name``) and the run directory, W&B run name and platform run name to it
        when not set explicitly."""
        if self.run.name is None:
            self.run.name = self.default_run_name()
        if self.run.dir is None:
            self.run.dir = self.run.name
        if self.monitors.wandb is not None and self.monitors.wandb.name is None:
            self.monitors.wandb.name = self.run.name
        if self.monitors.prime is not None and self.monitors.prime.name is None:
            self.monitors.prime.name = self.run.name
        return self

    @model_validator(mode="after")
    def validate_deployment(self):
        if self.deployment.type == "multi_node" and self.slurm is None:
            raise ValueError("Must use SLURM for multi-node deployment.")
        return self

    @model_validator(mode="after")
    def auto_setup_deployment(self):
        if self.deployment.type == "multi_node":
            if self.deployment.nodes_per_fsdp_group is not None:
                if self.deployment.num_train_nodes % self.deployment.nodes_per_fsdp_group != 0:
                    raise ValueError(
                        f"deployment.num_train_nodes ({self.deployment.num_train_nodes}) must be divisible by "
                        f"deployment.nodes_per_fsdp_group ({self.deployment.nodes_per_fsdp_group})"
                    )
                self.model.dp_replicate = self.deployment.num_train_nodes // self.deployment.nodes_per_fsdp_group
        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        if self.slurm is not None and self.slurm.job_name is None:
            self.slurm.job_name = self.run.name
        if self.slurm is not None and self.slurm.template_path is None:
            templates_dir = find_package_resource("templates")
            if templates_dir is not None:
                if self.deployment.type == "single_node":
                    self.slurm.template_path = templates_dir / "single_node_sft.sbatch.j2"
                else:
                    self.slurm.template_path = templates_dir / "multi_node_sft.sbatch.j2"
        return self


class TrainerConfig(BaseTrainerConfig):
    # RL keeps the pretrained router.
    auto_freeze_moe_router = True
    # RL optimizes the reward objective, not a fixed dataset; L2 decay toward zero fights it.
    auto_weight_decay = 0.0

    data: DataLoaderConfig = DataLoaderConfig()

    loss: LossConfig = IPOLossConfig()
    """Loss config for the rl loss component (see ``setup_rl_loss_fn``). The ce / ref_kl components are fixed and do not read this."""

    resume: ResumeConfig | None = None
    """Resume training from a checkpoint. None starts from scratch; an empty block resumes from the latest checkpoint, ``resume.step`` from that step, ``resume.dir`` from an external checkpoint step directory. Without ``ckpt`` the run loads but saves no new checkpoints."""

    weight_broadcast: WeightBroadcastConfig = FileSystemWeightBroadcastConfig()
    """Transport used to broadcast updated weights from trainer to inference."""

    rollout_transport: TransportConfig = ZMQTransportConfig()
    """Transport used to ship rollouts from orchestrator to trainer."""

    monitors: MonitorsConfig = MonitorsConfig()
    """Metric monitors (``monitors.wandb``, ``monitors.file``)."""

    output_dir: Path = Field(default_factory=default_output_dir)
    """Directory to write outputs to — checkpoints, weights, rollouts, and logs are written as subdirectories. Should be a persistent directory with enough disk space and unique per experiment running on a single node. Defaults to ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""

    enable_router_replay: bool = False
    """Return routed experts in the batch so the trainer can replay routing. Requires ``enable_return_routed_experts=true`` on the vLLM server (or ``--enable-return-routed-experts``) and is only supported for custom models."""

    @model_validator(mode="after")
    def validate_lora_broadcast(self):
        if self.model.lora is not None and self.weight_broadcast.type in ("nccl", "nixl"):
            raise ValueError(
                "LoRA requires weight_broadcast.type = 'filesystem': vLLM loads adapters only from a "
                "PEFT-shaped directory on disk - in-memory transports have no disk artifact to load from."
            )
        return self

    @model_validator(mode="after")
    def resolve_mega_moe_max_tokens_per_rank(self):
        """RL micro batches are one packed sequence of ``model.seq_len`` tokens."""
        dispatch = self.model.moe.dispatch
        if dispatch.type == "mega_moe":
            dispatch.max_tokens_per_rank = mega_moe_max_tokens_per_rank(
                dispatch.max_tokens_per_rank, self.model.seq_len
            )
        return self

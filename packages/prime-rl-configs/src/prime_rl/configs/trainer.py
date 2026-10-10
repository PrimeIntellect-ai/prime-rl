import re
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BeforeValidator, Field, field_validator, model_validator

from prime_rl.configs.monitors import MonitorsConfig
from prime_rl.configs.shared import (
    BaseModelConfig,
    EnvVars,
    FileSystemWeightBroadcastConfig,
    HeartbeatConfig,
    MetricsServerConfig,
    ResumeConfig,
    TrainerLogConfig,
    TransportConfig,
    WeightBroadcastConfig,
    ZMQTransportConfig,
)
from prime_rl.utils.config import BaseConfig, default_output_dir

# -- Shared trainer configs (used by both SFT and RL trainers) --

AttnImplementation: TypeAlias = Literal["flash_attention_2", "flash_attention_3", "flash_attention_4", "auto"]


class GCConfig(BaseConfig):
    interval: int = Field(50, ge=1)
    """Run garbage collection every N training steps. Disables Python's automatic GC so every rank collects together and one slow rank can't stall the others."""


ActivationCheckpointMode: TypeAlias = Literal["full_moe", "full", "projections", "attention", "matmul", "selective"]


class ActivationCheckpointConfig(BaseConfig):
    mode: ActivationCheckpointMode = "full"
    """What each checkpointed block keeps for backward; the rest is recomputed. Every mode checkpoints whole transformer blocks and keeps the operations that cannot be replayed (expert dispatch, top-k selections). ``full`` keeps nothing else. ``full_moe`` is ``full`` that also recomputes the FP8 Mega MoE (dispatch, expert GEMMs and combine) instead of keeping its pools; other MoE backends keep theirs. ``projections`` also keeps the projection outputs except the attention query's, so recompute runs the query projection, attention and the elementwise work. ``attention`` also keeps the attention outputs and the mHC collapses, so recompute runs the query projection and the elementwise work. ``matmul`` also keeps the query, so recompute runs no matmul. ``selective`` keeps ``targets``."""

    freq: int = Field(1, ge=1)
    """Apply activation checkpointing to every N layers."""

    targets: list[str] | None = None
    """Operator names or namespaces retained in selective mode. ``None`` uses the default targets; an explicit list replaces them."""

    layer_modes: list[ActivationCheckpointMode | Literal["none"]] | None = None
    """Mode of each decoder layer, one entry per layer of the full model (pipeline stages keep the full model's layer indices); ``none`` leaves a layer unchecked. Replaces ``mode`` and ``freq``. Early pipeline stages hold activations for more micro-batches, so they can retain less than later ones."""

    recompute_engram: bool = False
    """Engram layers (DeepSeek-V4.1) whose decoder layer runs ``full`` or ``full_moe`` also recompute their value/key projection and gate in backward, keeping only their inputs instead of the projection's output (51 KB per token)."""


class ActivationOffloadingConfig(BaseConfig):
    pin_memory: bool = True
    """Pin offloaded activations to CPU memory."""

    max_inflight_activations: int = Field(5, ge=1)
    """Max activations kept in flight while offloading. More activations smooth overlap at the cost of GPU memory."""


class PipelineActivationOffloadConfig(BaseConfig):
    """Activation offloading for the ``Async1F1B`` / ``DualPipeV`` pipeline schedules: after a micro-batch's
    forward, a stage copies the activations it keeps for the backward (its received inputs and its decoder and
    engram layers' inputs, which activation checkpointing keeps) to pinned host memory and frees them, and copies
    them back a few ops before that micro-batch's backward. Copies run on side streams beside the compute."""

    min_bytes: int = Field(64 * 2**20, ge=1)
    """Smallest tensor storage to offload, in bytes."""

    prefetch_ahead: int = Field(2, ge=1)
    """How many ops (forwards or backwards) before a micro-batch's backward its activations start coming back.
    Micro-batches whose backward follows their forward more closely stay on the GPU."""

    stages: list[int] | None = None
    """Pipeline stages that offload. ``None``: every stage."""


class OptimizerInBackwardOffloadConfig(BaseConfig):
    """Full CPU optimizer offload: FP32 masters, optimizer state (AdamW moments; SignSGD is
    stateless), and accumulated gradients live in CPU RAM, each optimizer chunk runs on CPU as
    soon as its last gradient arrives, and the refreshed BF16 weights stream back while backward
    is still executing.

    Gradient numerics: gradients are reduced across ranks in FP32 (``reduce_dtype``) but FSDP2
    materializes them in the sharded parameter's dtype, which is BF16 for the offload compute
    model — so each gradient is rounded to BF16 once before the FP32 CPU update. Masters,
    moments, accumulation, and optimizer arithmetic remain FP32. For gradient numerics
    bit-faithful to that path, disable offloading.
    """

    numa_bind: bool = True
    """Pin each rank's CPUs to its GPU's NUMA node. Disable when the launcher already manages CPU affinity or GPU sysfs topology is unavailable."""


def _normalize_optimizer_in_backward_offload(value: Any) -> Any:
    if value is True:
        return {}
    if value is False:
        return None
    return value


OptimizerInBackwardOffload = Annotated[
    OptimizerInBackwardOffloadConfig | None, BeforeValidator(_normalize_optimizer_in_backward_offload)
]


class CompileConfig(BaseConfig):
    fullgraph: bool = False
    """Compile transformer blocks with ``fullgraph=True``. Custom MoE models need ``moe_router_dtype="bfloat16"``: the fp32 router is its own FSDP unit inside the block, and dynamo cannot trace FSDP hooks."""

    mode: Literal["reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs", "lite"] | None = None
    """``torch.compile`` mode. ``reduce-overhead`` records CUDA graphs to cut kernel launch overhead; ``max-autotune`` modes trade longer compile times for tuned kernels (``max-autotune`` also records CUDA graphs, ``max-autotune-no-cudagraphs`` does not). CUDA-graphed layers re-record on new input shapes. ``None`` uses PyTorch's default mode."""


class FusionsConfig(BaseConfig):
    enabled: list[Literal["gate_up", "qkv"]] = ["gate_up", "qkv"]
    """Runtime parameter fusions. ``gate_up`` runs each MoE expert's gate and up projections as one grouped GEMM; ``qkv`` runs attention's q, k and v projections as one GEMM. Only modules that support a fusion are packed, checkpoints keep the canonical parameter names and shapes, and fusions are skipped when LoRA is enabled. Set to ``[]`` to disable."""

    shard_fused_on_dim1: bool = False
    """Experimental. Shard fused 2-D weights along dim 1 under FSDP so that weight loading and checkpointing are zero-copy: the checkpoint reads and writes the fused weights and their optimizer state in place, instead of assembling a full copy of every fused weight on each rank first. Requires the hidden size to be divisible by the FSDP shard mesh size."""


class IndexCacheConfig(BaseConfig):
    topk_freq: int = Field(1, ge=1)
    """Recompute DSA top-k indices every N layers; intervening layers reuse the cached indices. ``1`` recomputes every layer (effectively no reuse). Mirrors vLLM's ``index_topk_freq`` HF override."""

    topk_pattern: str | None = None
    """Optional per-layer schedule that overrides ``topk_freq``. ``'F'`` computes fresh indices for that layer; ``'S'`` reuses the previously cached indices. Length should match the number of decoder layers."""


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

    modules_to_save: list[str] = []
    """Module names or regex patterns to keep fully trainable (not freeze). Same matching rules as ``target_modules``."""


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


class MXFP8Config(BaseConfig):
    type: Literal["mxfp8"] = "mxfp8"
    recipe: MXFP8Recipe = "mxfp8_rceil"
    """MXFP8 recipe for dense linear modules."""

    ignore_patterns: list[str] = _DEFAULT_FP8_IGNORE_PATTERNS
    """Dense linear module names excluded from torchao MXFP8 replacement."""


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
    backend: Literal["torch", "sonicmoe", "prime_kernels"] = "torch"
    """Expert compute implementation. ``prime_kernels`` fuses the clamped SwiGLU into Hopper grouped GEMMs."""


class DeepGemmFP8MoEComputeConfig(MoEComputeConfigBase):
    """Run routed-expert grouped GEMMs with DeepGEMM FP8 kernels."""

    type: Literal["deepgemm_fp8"] = "deepgemm_fp8"
    backend: Literal["torch", "prime_kernels"] = "torch"
    """``prime_kernels`` fuses the FP8 quantization and the clamped SwiGLU into the passes around DeepGEMM's
    GEMMs (prime-kernels' ``moe_experts``, SM90)."""


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

    overlap_chunks: int = Field(1, ge=1)
    """Split each MoE layer's tokens into this many chunks and pipeline them, so one chunk's
    all-to-alls overlap another chunk's expert compute. ``1`` dispatches all tokens at once.
    Only the bf16 transport supports more than one chunk."""

    @model_validator(mode="after")
    def validate_overlap_chunks(self):
        if self.overlap_chunks > 1 and self.transport != "bf16":
            raise ValueError("overlap_chunks > 1 requires transport='bf16'")
        return self


class DeepEPMoEDispatchConfig(BaseConfig):
    """Dispatch and combine routed tokens with DeepEP."""

    type: Literal["deepep"] = "deepep"
    num_sms: int = Field(20, ge=1)
    """SMs allocated to DeepEP communication kernels."""

    token_chunk_size: int | None = Field(None, ge=1)
    """Optional chunk size used to pipeline dispatch with local expert compute."""

    fp8: bool = False
    """Send tokens to the experts as FP8 (1 x 128 blocks, power-of-two scales), halving the forward
    dispatch traffic and the received tokens kept for backward. Requires FP8 expert compute, which
    quantizes its input the same way; gradients travel in bf16 unless ``fp8_grad`` is set."""

    fp8_grad: bool = False
    """Also send the output gradient to the experts as FP8 in backward (the DeepSeek-V3 recipe), halving
    that traffic. The experts' backward quantizes it the same way for its data-gradient GEMM, but the
    router's gradient and the weight-gradient quantization then read the FP8 values. Requires ``fp8``."""

    keep_expert_activations: bool = False
    """Keep the routed experts' forward activations (their output and what the FP8 expert backward
    reads) instead of recomputing the expert forward in backward. Costs ~2.5 GB per layer at 16k
    tokens per GPU; requires prime-kernels' FP8 expert compute."""

    weight_grads_after_combine: bool = False
    """In backward, send each token chunk's input gradient back (combine) before running its experts'
    weight-gradient GEMMs, so those GEMMs hide the combine instead of delaying it. Same numerics.
    Takes effect with prime-kernels' FP8 expert compute when the expert gradients accumulate into
    FSDP's fp32 buffers."""


class MegaMoEDispatchConfig(BaseConfig):
    """Run each MoE layer as prime-mega-moe's fused BF16 Mega MoE kernel: dispatch, the routed experts'
    clamped SwiGLU, the shared expert and the combine in one kernel over NVLink symmetric memory.
    Replaces the expert compute backend for those layers; expert parallelism must stay within a node."""

    type: Literal["mega_moe"] = "mega_moe"

    max_tokens_per_rank: int | None = Field(None, ge=1)
    """Tokens per rank the symmetric buffer is sized for. Defaults to ``model.seq_len``."""

    num_sms: int | None = Field(None, ge=1)
    """SMs the Mega MoE kernels run on. Defaults to every SM; fewer leaves room for collectives that
    overlap them (the kernel's blocks wait on other ranks, so it needs all of its blocks resident)."""

    fp8: bool = False
    """Use the SM90 FP8 Mega MoE instead: the routed experts in blockwise FP8 (prime-kernels' recipe), with
    the forward keeping each routed row's FP8 input and bf16 gate/up output so the backward does not
    recompute it. The shared expert stays with the layer."""

    capacity_factor: float = Field(1.25, gt=0)
    """FP8 only: routed rows per rank the kept pools hold, as a multiple of ``max_tokens_per_rank * top_k``
    (plus one partial block per local expert). A rank receiving more rows stops with a device-side error."""

    wgrad_tile_scales: bool = False
    """FP8 only: quantize the weight gradients' ``x`` and ``dy`` operands per 128 x 128 tile instead of per column
    over each 128 rows, so the K-grouped GEMMs promote with one FFMA per element (the other operand keeps per-column
    scales). Changes numerics; ~0.5 ms less per layer in the weight-gradient GEMMs on H200."""

    free_bf16_expert_weights: bool = False
    """FP8 only: free the local experts' unsharded bf16 weights once they are quantized for the step, so only the
    FP8 copies the kernels read stay resident (2 bytes less per local expert parameter: 3.4 GB per V4.1 layer at EP8).
    FSDP all-gathers them again at the next step's first forward. Same numerics."""


MoEDispatchConfig: TypeAlias = Annotated[
    TorchMoEDispatchConfig | DeepEPMoEDispatchConfig | MegaMoEDispatchConfig,
    Field(discriminator="type"),
]


class MoERuntimeConfig(BaseConfig):
    """Independent routed-expert compute and token-dispatch choices."""

    compute: MoEComputeConfig = BF16MoEComputeConfig()
    dispatch: MoEDispatchConfig = TorchMoEDispatchConfig()

    reduce_local_expert_grads_once: bool = False
    """When expert parallelism spans every data-parallel rank, so each expert's FSDP group holds one
    rank, reduce and reshard the expert parameters only after the last micro-batch's backward instead
    of after every one: their gradients keep accumulating in FSDP's fp32 buffer, and later micro-batches
    reuse the bf16 copy the first one cast. Same values up to fp32 summation order."""

    @model_validator(mode="after")
    def fp8_dispatch_requires_fp8_compute(self):
        if isinstance(self.dispatch, DeepEPMoEDispatchConfig) and self.dispatch.fp8:
            if not isinstance(self.compute, DeepGemmFP8MoEComputeConfig):
                raise ValueError("dispatch.fp8 requires compute.type = 'deepgemm_fp8'")
        if isinstance(self.dispatch, DeepEPMoEDispatchConfig) and self.dispatch.fp8_grad and not self.dispatch.fp8:
            raise ValueError("dispatch.fp8_grad requires dispatch.fp8")
        if isinstance(self.dispatch, MegaMoEDispatchConfig) and self.dispatch.free_bf16_expert_weights:
            if not self.dispatch.fp8:
                raise ValueError("dispatch.free_bf16_expert_weights requires dispatch.fp8")
        return self


class ModelConfig(BaseModelConfig):
    conversion_dir: Path | None = None
    """Directory for the auto-converted weights (written to a `prime`/`hf` subdirectory). If not set, we write into the model snapshot directory."""

    seq_len: int = 2048
    """Sequence length the model is trained on."""

    attn: AttnImplementation = "auto"
    """Attention implementation. ``auto`` selects FA3 on Hopper (SM90) and FA4 on Blackwell (SM100+). With CP enabled, ring attention uses the matching kernel family (FA2/FA3/FA4)."""

    compile: CompileConfig | None = CompileConfig()
    """Compile the model with ``torch.compile``."""

    fusions: FusionsConfig = FusionsConfig()
    """Runtime parameter fusions, on by default."""

    ac: ActivationCheckpointConfig | None = ActivationCheckpointConfig()
    """Activation checkpointing configuration. If None, activation checkpointing is disabled."""

    ac_offloading: ActivationOffloadingConfig | None = ActivationOffloadingConfig()
    """Activation offloading configuration. If None, activation offloading is disabled."""

    fsdp_cpu_offload: bool = False
    """Enable FSDP CPU offloading for parameters, gradients, and optimizer states. Uses pinned memory for efficient CPU↔GPU transfers."""

    optim_cpu_offload: bool = True
    """Offload only optimizer states (momentum, variance) to CPU, keeping weights on GPU. Avoids the H2D all-gather overhead of FSDP CPU offload while still saving GPU memory."""

    full_offload: OptimizerInBackwardOffload = None
    """Full CPU optimizer offload: FP32 masters, moments, and gradients live in CPU RAM and the optimizer runs on CPU, overlapped with backward. Enable with ``true`` or a ``[model.full_offload]`` section; disabled by default."""

    reshard_after_forward: bool = True
    """Reshard the model after each forward pass."""

    dp_replicate: int = 1
    """Data parallel dim where model weights are replicated."""

    ep: int | Literal["auto"] = "auto"
    """Expert parallelism degree for MoE layers. 1 disables EP. ``auto`` resolves to ``min(fsdp_island_size, 8)`` for MoE models (where ``fsdp_island_size = world_size // dp_replicate``), and to 1 for non-MoE models. Set an explicit integer to override."""

    moe: MoERuntimeConfig = MoERuntimeConfig()
    """Routed-expert compute and token-dispatch runtime."""

    cp: int = 1
    """Context parallelism degree. 1 disables CP."""

    pp: int = 1
    """Pipeline parallelism degree. 1 disables PP. The decoder layers are split into ``pp * pp_stages_per_rank`` stages of consecutive layers; each stage is FSDP-sharded (and expert-parallel) over its pipeline rank's devices."""

    pp_schedule: Literal[
        "1F1B", "Async1F1B", "DualPipeV", "GPipe", "Interleaved1F1B", "InterleavedZeroBubble", "ZBVZeroBubble"
    ] = "1F1B"
    """Pipeline schedule. The step's micro-batches are the pipeline's micro-batches. ``1F1B`` and ``GPipe`` run one stage per rank; the interleaved schedules loop ``pp_stages_per_rank`` stages over the ranks; ``ZBVZeroBubble`` places two stages per rank in a V. ``Async1F1B`` (one stage per rank) and ``DualPipeV`` (two stages per rank in a V, DeepSeek's order with full backwards, at least ``2 * pp`` micro-batches) run on an executor that never waits on a stage transfer before it is needed; DualPipeV also hides the transfer time that 1F1B adds to every cycle, at the same worst-rank activation memory."""

    pp_warmup_step: list[Annotated[int, Field(ge=1)]] = [1]
    """``Async1F1B`` only: how many more warmup forwards each stage runs than the next one, one entry per neighbour pair (``pp - 1``, first pair first) or a single entry for all. 1 is classic 1F1B, which waits for one activation and one gradient transfer every cycle; 2 hides both transfers at one more micro-batch in flight on every earlier stage (all 2: stage s keeps ``2 * (pp - s) - 1``). Memory-bound early stages can keep 1 if they carry about two transfers less work per micro-batch than the bottleneck stage. Larger steps only add in-flight micro-batches (useful to emulate a deeper pipeline's memory)."""

    pp_transport: Literal["nccl", "copy_engine"] = "nccl"
    """``Async1F1B`` / ``DualPipeV`` stage transfers. ``nccl`` uses send/recv, whose kernels hold SMs while a transfer is in flight and slow persistent kernels (fused MoE, DeepGEMM) that overlap it. ``copy_engine`` sends each micro-batch as an in-place two-rank all-gather of an NCCL symmetric-window buffer with the zero-CTA policy, which NCCL >= 2.32 runs on copy engines only; it costs one buffer of a micro-batch's activations per transfer edge, sends as many bytes back as forward, and needs NCCL >= 2.32 at runtime (older NCCL falls back to SM kernels)."""

    pp_transport_ctas: int | None = Field(None, ge=1)
    """``nccl`` transport only: CTAs (SMs) each stage send/recv kernel takes. Pin it and shrink the persistent kernels by as many SMs (``moe.dispatch.num_sms``; DeepGEMM via ``pp_gemm_sms``) so a transfer in flight does not slow them. ``None`` leaves NCCL's default."""

    pp_gemm_sms: int | None = Field(None, ge=2, multiple_of=2)
    """SMs DeepGEMM's dense FP8 GEMMs run on under pipeline parallelism (``deep_gemm.set_num_sms``); with ``pp_transport_ctas`` this keeps their waves off the SMs a transfer holds. ``None`` uses every SM."""

    pp_stages_per_rank: int = Field(1, ge=1)
    """Pipeline stages each pipeline rank holds."""

    pp_layers_per_stage: list[float] | None = None
    """Decoder layers of each stage, in stage order (``pp * pp_stages_per_rank`` entries; DualPipeV numbers its stages along the V, so rank 0 holds the first and the last). Multiples of 0.5 cut a layer between its attention and its MoE block (DeepSeek-V4.1); 0 gives a stage with only the embedding or the head. Defaults to an even split of whole layers, earlier stages taking the remainder."""

    pp_activation_offload: PipelineActivationOffloadConfig | None = None
    """``Async1F1B`` / ``DualPipeV`` only: offload the activations a stage keeps for its in-flight micro-batches to host memory between their forward and backward. ``None`` keeps them on the GPU. (``ac_offloading`` applies only without pipeline parallelism.)"""

    cp_style: Literal["ring", "ulysses"] = "ring"
    """CP communication style. ``ring`` uses ring-attention all-gather/reduce-scatter (requires custom kernels per attention type). ``ulysses`` uses all-to-all to redistribute Q/K/V from sequence-sharded to head-sharded, runs vanilla attention locally on the full sequence, then all-to-all back — works out-of-the-box with any attention kernel (softmax FA, linear attention, mamba, etc.)."""

    optimization_dtype: Literal["bfloat16", "float32"] = "float32"
    """dtype for model optimization."""

    reduce_dtype: Literal["bfloat16", "float32"] = "float32"
    """dtype for gradient/parameter reductions."""

    moe_router_dtype: Literal["bfloat16", "float32", "auto"] = "auto"
    """Compute dtype for MoE router gates. ``float32`` keeps router gate weights in fp32 through forward and backward (exempt from FSDP bf16 parameter casting) and computes the gate GEMM and routing logits in fp32, matching models trained with fp32 routing (e.g. GLM-5.x via Megatron's ``--moe-router-dtype fp32``). ``bfloat16`` computes the gate GEMM in the model compute dtype. ``auto`` (default) resolves to ``float32`` for RL and ``bfloat16`` for SFT. Router score functions (sigmoid/softmax) run in fp32 regardless. A no-op for non-MoE models."""

    quantization: QuantizationConfig | None = None

    index_cache: IndexCacheConfig | None = None
    """DSA IndexCache sub-configuration. If set, sparse-attention top-k indices are reused across decoder layers per the configured schedule (mirrors vLLM's IndexCache HF overrides). If None, every layer recomputes its own indices."""

    freeze_moe_router: bool = False
    """Freeze MoE router parameters during training."""

    freeze_engram_tables: bool = False
    """Freeze DeepSeek-V4.1's engram n-gram tables (~98B parameters each). Required for RL: the tables are
    too large to broadcast, so the inference engine keeps serving its own copy."""

    lora: LoRAConfig | None = None
    """LoRA configuration. If None, LoRA is disabled."""

    debug: DebugModelConfig = DebugModelConfig()
    """Debugging knobs for the model and distributed training."""

    fused_lm_head_token_chunk_size: int | Literal["disabled"] = 8192
    """Flattened token chunk size for the fused LM head. ``int >= 1`` sets the tokens per LM-head chunk explicitly; ``disabled`` uses the vanilla LM head. In SFT the fused head computes the summed cross-entropy and its gradients chunk by chunk, holding one chunk's full-vocab logits at a time."""

    @model_validator(mode="after")
    def vlm_cp_requires_ulysses(self):
        if self.vlm is not None and self.cp > 1 and self.cp_style != "ulysses":
            raise ValueError("VLM models require cp_style='ulysses' for context parallelism")
        return self

    @model_validator(mode="after")
    def validate_cp(self):
        if self.cp > 1 and self.attn not in ["flash_attention_2", "flash_attention_3", "flash_attention_4", "auto"]:
            raise ValueError("CP is only supported with flash attention 2, 3, or 4")
        return self

    @model_validator(mode="after")
    def pp_activation_offload_requires_async_schedule(self):
        if self.pp_activation_offload is not None and self.pp_schedule not in ("Async1F1B", "DualPipeV"):
            raise ValueError("model.pp_activation_offload requires pp_schedule 'Async1F1B' or 'DualPipeV'")
        return self

    @model_validator(mode="after")
    def ac_offloading_requires_ac(self):
        """Automatically enable activation checkpointing when activation offloading is enabled."""
        if self.ac_offloading is not None and self.ac is None:
            self.ac = ActivationCheckpointConfig()
        return self

    @model_validator(mode="after")
    def cpu_offload_mutual_exclusion(self):
        if self.fsdp_cpu_offload and (self.optim_cpu_offload or self.full_offload):
            raise ValueError("Cannot combine fsdp_cpu_offload with optimizer CPU offloading.")
        if self.optim_cpu_offload and self.full_offload:
            raise ValueError(
                "Cannot enable both optim_cpu_offload and full_offload. "
                "Set optim_cpu_offload=false when enabling full optimizer offload."
            )
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
        elif isinstance(dispatch, TorchMoEDispatchConfig) and dispatch.transport == "mxfp8":
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
            if scheduler.warmup_steps + scheduler.decay_steps > max_steps:
                raise ValueError(
                    f"warmup_steps ({scheduler.warmup_steps}) + decay_steps ({scheduler.decay_steps}) "
                    f"must not exceed max_steps ({max_steps})"
                )
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

    embedding_update: Literal["sinkhorn", "adamw"] | None = None
    """How the token embedding, the LM head and the Engram hash tables are updated, with DeepSeek-V4.1's
    split of the other parameters (tech report §2.5): Engram projections go to Muon with the other
    matrices, normalization weights to AdamW with weight decay, biases and scaling factors (Engram gate
    weights, mHC bias and scale, attention sinks) to AdamW without. ``"sinkhorn"`` is the report's
    momentum update with Sinkhorn balancing (one fp32 state per parameter, no weight decay);
    ``"adamw"`` is AdamW without weight decay. ``None`` sends the embedding and LM head to AdamW and every
    other 2-D parameter, Engram tables included, to Muon."""

    sinkhorn_iters: int = Field(11, ge=1)
    """Alternating row / column normalizations of the Sinkhorn update (odd: it starts and ends with rows)."""

    sinkhorn_tau: float = Field(1e-3, ge=0)
    """Rows of the momentum update whose norm is at most this fraction of the mean row norm are not updated."""

    sinkhorn_eps: float = Field(1e-20, gt=0)
    """Added to every row and column norm of the Sinkhorn update."""

    sinkhorn_lr_scale: float = Field(0.18, gt=0)
    """Learning-rate correction of the Sinkhorn update (gamma), which has unit row-wise RMS."""

    engram_lr_scale: float = Field(5.0, gt=0)
    """Learning-rate multiplier of the Engram hash tables. Only used with ``embedding_update`` set."""

    @model_validator(mode="after")
    def validate_sinkhorn_iters(self):
        if self.sinkhorn_iters % 2 != 1:
            raise ValueError(f"optim.sinkhorn_iters must be odd, got {self.sinkhorn_iters}")
        return self


class SignSGDConfig(BaseOptimizerConfig):
    type: Literal["sign_sgd"] = "sign_sgd"

    apply_in_backward: bool = False
    """Update each parameter as soon as its gradient is final and free the gradient, so no step holds
    every gradient at once. SFT only, with one micro-batch per step and no optimizer offload. Exact:
    sign updates ignore the positive loss scaling and clipping applied after backward."""


OptimizerConfig: TypeAlias = Annotated[
    SGDConfig | AdamWConfig | MuonConfig | SignSGDConfig, Field(discriminator="type")
]


class CheckpointConfig(BaseConfig):
    output_dir: Path | None = None
    """Override directory for checkpoints. If set, checkpoints are written here instead of under the trainer ``output_dir`` — useful for writing large checkpoints to a separate storage volume."""

    interval: int | None = Field(None, ge=1)
    """Interval at which to save the training checkpoint. If None, only checkpoints at the end of training."""

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
    IPOLossConfig | IcePopLossConfig | PPOLossConfig | CISPOLossConfig | CustomLossConfig, Field(discriminator="type")
]


class FakeDataLoaderConfig(BaseConfig):
    batch_size: int = Field(2, ge=1)
    """Batch size of the fake data loader."""

    generate_samples: bool = False
    """Generate separate samples and pack them into a single micro-batch instead of using random tensors."""


class DataLoaderConfig(BaseConfig):
    fake: FakeDataLoaderConfig | None = None
    """Use a fake data loader sampling random micro-batches (for debugging)."""


class TrainerConfig(BaseConfig):
    model: ModelConfig = ModelConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    data: DataLoaderConfig = DataLoaderConfig()

    loss: LossConfig = IPOLossConfig()
    """Loss config for the rl loss component (see ``setup_rl_loss_fn``). The ce / ref_kl components are fixed and do not read this."""

    optim: OptimizerConfig = AdamWConfig()

    scheduler: SchedulerConfig = ConstantSchedulerConfig()

    ckpt: CheckpointConfig | None = None

    resume: ResumeConfig | None = None
    """Resume training from a checkpoint. None starts from scratch; an empty block resumes from the latest checkpoint, ``resume.step`` from that step, ``resume.dir`` from an external checkpoint step directory. Without ``ckpt`` the run loads but saves no new checkpoints."""
    """Full training-state checkpoint configuration (model + optimizer + scheduler). If None, no resume-capable checkpoints are written."""

    weight_broadcast: WeightBroadcastConfig = FileSystemWeightBroadcastConfig()
    """Transport used to broadcast updated weights from trainer to inference."""

    rollout_transport: TransportConfig = ZMQTransportConfig()
    """Transport used to ship rollouts from orchestrator to trainer."""

    log: TrainerLogConfig = TrainerLogConfig()

    monitors: MonitorsConfig = MonitorsConfig()
    """Metric monitors (``monitors.wandb``, ``monitors.file``)."""

    output_dir: Path = Field(default_factory=default_output_dir)
    """Directory to write outputs to — checkpoints, weights, rollouts, and logs are written as subdirectories. Should be a persistent directory with enough disk space and unique per experiment running on a single node. Defaults to ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""

    matmul_precision: Literal["highest", "high", "medium"] = "high"
    """Precision for float32 matrix multiplications. ``highest`` is full FP32 (required on ROCm/AMD GPUs to avoid catastrophic precision loss in softmax over large vocabularies). ``high`` enables TF32 on NVIDIA GPUs for a speedup with minor precision tradeoff. See ``torch.set_float32_matmul_precision``."""

    max_steps: int | None = None
    """Maximum number of training steps. If None, runs indefinitely."""

    enable_router_replay: bool = False
    """Return routed experts in the batch so the trainer can replay routing. Requires ``enable_return_routed_experts=true`` on the vLLM server (or ``--enable-return-routed-experts``) and is only supported for custom models."""

    memory_profiler_path: Path | None = None
    """Path to write the memory profile to."""

    gc: GCConfig | None = GCConfig()
    """Garbage collection config. Disables automatic GC and runs deterministic collections every N steps to avoid stragglers. Set to null to use Python's default GC behavior."""

    trace_path: Path | None = None
    """Path to write the PyTorch profiler trace to."""

    dist_timeout_seconds: int = 3600
    """Timeout in seconds for torch distributed ops."""

    heartbeat: HeartbeatConfig | None = None
    """BetterStack heartbeat configuration for monitoring training progress."""

    metrics_server: MetricsServerConfig | None = None
    """Prometheus metrics server configuration. If set, exposes a ``/metrics`` endpoint for scraping."""

    env_vars: EnvVars = {}
    """Extra environment variables for the trainer process(es). Merged on top of the launcher defaults."""

    @model_validator(mode="after")
    def resolve_moe_router_dtype_auto(self):
        """Resolve ``model.moe_router_dtype='auto'``: RL routes in fp32, matching the fp32-routed checkpoints it trains from (e.g. GLM-5.x)."""
        if self.model.moe_router_dtype == "auto":
            self.model.moe_router_dtype = "float32"
        return self

    @model_validator(mode="after")
    def resolve_weight_decay_auto(self):
        """Resolve ``optim.weight_decay='auto'``: RL optimizes the reward objective, not a fixed dataset — L2 decay toward zero fights it, so default to no weight decay."""
        if self.optim.weight_decay == "auto":
            self.optim.weight_decay = 0.0
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
        if self.model.full_offload and self.optim.type not in ("adamw", "sign_sgd"):
            raise ValueError("Full optimizer offload only supports AdamW and SignSGD")
        return self

    @model_validator(mode="after")
    def full_optimizer_offload_disables_grad_clipping(self):
        if self.model.full_offload and self.optim.max_norm is not None:
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
    def validate_opt_and_fsdp_offload(self):
        if self.optim.type == "muon" and self.model.fsdp_cpu_offload:
            raise ValueError("Muon optimizer does not support FSDP CPU offload")
        return self

    @model_validator(mode="after")
    def validate_lora_broadcast(self):
        if self.model.lora is not None and self.weight_broadcast.type in ("nccl", "nixl"):
            raise ValueError(
                "LoRA requires weight_broadcast.type = 'filesystem': vLLM loads adapters only from a "
                "PEFT-shaped directory on disk - in-memory transports have no disk artifact to load from."
            )
        if self.model.lora is not None and self.model.lora.modules_to_save and self.data.fake is None:
            raise ValueError(
                "model.lora.modules_to_save cannot be served: the weight broadcast ships only the "
                "adapter tensors, so fully-trained modules would silently diverge from inference."
            )
        return self

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

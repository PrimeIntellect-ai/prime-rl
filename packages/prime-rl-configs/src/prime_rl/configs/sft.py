import warnings
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.configs.shared import (
    HeartbeatConfig,
    RendererConfig,
    SlurmConfig,
    TrainerLogConfig,
    WandbConfig,
)
from prime_rl.configs.trainer import (
    AdamWConfig,
    BenchConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    GCConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
)
from prime_rl.utils.config import BaseConfig, find_package_resource


class BaseDataConfig(BaseConfig):
    batch_size: int = Field(128, ge=1)
    """Global batch size."""

    seq_len: int = Field(128, ge=1)
    """Sequence length."""

    pack_function: Literal["cat", "stack"] = "cat"
    """Sample packing strategy. ``cat`` concatenates; ``stack`` requires ``seq_len`` divisible by 256."""

    micro_batch_size: int = Field(1, ge=1)
    """Per-step micro batch size. ``batch_size`` must be divisible by this."""

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")
        return self


class FakeDataConfig(BaseDataConfig):
    type: Literal["fake"] = "fake"

    length: Literal["fixed", "variable"] = "fixed"
    """Use fixed-length samples or variable-length samples."""

    input_ids: Literal["increasing", "random"] = "increasing"
    """Token id generator: ``increasing`` for deterministic sequences, ``random`` for random ids."""


class CaterpillarFakeDataConfig(BaseDataConfig):
    """Configures synthetic tree data for Tree Training v1 validation."""

    type: Literal["caterpillar_fake"] = "caterpillar_fake"

    vocab_size: Annotated[int | None, Field(ge=1)] = None
    num_turns: Annotated[int, Field(ge=1)] = 3
    turns: Annotated[int | None, Field(ge=1)] = None
    user_len: tuple[int, int] = (4, 4)
    think_len: tuple[int, int] = (6, 6)
    response_len: tuple[int, int] = (6, 6)
    seed: int = 0
    train_response: bool = True
    train_think: bool = True

    @model_validator(mode="after")
    def validate_caterpillar_fake(self):
        if self.turns is not None:
            self.num_turns = self.turns
        for name in ("user_len", "think_len", "response_len"):
            low, high = getattr(self, name)
            if low < 1 or high < 1:
                raise ValueError(f"{name} bounds must be at least 1")
            if low > high:
                raise ValueError(f"{name} lower bound must be <= upper bound")
        if not (self.train_response or self.train_think):
            raise ValueError("At least one of train_response or train_think must be true")
        return self


class CaterpillarPerBranchDataConfig(BaseDataConfig):
    """Per-branch baseline counterpart of CaterpillarFakeDataConfig.

    Builds the same caterpillar trees (same seed, same turns), then yields each
    leaf's root-to-leaf token path as an independent flat SFT sample. Pair with
    CaterpillarFakeDataConfig (same fields) to form a tree-vs-baseline ablation.
    """

    type: Literal["caterpillar_per_branch"] = "caterpillar_per_branch"

    vocab_size: Annotated[int | None, Field(ge=1)] = None
    num_turns: Annotated[int, Field(ge=1)] = 3
    turns: Annotated[int | None, Field(ge=1)] = None
    user_len: tuple[int, int] = (4, 4)
    think_len: tuple[int, int] = (6, 6)
    response_len: tuple[int, int] = (6, 6)
    seed: int = 0
    train_response: bool = True
    train_think: bool = True

    @model_validator(mode="after")
    def validate_caterpillar_per_branch(self):
        if self.turns is not None:
            self.num_turns = self.turns
        for name in ("user_len", "think_len", "response_len"):
            low, high = getattr(self, name)
            if low < 1 or high < 1:
                raise ValueError(f"{name} bounds must be at least 1")
            if low > high:
                raise ValueError(f"{name} lower bound must be <= upper bound")
        if not (self.train_response or self.train_think):
            raise ValueError("At least one of train_response or train_think must be true")
        return self


class _SFTCaterpillarBaseConfig(BaseDataConfig):
    """Shared fields for the realistic SFT caterpillar datasets.

    Wraps an HF SFT dataset whose assistant messages contain ``<think>...</think>``
    segments. Each example becomes a 1-turn caterpillar: user prompt as the trunk's
    user node, ``<think>...</think>`` as a leaf, the post-``</think>`` content as a
    sibling leaf. Useful for tree-vs-baseline ablation on real reasoning data.
    """

    name: Annotated[str, Field(description="Name or path of the HF dataset to use.")] = (
        "PrimeIntellect/INTELLECT-3-SFT-10K"
    )
    subset: Annotated[str | None, Field(description="HF subset (config) name.")] = "default"
    split: Annotated[str, Field(description="HF split name.")] = "math"
    shuffle: bool = True
    seed: int = 0
    train_response: bool = True
    train_think: bool = True


class SFTCaterpillarDataConfig(_SFTCaterpillarBaseConfig):
    """Real-data tree-mode counterpart of CaterpillarFakeDataConfig."""

    type: Literal["sft_caterpillar"] = "sft_caterpillar"


class SFTCaterpillarPerBranchDataConfig(_SFTCaterpillarBaseConfig):
    """Real-data per-branch baseline. Yields each leaf's root-to-leaf path as a flat sample."""

    type: Literal["sft_caterpillar_per_branch"] = "sft_caterpillar_per_branch"


class _SFTRawToolCaterpillarBaseConfig(BaseDataConfig):
    """Shared fields for raw tool-trajectory caterpillar datasets.

    Wraps raw multi-turn SFT trajectories whose assistant messages carry
    ``reasoning_content`` separately from visible content/tool calls. Each assistant
    turn adds a visible trunk continuation and, when non-empty, a reasoning side leaf.
    """

    name: Annotated[str, Field(description="Name or path of the HF dataset to use.")] = (
        "PrimeIntellect/INTELLECT-5-SFT-Raw"
    )
    subset: Annotated[str | None, Field(description="HF subset (config) name.")] = "rlm_science"
    split: Annotated[str, Field(description="HF split name.")] = "train"
    sort_by_num_turns: bool = True
    selection_metric: Literal["num_turns", "branching_score"] = "num_turns"
    selection_num_proc: Annotated[int | None, Field(ge=1)] = None
    max_examples: Annotated[int | None, Field(ge=1)] = None
    max_packed_tokens: Annotated[int | None, Field(ge=1)] = None
    filter_by_final_token_estimate: bool = True
    seed: int = 0
    train_response: bool = True
    train_reasoning: bool = True
    include_attn_mask: bool = True

    @model_validator(mode="after")
    def validate_raw_tool_tree_limits(self):
        if self.max_packed_tokens is not None and self.max_packed_tokens < self.seq_len:
            raise ValueError("max_packed_tokens must be greater than or equal to seq_len")
        return self


class SFTRawToolCaterpillarDataConfig(_SFTRawToolCaterpillarBaseConfig):
    """Raw tool-trajectory tree-mode counterpart for reasoning_content ablations."""

    type: Literal["sft_raw_tool_caterpillar"] = "sft_raw_tool_caterpillar"


class SFTRawToolCaterpillarPerBranchDataConfig(_SFTRawToolCaterpillarBaseConfig):
    """Raw tool-trajectory per-branch baseline."""

    type: Literal["sft_raw_tool_caterpillar_per_branch"] = "sft_raw_tool_caterpillar_per_branch"


class SFTRawToolCaterpillarGroupedBranchesDataConfig(_SFTRawToolCaterpillarBaseConfig):
    """Raw tool-trajectory per-tree grouped branch equivalence baseline."""

    type: Literal["sft_raw_tool_caterpillar_grouped_branches"] = "sft_raw_tool_caterpillar_grouped_branches"


class SFTRawToolCurrentRLBaselineDataConfig(_SFTRawToolCaterpillarBaseConfig):
    """Raw tool-trajectory baseline matching the current RL interleaved sample path."""

    type: Literal["sft_raw_tool_current_rl_baseline"] = "sft_raw_tool_current_rl_baseline"


class LossMaskConfig(BaseConfig):
    system: bool = False
    """System messages contribute to the loss."""

    user: bool = False
    """User messages contribute to the loss."""

    assistant: bool = True
    """Assistant messages contribute to the loss."""

    tool: bool = False
    """Tool messages contribute to the loss."""


class SFTDataConfig(BaseDataConfig):
    type: Literal["sft"] = "sft"

    name: str = "PrimeIntellect/Reverse-Text-SFT"
    """HF dataset name or path."""

    subsets: list[str] | None = None
    """Subsets to load from the HF dataset."""

    splits: list[str] | None = None
    """Splits to load from the HF dataset."""

    probabilities: list[float] | None = None
    """Sampling probabilities for each subset/split."""

    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "all_exhausted"
    """Stopping strategy when interleaving multiple subsets/splits."""

    shuffle: bool = True
    """Shuffle the dataset at the start of each epoch."""

    seed: int = 0
    """Random seed for shuffling. Re-shuffled per epoch by adding the epoch count to the seed."""

    # Configuring
    loss_mask: LossMaskConfig = LossMaskConfig()
    """Which message types contribute to the loss."""

    @model_validator(mode="after")
    def validate_subsets_and_splits(self):
        if self.subsets is not None or self.splits is not None:
            if self.subsets is not None and self.splits is not None:
                if len(self.subsets) != len(self.splits):
                    raise ValueError(
                        "Number of subsets must be equal to number of splits. Please specify which split to load for each subset."
                    )
            if self.subsets is not None and self.probabilities is not None:
                if len(self.probabilities) != len(self.subsets):
                    raise ValueError(
                        "Number of probabilities must be equal to number of subsets. Please specify a probability for each subset."
                    )
            if self.splits is not None and self.probabilities is not None:
                if len(self.probabilities) != len(self.splits):
                    raise ValueError(
                        "Number of probabilities must be equal to number of splits. Please specify a probability for each split."
                    )
        return self


class SFTValConfig(BaseConfig):
    interval: int = Field(50, ge=1)
    """Run validation every N training steps."""

    eval_on_start: bool = False
    """Run validation before the first training step."""

    data: SFTDataConfig


DataConfig: TypeAlias = Annotated[
    FakeDataConfig
    | CaterpillarFakeDataConfig
    | CaterpillarPerBranchDataConfig
    | SFTCaterpillarDataConfig
    | SFTCaterpillarPerBranchDataConfig
    | SFTRawToolCaterpillarDataConfig
    | SFTRawToolCaterpillarPerBranchDataConfig
    | SFTRawToolCaterpillarGroupedBranchesDataConfig
    | SFTRawToolCurrentRLBaselineDataConfig
    | SFTDataConfig,
    Field(discriminator="type"),
]


class BaseDeploymentConfig(BaseConfig):
    gpus_per_node: int = 8
    """GPUs per node."""


class SingleNodeDeploymentConfig(BaseDeploymentConfig):
    type: Literal["single_node"] = "single_node"

    num_gpus: int = 1
    """GPUs to use."""

    @model_validator(mode="after")
    def validate_gpu_count(self):
        if self.num_gpus > self.gpus_per_node:
            raise ValueError(f"num_gpus ({self.num_gpus}) exceeds gpus_per_node ({self.gpus_per_node}).")
        return self


class MultiNodeDeploymentConfig(BaseDeploymentConfig):
    type: Literal["multi_node"] = "multi_node"

    num_nodes: int = 2
    """Training nodes."""

    nodes_per_fsdp_group: int | None = None
    """Nodes per FSDP island. Auto-sets ``model.dp_replicate = num_nodes / nodes_per_fsdp_group``."""


SFTDeploymentConfig: TypeAlias = Annotated[
    SingleNodeDeploymentConfig | MultiNodeDeploymentConfig, Field(discriminator="type")
]


class SFTExperimentalConfig(BaseConfig):
    pass


class SFTConfig(BaseConfig):
    model: ModelConfig = ModelConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    renderer: RendererConfig = RendererConfig()
    """Client-side renderer configuration. Only consumed when ``use_renderer=true``."""

    use_renderer: bool = False
    """Tokenize SFT samples through the ``renderers`` library (single ``render()`` + ``message_indices`` mask) instead of the default ``build_incremental_token_mask`` path. Required for chat templates that render position-dependently (e.g. Qwen3, Qwen3.5)."""

    data: DataConfig = SFTDataConfig()

    val: SFTValConfig | None = None
    """Validation configuration. If None, no validation runs."""

    optim: OptimizerConfig = AdamWConfig()

    scheduler: SchedulerConfig = ConstantSchedulerConfig()

    ckpt: CheckpointConfig | None = None

    log: TrainerLogConfig = TrainerLogConfig()

    wandb: WandbConfig | None = None

    output_dir: Path = Path("outputs")
    """Directory to write outputs to — checkpoints and logs are written as subdirectories. Should be a persistent directory with enough disk space and unique per experiment running on a single node."""

    clean_output_dir: bool = False
    """Delete the output directory before starting training. Required to overwrite an output directory that contains checkpoints from a previous run when not resuming."""

    matmul_precision: Literal["highest", "high", "medium"] = "high"
    """Precision for float32 matrix multiplications. ``highest`` is full FP32 (required on ROCm/AMD GPUs to avoid catastrophic precision loss in softmax over large vocabularies). ``high`` enables TF32 on NVIDIA GPUs for a speedup with minor precision tradeoff. See ``torch.set_float32_matmul_precision``."""

    max_steps: int | None = None
    """Maximum training steps. If None, runs indefinitely."""

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

    loss_impl: Literal["liger", "torch", "liger_fused", "quack_fused"] = "torch"
    """Cross-entropy loss implementation. ``liger_fused`` fuses the lm_head projection with the CE loss to avoid materializing full logits. ``quack_fused`` uses quack-kernels for chunked linear + CE with CuTe DSL CUDA kernels."""

    heartbeat: HeartbeatConfig | None = None
    """BetterStack heartbeat configuration for monitoring training progress."""

    deployment: SFTDeploymentConfig = SingleNodeDeploymentConfig()

    slurm: SlurmConfig | None = None
    """SLURM configuration. When set, the run is submitted as a SLURM job instead of running locally."""

    dry_run: bool = False
    """Only validate and dump resolved configs, then exit early."""

    experimental: SFTExperimentalConfig = SFTExperimentalConfig()

    ### Pre-validation normalization

    @model_validator(mode="before")
    @classmethod
    def normalize_deployment(cls, data):
        if not isinstance(data, dict):
            return data
        deployment = data.get("deployment")
        if isinstance(deployment, dict) and deployment.get("type") == "multi_node":
            for key in ("num_gpus",):
                deployment.pop(key, None)
        return data

    ### Validate configs (e.g. raise for unsupported (combinations of) configs)

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
    def validate_deployment(self):
        if self.deployment.type == "multi_node" and self.slurm is None:
            raise ValueError("Must use SLURM for multi-node deployment.")
        return self

    @model_validator(mode="after")
    def validate_tree_training_v1(self):
        if self.data.type not in (
            "caterpillar_fake",
            "sft_caterpillar",
            "sft_raw_tool_caterpillar",
            "sft_raw_tool_caterpillar_grouped_branches",
        ):
            return self

        if self.model.cp != 1:
            raise ValueError("Tree Training v1 requires model.cp == 1")
        if self.model.ep != 1:
            raise ValueError("Tree Training v1 requires model.ep == 1")
        if self.model.attn not in ("sdpa", "flex_attention"):
            raise ValueError("Tree Training requires model.attn in {'sdpa', 'flex_attention'}")
        if self.model.impl not in ("hf", "auto"):
            raise ValueError("Tree Training v1 requires model.impl in {'hf', 'auto'}")
        if self.loss_impl not in ("torch", "liger"):
            raise ValueError("Tree Training v1 requires loss_impl in {'torch', 'liger'}")
        if self.data.pack_function != "cat":
            raise ValueError("Tree Training v1 requires data.pack_function == 'cat'")
        if self.data.micro_batch_size != 1:
            raise ValueError("Tree Training v1 requires data.micro_batch_size == 1")
        return self

    @model_validator(mode="after")
    def validate_pack_function(self):
        if self.model.cp > 1:
            if self.data.pack_function != "cat":
                raise ValueError("Packing function must be 'cat' when CP is enabled")
            if self.val is not None and self.val.data.pack_function != "cat":
                raise ValueError("Validation packing function must be 'cat' when CP is enabled")
        return self

    @model_validator(mode="after")
    def validate_cp_seq_len(self):
        if self.model.cp > 1:
            if self.data.seq_len % self.model.cp != 0:
                raise ValueError("Sequence length must be divisible by CP degree")
            if self.val is not None and self.val.data.seq_len % self.model.cp != 0:
                raise ValueError("Validation sequence length must be divisible by CP degree")
        return self

    @model_validator(mode="after")
    def validate_cp_micro_batch_size(self):
        if self.model.cp > 1:
            if self.data.micro_batch_size != 1:
                raise ValueError("Micro batch size must be 1 when CP is enabled")
            if self.val is not None and self.val.data.micro_batch_size != 1:
                raise ValueError("Validation micro batch size must be 1 when CP is enabled")
        return self

    @model_validator(mode="after")
    def validate_seq_len(self):
        if self.data.pack_function == "stack" and self.data.seq_len % 256 != 0:
            raise ValueError("The sequence length must be divisible by 256 when using pack function stack")
        if self.val is not None and self.val.data.pack_function == "stack" and self.val.data.seq_len % 256 != 0:
            raise ValueError("The validation sequence length must be divisible by 256 when using pack function stack")
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
    def validate_renderer_vs_vlm(self):
        if self.use_renderer and self.model.vlm is not None:
            raise ValueError(
                "use_renderer is not supported for VLMs. The renderer tokenizes "
                "text-only message dicts client-side and cannot handle image inputs."
            )
        return self

    @model_validator(mode="after")
    def validate_renderer_args(self):
        # pool_size is orchestrator-only. An in-process renderer pool exists
        # to amortize tokenization across concurrent rollouts in the
        # orchestrator (many async requests render at once, HF fast
        # tokenizers release the GIL during Rust encoding, so a pool of N
        # tokenizer copies parallelizes well). SFT has no such concurrency:
        # the StatefulDataLoader is constructed with num_workers=0, so the
        # main process tokenizes one example at a time, between training
        # steps. Across DP, each rank already owns its own renderer — an
        # implicit pool of size world_size. Pooling within a rank gives
        # nothing on top of that. Reject so callers don't silently set a
        # knob that does nothing; if SFT tokenization ever becomes a
        # bottleneck the fix is num_workers on the dataloader, not a pool.
        if self.renderer.pool_size is not None:
            raise ValueError(
                f"renderer.pool_size={self.renderer.pool_size!r} is only used by the orchestrator. "
                "SFT tokenizes synchronously (num_workers=0) and already gets one renderer per DP "
                "rank — an in-process pool adds nothing. If tokenization is a bottleneck, raise "
                "num_workers on the dataloader instead."
            )

        if self.use_renderer:
            return self

        renderer_args_set = []
        if self.renderer.name != "auto":
            renderer_args_set.append(f"renderer.name={self.renderer.name!r}")
        if self.renderer.tool_parser is not None:
            renderer_args_set.append(f"renderer.tool_parser={self.renderer.tool_parser!r}")
        if self.renderer.reasoning_parser is not None:
            renderer_args_set.append(f"renderer.reasoning_parser={self.renderer.reasoning_parser!r}")
        if self.renderer.preserve_all_thinking:
            renderer_args_set.append(f"renderer.preserve_all_thinking={self.renderer.preserve_all_thinking!r}")
        if self.renderer.preserve_thinking_between_tool_calls:
            renderer_args_set.append(
                f"renderer.preserve_thinking_between_tool_calls={self.renderer.preserve_thinking_between_tool_calls!r}"
            )

        if renderer_args_set:
            raise ValueError(
                "Renderer-specific args set without use_renderer=True: "
                f"{', '.join(renderer_args_set)}. Either enable the renderer or remove these knobs."
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
    def validate_opt_and_fsdp_offload(self):
        if self.optim.type == "muon" and self.model.fsdp_cpu_offload:
            raise ValueError("Muon optimizer does not support FSDP CPU offload")
        return self

    @model_validator(mode="after")
    def validate_and_disable_chunked_loss(self):
        if isinstance(self.model.fused_lm_head_token_chunk_size, int):
            raise ValueError(
                "Chunked loss is not supported for SFT training yet, please set "
                "`model.fused_lm_head_token_chunk_size` to 'disabled'"
            )

        self.model.fused_lm_head_token_chunk_size = "disabled"
        return self

    @model_validator(mode="after")
    def ep_only_with_custom_impl(self):
        if self.model.ep > 1 and self.model.impl not in ("custom", "auto"):
            raise ValueError("EP is only supported with the custom implementation or auto mode")

        return self

    ### Auto-setup and validate shared configs

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench is not None:
            self.max_steps = 4  # 1 Warmup + 3 Benchmark
            if self.ckpt:  # Do not checkpoint
                self.ckpt = None
        return self

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def auto_setup_deployment(self):
        if self.deployment.type == "multi_node":
            if self.deployment.nodes_per_fsdp_group is not None:
                if self.deployment.num_nodes % self.deployment.nodes_per_fsdp_group != 0:
                    raise ValueError(
                        f"deployment.num_nodes ({self.deployment.num_nodes}) must be divisible by "
                        f"deployment.nodes_per_fsdp_group ({self.deployment.nodes_per_fsdp_group})"
                    )
                self.model.dp_replicate = self.deployment.num_nodes // self.deployment.nodes_per_fsdp_group
        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        if self.slurm is not None and self.slurm.template_path is None:
            templates_dir = find_package_resource("templates")
            if templates_dir is not None:
                if self.deployment.type == "single_node":
                    self.slurm.template_path = templates_dir / "single_node_sft.sbatch.j2"
                else:
                    self.slurm.template_path = templates_dir / "multi_node_sft.sbatch.j2"
        return self

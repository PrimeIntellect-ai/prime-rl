import uuid
import warnings
from pathlib import Path
from typing import Annotated, Literal, TypeAlias
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, model_validator
from renderers import AutoRendererConfig, DefaultRendererConfig, RendererConfig
from renderers.base import MODEL_RENDERER_MAP

from prime_rl.configs.evaluator import OnlineEvalConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.monitors import MonitorsConfig
from prime_rl.configs.shared import (
    EnvVars,
    HeartbeatConfig,
    ResumeConfig,
    RunConfig,
    SlurmConfig,
    TrainerLogConfig,
)
from prime_rl.configs.trainer import (
    AdamWConfig,
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


DataConfig: TypeAlias = Annotated[FakeDataConfig | SFTDataConfig, Field(discriminator="type")]


class BaseDeploymentConfig(BaseConfig):
    gpus_per_node: int = 8
    """GPUs per node."""


class SingleNodeDeploymentConfig(BaseDeploymentConfig):
    type: Literal["single_node"] = "single_node"

    num_train_gpus: int = 1
    """GPUs allocated to the trainer."""

    num_infer_gpus: int = Field(1, validation_alias=AliasChoices("num_infer_gpus", "num_eval_gpus"))
    """GPUs allocated to inference for online evals (alias: ``num_eval_gpus``). Only used when an ``[inference]`` block is configured."""

    @model_validator(mode="after")
    def validate_gpu_count(self):
        if self.num_train_gpus > self.gpus_per_node:
            raise ValueError(f"num_train_gpus ({self.num_train_gpus}) exceeds gpus_per_node ({self.gpus_per_node}).")
        return self


class MultiNodeDeploymentConfig(BaseDeploymentConfig):
    type: Literal["multi_node"] = "multi_node"

    num_train_nodes: int = Field(2, ge=1)
    """Training nodes."""

    num_infer_nodes: int = Field(0, ge=0, validation_alias=AliasChoices("num_infer_nodes", "num_eval_nodes"))
    """Inference nodes for online evals (alias: ``num_eval_nodes``). Submitted as a separate SLURM job, decoupled
    from the trainer job: the handoff is weight checkpoints on the shared filesystem,
    so the eval job can outlive the trainer job (evals drain after training ends) and
    exits after evaluating the final checkpoint."""

    nodes_per_fsdp_group: int | None = None
    """Nodes per FSDP island. Auto-sets ``model.dp_replicate = num_train_nodes / nodes_per_fsdp_group``."""


SFTDeploymentConfig: TypeAlias = Annotated[
    SingleNodeDeploymentConfig | MultiNodeDeploymentConfig, Field(discriminator="type")
]


class SFTConfig(BaseConfig):
    model: ModelConfig = ModelConfig()

    env_vars: EnvVars = {}
    """Extra environment variables for the SFT trainer process(es). Merged on top of the launcher defaults."""

    tokenizer: TokenizerConfig = TokenizerConfig()

    renderer: RendererConfig = AutoRendererConfig()
    """Renderer config. Defaults to auto-selecting from the tokenizer model name."""

    data: DataConfig = SFTDataConfig()

    val: SFTValConfig | None = None
    """Validation configuration. If None, no validation runs."""

    eval: OnlineEvalConfig | None = None
    """Online evaluation configuration: rollout-based evals against a live inference
    server that reloads the trainer's HF weight checkpoints from disk. If None, no
    online evals run."""

    inference: InferenceConfig | None = None
    """Inference server for online evals. If None (with ``eval`` set), the launcher
    does not start a server and evals connect to ``eval.client.base_url``."""

    optim: OptimizerConfig = AdamWConfig()

    scheduler: SchedulerConfig = ConstantSchedulerConfig()

    ckpt: CheckpointConfig | None = None

    resume: ResumeConfig | None = None
    """Resume the run from a checkpoint (point at it with the previous run's ``run.name``). Without ``[ckpt]`` the run loads the checkpoint but saves no new ones. If None, does not resume."""

    log: TrainerLogConfig = TrainerLogConfig()

    monitors: MonitorsConfig = MonitorsConfig()
    """Metric monitors (``monitors.wandb``, ``monitors.file``)."""

    run: RunConfig = Field(default_factory=RunConfig)
    """Run metadata. ``run.name`` names the run directory under ``output_dir``."""

    output_dir: Path = Path("outputs")
    """Directory that groups related runs. Each run writes its artifacts (checkpoints, logs, ...) to ``output_dir / run.name``. Should be a persistent directory with enough disk space."""

    clean: bool = False
    """Delete the run directory (``output_dir / run.name``) before starting training. Required to overwrite a run directory that contains artifacts from a previous run when not resuming."""

    @property
    def run_dir(self) -> Path:
        assert self.run.dir is not None  # resolved at construction
        return self.output_dir / self.run.dir

    @model_validator(mode="after")
    def auto_setup_run_identity(self):
        """Auto-generate the run name (``<dataset>--<model>--<short-id>``) when unset and
        default the run directory and W&B run name to it when not set explicitly."""
        if self.run.name is None:
            dataset = str(getattr(self.data, "name", "")).split("/")[-1]
            model = self.model.name.split("/")[-1]
            parts = [part for part in (dataset, model) if part]
            self.run.name = "--".join([*parts, uuid.uuid4().hex[:8]]).lower()
        if self.run.dir is None:
            self.run.dir = self.run.name
        if self.monitors.wandb is not None and self.monitors.wandb.name is None:
            self.monitors.wandb.name = self.run.name
        return self

    matmul_precision: Literal["highest", "high", "medium"] = "high"
    """Precision for float32 matrix multiplications. ``highest`` is full FP32 (required on ROCm/AMD GPUs to avoid catastrophic precision loss in softmax over large vocabularies). ``high`` enables TF32 on NVIDIA GPUs for a speedup with minor precision tradeoff. See ``torch.set_float32_matmul_precision``."""

    max_steps: int | None = None
    """Maximum training steps. If None, runs indefinitely."""

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

    deployment: SFTDeploymentConfig = SingleNodeDeploymentConfig()

    slurm: SlurmConfig | None = None
    """SLURM configuration. When set, the run is submitted as a SLURM job instead of running locally."""

    dry_run: bool = False
    """Only validate and dump resolved configs, then exit early."""

    ### Pre-validation normalization

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

    @model_validator(mode="before")
    @classmethod
    def propagate_model_to_inference(cls, data):
        """Fill ``inference.vllm.model`` from ``model.name`` before InferenceConfig
        construction, so its parser auto-resolution sees the right model name."""
        if not isinstance(data, dict):
            return data
        inference = data.get("inference")
        model = data.get("model")
        name = model.get("name") if isinstance(model, dict) else None
        if isinstance(inference, dict) and name:
            vllm = inference.setdefault("vllm", {})
            if isinstance(vllm, dict):
                vllm.setdefault("model", name)
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
    def validate_deployment(self):
        if self.deployment.type == "multi_node" and self.slurm is None:
            raise ValueError("Must use SLURM for multi-node deployment.")
        return self

    @model_validator(mode="after")
    def auto_setup_online_eval(self):
        """Wire online evals: require weight checkpoints (the disk handoff to
        inference) and connect the eval client to the launcher-managed server."""
        if self.eval is None:
            if self.inference is not None:
                raise ValueError("[inference] is only used for online evals — add an [eval] block or remove it.")
            return self

        # The trainer's HF weight checkpoints are how inference picks up new policies.
        if self.ckpt is None:
            self.ckpt = CheckpointConfig()
        if self.ckpt.weights is None or self.ckpt.skip_gather_master_weights:
            raise ValueError(
                "Online evals require HF weight checkpoints. Enable ckpt.weights and "
                "disable ckpt.skip_gather_master_weights."
            )

        if self.ckpt.keep_last is not None or self.ckpt.keep_interval is not None:
            warnings.warn(
                "ckpt.keep_last / ckpt.keep_interval can delete a weight checkpoint before the "
                "evaluator consumes it when evals run slower than training - such steps are "
                "skipped with a warning instead of evaluated.",
                stacklevel=2,
            )

        if self.deployment.type == "multi_node":
            # Decoupled deployment: the launcher submits a dedicated SLURM job running the
            # inference pool (one engine per DP rank behind a router) plus the evaluator.
            if self.inference is None:
                raise ValueError(
                    "Multi-node online evals require an [inference] block - the launcher submits "
                    "a dedicated SLURM job running the inference pool and the evaluator."
                )
            if self.deployment.num_infer_nodes < 1:
                raise ValueError("Online evals on a multi-node deployment require deployment.num_infer_nodes >= 1.")
            if self.inference.router is None:
                raise ValueError(
                    "Multi-node online evals require an inference router - the eval job starts one "
                    "router in front of the per-rank engines. Remove inference.router = 'None'."
                )
            if self.inference.vllm.model != self.model.name:
                raise ValueError(
                    f"inference.vllm.model ({self.inference.vllm.model}) does not match model.name "
                    f"({self.model.name}). Remove inference.vllm.model to inherit it."
                )
            if self.deployment.gpus_per_node % self.inference.vllm.tensor_parallel_size != 0:
                raise ValueError(
                    f"deployment.gpus_per_node ({self.deployment.gpus_per_node}) must be divisible by "
                    f"inference.vllm.tensor_parallel_size ({self.inference.vllm.tensor_parallel_size})."
                )
            if self.inference.weight_broadcast.type != "filesystem":
                raise ValueError(
                    "Online evals reload weights from disk — inference.weight_broadcast.type must be 'filesystem'."
                )
            if self.max_steps is None:
                warnings.warn(
                    "Online evals without max_steps: the evaluator never sees a final checkpoint, "
                    "so the eval SLURM job holds its allocation until its walltime.",
                    stacklevel=2,
                )
            # The client is wired at runtime by the eval sbatch script (the router and
            # per-rank admin URLs are only known once SLURM assigns hosts).
            return self

        if self.inference is None:
            warnings.warn(
                "Online evals are configured without an [inference] block - the launcher will not "
                f"start an inference server. Make sure one is running at eval.client.base_url "
                f"({self.eval.client.base_url}) with weight_broadcast.type = 'filesystem', "
                "otherwise the evaluator will hang waiting for it. If a router fronts the "
                "deployment, set eval.client.admin_base_url to the engine URLs - admin ops "
                "(pause/update_weights/resume) must bypass the router.",
                stacklevel=2,
            )
            return self

        total_gpus = self.deployment.num_train_gpus + self.deployment.num_infer_gpus
        if total_gpus > self.deployment.gpus_per_node:
            raise ValueError(
                f"Total GPU count ({total_gpus} = {self.deployment.num_train_gpus} train + "
                f"{self.deployment.num_infer_gpus} infer) exceeds gpus_per_node "
                f"({self.deployment.gpus_per_node})."
            )

        if self.inference.vllm.model != self.model.name:
            raise ValueError(
                f"inference.vllm.model ({self.inference.vllm.model}) does not match model.name "
                f"({self.model.name}). Remove inference.vllm.model to inherit it."
            )

        # Fill inference capacity with DP ranks (mirrors RLConfig.auto_setup_deployment).
        num_infer_gpus = self.deployment.num_infer_gpus
        vllm = self.inference.vllm
        if num_infer_gpus != vllm.data_parallel_size * vllm.tensor_parallel_size:
            if num_infer_gpus % vllm.tensor_parallel_size != 0:
                raise ValueError(
                    f"deployment.num_infer_gpus ({num_infer_gpus}) must be divisible by "
                    f"inference.vllm.tensor_parallel_size ({vllm.tensor_parallel_size})."
                )
            vllm.data_parallel_size = num_infer_gpus // vllm.tensor_parallel_size
        if vllm.api_server_count < vllm.data_parallel_size and not vllm.enable_lora:
            vllm.api_server_count = vllm.data_parallel_size
        if self.inference.weight_broadcast.type != "filesystem":
            raise ValueError(
                "Online evals reload weights from disk — inference.weight_broadcast.type must be 'filesystem'."
            )

        host = self.inference.server.host or "localhost"
        client = self.eval.client
        if "base_url" not in client.model_fields_set:
            client.base_url = f"http://{host}:{self.inference.server.port}/v1"
        elif urlparse(client.base_url).port != self.inference.server.port:
            raise ValueError(
                f"eval.client.base_url port ({urlparse(client.base_url).port}) does not match "
                f"inference.server.port ({self.inference.server.port})."
            )
        if self.inference.router is not None and "admin_base_url" not in client.model_fields_set:
            # Admin ops (pause/update_weights/resume) must bypass the router and hit
            # the engine directly.
            client.admin_base_url = [f"http://{host}:{self.inference.backend_port}/v1"]
        return self

    @model_validator(mode="after")
    def validate_typed_renderer(self):
        """Require a typed renderer whenever SFT renders real samples."""
        if self.data.type == "fake" and self.val is None:
            return self

        model_id = self.tokenizer.name or self.model.name
        if isinstance(self.renderer, AutoRendererConfig):
            if model_id in MODEL_RENDERER_MAP:
                return self
            reason = f"no typed renderer is registered for {model_id!r}"
        elif isinstance(self.renderer, DefaultRendererConfig):
            reason = "renderer.name='default' selects DefaultRenderer"
        else:
            return self

        raise ValueError(
            f"SFT requires a typed renderer with sampled-token and content attribution, but {reason}. "
            "Implement and register the renderer in the renderers package, or explicitly select an existing "
            "typed renderer only when its template is verified to match."
        )

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
    def vlm_freeze_incompatible_with_lora(self):
        if self.model.vlm is not None and not self.model.vlm.freeze_vision_encoder and self.model.lora is not None:
            raise ValueError(
                "freeze_vision_encoder=false is incompatible with LoRA. "
                "LoRA freezes all non-adapter parameters including the vision encoder."
            )
        return self

    @model_validator(mode="after")
    def validate_vlm_constraints(self):
        if self.model.vlm is None:
            return self
        if self.model.optimization_dtype != "bfloat16" or self.model.reduce_dtype != "bfloat16":
            raise ValueError(
                "VLM models must use optimization_dtype='bfloat16' and reduce_dtype='bfloat16' to match vLLM inference."
            )
        if self.data.micro_batch_size != 1:
            raise ValueError("VLM SFT requires data.micro_batch_size = 1.")
        if self.val is not None and self.val.data.micro_batch_size != 1:
            raise ValueError("VLM SFT requires val.data.micro_batch_size = 1.")
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
    def validate_opt_and_fsdp_offload(self):
        if self.optim.type == "muon" and self.model.fsdp_cpu_offload:
            raise ValueError("Muon optimizer does not support FSDP CPU offload")
        return self

    @model_validator(mode="after")
    def ep_only_with_custom_impl(self):
        if self.model.ep != 1 and self.model.ep != "auto" and self.model.impl not in ("custom", "auto"):
            raise ValueError("EP is only supported with the custom implementation or auto mode")

        return self

    ### Auto-setup and validate shared configs

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
                if self.deployment.num_train_nodes % self.deployment.nodes_per_fsdp_group != 0:
                    raise ValueError(
                        f"deployment.num_train_nodes ({self.deployment.num_train_nodes}) must be divisible by "
                        f"deployment.nodes_per_fsdp_group ({self.deployment.nodes_per_fsdp_group})"
                    )
                self.model.dp_replicate = self.deployment.num_train_nodes // self.deployment.nodes_per_fsdp_group
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

import uuid
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.inference import WeightBroadcastConfig as InferenceWeightBroadcastConfig
from prime_rl.configs.orchestrator import (
    FileSystemWeightBroadcastConfig as OrchestratorFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.orchestrator import (
    NCCLWeightBroadcastConfig as OrchestratorNCCLWeightBroadcastConfig,
)
from prime_rl.configs.orchestrator import (
    NIXLWeightBroadcastConfig as OrchestratorNIXLWeightBroadcastConfig,
)
from prime_rl.configs.orchestrator import (
    OrchestratorConfig,
)
from prime_rl.configs.shared import (
    EnvVars,
    FileMonitorConfig,
    ResumeConfig,
    RunConfig,
    SlurmConfig,
    TransportConfig,
    VLMConfig,
)
from prime_rl.configs.trainer import (
    FileSystemWeightBroadcastConfig as TrainerFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.trainer import (
    NCCLWeightBroadcastConfig as TrainerNCCLWeightBroadcastConfig,
)
from prime_rl.configs.trainer import (
    NIXLWeightBroadcastConfig as TrainerNIXLWeightBroadcastConfig,
)
from prime_rl.configs.trainer import (
    TokenizerConfig,
    TrainerConfig,
)
from prime_rl.utils.config import BaseConfig, find_package_resource
from prime_rl.utils.validation import (
    propagate_shared_fields,
    validate_shared_ckpt_config,
    validate_shared_max_steps,
    validate_shared_model_name,
    validate_shared_seq_len,
    validate_shared_tokenizer,
    validate_shared_wandb_config,
    validate_shared_weight_broadcast,
)


class SharedLogConfig(BaseConfig):
    level: str | None = None
    """Log level for trainer, orchestrator, and inference. When unset, each sub-config's own log level applies (defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``)."""

    json_logging: bool = False
    """Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.). Propagated to trainer, orchestrator, and inference."""


class SharedWandbConfig(BaseConfig):
    project: str | None = "prime-rl"
    """W&B project."""

    entity: str | None = None
    """W&B entity."""

    name: str | None = None
    """W&B run name."""

    group: str | None = None
    """W&B group."""

    tags: list[str] | None = None
    """W&B tags attached to the run."""

    offline: bool | None = False
    """Run W&B in offline mode. Incompatible with shared mode, which is always on for the ``rl`` entrypoint."""

    @model_validator(mode="after")
    def validate_not_offline(self):
        if self.offline:
            raise ValueError(
                "W&B shared mode is always on for the rl entrypoint and requires server "
                "connectivity; wandb.offline = true is not supported. Use offline mode "
                "via the sub-config wandb blocks (trainer.wandb.offline, "
                "orchestrator.wandb.offline) if you really need it per-process."
            )
        return self


class SharedCheckpointConfig(BaseConfig):
    output_dir: Path | None = None
    """Override directory for checkpoints and weights. When set, checkpoints and weight snapshots are written here instead of under the trainer ``output_dir``."""

    interval: int | None = None
    """Interval at which to save checkpoints."""

    keep_last: int | None = Field(None, ge=1)
    """Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency."""

    keep_interval: int | None = Field(None, ge=1)
    """Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping."""


class SharedModelConfig(BaseConfig):
    name: str = "Qwen/Qwen3-0.6B"
    """HF model name or local path."""

    vlm: "VLMConfig | None" = None
    """VLM configuration. Set this to enable vision-language model support."""


class SharedInMemoryWeightBroadcastConfig(BaseConfig):
    host: str = "localhost"
    """Weight transfer host."""

    port: int
    """Weight transfer port."""

    timeout: int = 1200
    """Timeout in seconds for in-memory weight transfer."""


class SharedNCCLWeightBroadcastConfig(SharedInMemoryWeightBroadcastConfig):
    type: Literal["nccl"] = "nccl"

    port: int = 29501
    """Port for NCCL weight broadcast."""

    quantize_in_weight_transfer: bool = False
    """Use kernel-format FP8 quantized NCCL transfer for weight updates. When disabled, uses default HF checkpoint-format transfer."""


class SharedNIXLWeightBroadcastConfig(SharedInMemoryWeightBroadcastConfig):
    type: Literal["nixl"] = "nixl"

    port: int = 8001
    """ModelExpress gRPC port."""

    session_id: str = "default"
    """ModelExpress session ID."""


class SharedFileSystemWeightBroadcastConfig(BaseConfig):
    type: Literal["filesystem"] = "filesystem"


SharedWeightBroadcastConfig: TypeAlias = Annotated[
    SharedFileSystemWeightBroadcastConfig | SharedNCCLWeightBroadcastConfig | SharedNIXLWeightBroadcastConfig,
    Field(discriminator="type"),
]


class BaseDeploymentConfig(BaseConfig):
    gpus_per_node: int = 8
    """GPUs per node."""


class SingleNodeDeploymentConfig(BaseDeploymentConfig):
    type: Literal["single_node"] = "single_node"

    num_train_gpus: int = 1
    """GPUs allocated to the trainer."""

    num_infer_gpus: int = 1
    """GPUs allocated to inference."""

    @model_validator(mode="after")
    def validate_gpu_count(self):
        total = self.num_train_gpus + self.num_infer_gpus
        if total > self.gpus_per_node:
            raise ValueError(
                f"Total GPU count ({total} = {self.num_train_gpus} train + {self.num_infer_gpus} infer)"
                f" exceeds gpus_per_node ({self.gpus_per_node})."
            )
        return self


class MultiNodeDeploymentConfig(BaseDeploymentConfig):
    type: Literal["multi_node"] = "multi_node"

    num_train_nodes: int
    """Training nodes."""

    num_infer_nodes: int | None = Field(None, ge=0)
    """Inference nodes per replica. If unset, inferred from ``inference.deployment``. Set to 0 to skip inference and orchestrator (requires fake data)."""

    num_infer_replicas: int = Field(1, ge=1)
    """Independent inference replicas. Total inference nodes = ``num_infer_nodes * num_infer_replicas``."""

    nodes_per_fsdp_group: int | None = None
    """Training nodes per FSDP island. Auto-sets ``trainer.dp_replicate = num_train_nodes / nodes_per_fsdp_group``."""

    orchestrator_on_inference: bool = False
    """Run the orchestrator on the last inference node instead of trainer rank 0 (frees host RAM on the trainer node)."""

    @property
    def infer_nodes_per_replica(self) -> int:
        return self.num_infer_nodes or 0

    @property
    def total_infer_nodes(self) -> int:
        return self.infer_nodes_per_replica * self.num_infer_replicas


DeploymentConfig: TypeAlias = Annotated[
    SingleNodeDeploymentConfig | MultiNodeDeploymentConfig, Field(discriminator="type")
]


class RLConfig(BaseConfig):
    trainer: TrainerConfig

    orchestrator: OrchestratorConfig

    inference: InferenceConfig | None = None
    """Inference server configuration. If None, the rl entrypoint will not start an inference server (useful for manually started servers)."""

    env_vars: EnvVars = {}
    """Extra environment variables for every launched RL component. Component-specific env_vars override these."""

    run: RunConfig = Field(default_factory=RunConfig)
    """Run metadata. ``run.name`` names the run directory under ``output_dir``."""

    output_dir: Path = Path("outputs")
    """Directory that groups related runs. Each run writes its artifacts to ``output_dir / run.name``."""

    clean: bool = False
    """Delete the run directory (``output_dir / run.name``) before starting training. Required to overwrite a run directory that contains artifacts from a previous run when not resuming."""

    @property
    def run_dir(self) -> Path:
        assert self.run.dir is not None  # resolved at construction
        return self.output_dir / self.run.dir

    def _resolve_run_name(self) -> None:
        """Fill the auto-generated run name and directory (idempotent — called by every
        validator that reads them, so it does not depend on validator ordering)."""
        if self.run.name is None:
            envs = "+".join(dict.fromkeys(source.resolved_name for source in self.orchestrator.train.source))
            model = self.trainer.model.name.split("/")[-1]
            self.run.name = f"{envs or 'no-env'}--{model}--{uuid.uuid4().hex[:8]}".lower()
        if self.run.dir is None:
            self.run.dir = self.run.name

    ### Shared configurations

    log: SharedLogConfig = SharedLogConfig()
    """Shared log config. Propagated to trainer and orchestrator."""

    ckpt: SharedCheckpointConfig | None = None
    """Shared checkpoint config. If None, falls back to the sub-config checkpoint settings."""

    resume: ResumeConfig | None = None
    """Resume the run from a checkpoint (point at it with the previous run's ``run.name``). Without ``[ckpt]`` the run loads the checkpoint but saves no new ones. If None, does not resume."""

    wandb: SharedWandbConfig | None = None
    """Shared W&B config. If None, falls back to the sub-config W&B settings."""

    file_monitor: FileMonitorConfig | None = None
    """Shared local JSONL metric sink. If set, enables ``<output_dir>/metrics.jsonl`` on both trainer and orchestrator. If None, falls back to the sub-config settings."""

    model: SharedModelConfig | None = None
    """Shared model config. If None, falls back to the sub-config model settings."""

    tokenizer: TokenizerConfig | None = None
    """Shared tokenizer config. Propagated to trainer, orchestrator, and inference. If None, each component uses its own tokenizer config (defaulting to model name)."""

    max_steps: int | None = None
    """Shared maximum training steps. If None, falls back to the sub-config ``max_steps``."""

    seq_len: int | None = None
    """Shared sequence length. Propagates to ``trainer.model.seq_len`` and ``orchestrator.seq_len`` only when those values were not explicitly set; explicit per-component values always win."""

    weight_broadcast: SharedWeightBroadcastConfig | None = None

    rollout_transport: TransportConfig | None = None

    deployment: DeploymentConfig = SingleNodeDeploymentConfig()

    slurm: SlurmConfig | None = None
    """SLURM configuration. If None, runs locally."""

    dry_run: bool = False
    """Only validate and dump resolved configs, then exit early."""

    ### Validate configs (e.g. raise for unsupported (combinations of) configs)

    @model_validator(mode="after")
    def auto_setup_infer_nodes(self):
        if self.deployment.type != "multi_node":
            return self

        if self.inference is None:
            inferred_nodes = 0
        elif self.inference.deployment.type == "multi_node":
            inferred_nodes = self.inference.deployment.num_nodes
        elif self.inference.deployment.type == "disaggregated":
            inferred_nodes = self.inference.deployment.num_nodes
        else:
            inferred_nodes = 1

        if self.deployment.num_infer_nodes is None:
            self.deployment.num_infer_nodes = inferred_nodes
        elif (
            self.inference is not None
            and self.inference.deployment.type == "multi_node"
            and self.deployment.num_infer_nodes != inferred_nodes
        ):
            raise ValueError(
                f"deployment.num_infer_nodes ({self.deployment.num_infer_nodes}) must equal "
                f"inference.deployment.num_nodes ({inferred_nodes}) for multi-node inference."
            )
        return self

    @model_validator(mode="after")
    def validate_deployment(self):
        if self.deployment.type == "multi_node":
            if self.slurm is None:
                raise ValueError("Must use SLURM for multi-node deployment.")
            num_infer_nodes = self.deployment.infer_nodes_per_replica
            if num_infer_nodes > 0 and not self.inference:
                raise ValueError("Must configure inference when using multi-node deployment with inference nodes.")
            if num_infer_nodes == 0 and self.inference:
                raise ValueError(
                    "Cannot configure inference with num_infer_nodes = 0. "
                    "Either set num_infer_nodes > 0 or remove the inference config."
                )
            if num_infer_nodes == 0 and not self.trainer.data.fake:
                raise ValueError(
                    "Must use fake data (trainer.data.fake) when num_infer_nodes = 0, "
                    "since no orchestrator or inference server will be running."
                )
        return self

    @model_validator(mode="after")
    def validate_enough_devices_for_nccl(self):
        if self.deployment.type == "single_node":
            if self.trainer.weight_broadcast.type == "nccl":
                if self.deployment.num_train_gpus + self.deployment.num_infer_gpus < 2:
                    raise ValueError(
                        "NCCL weight broadcast requires at least 2 GPUs to build the broadcast process group."
                    )
        return self

    @model_validator(mode="after")
    def validate_quantize_in_weight_transfer(self):
        if not isinstance(self.weight_broadcast, SharedNCCLWeightBroadcastConfig):
            return self

        if not self.weight_broadcast.quantize_in_weight_transfer:
            return self

        if self.inference is None:
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires an inference config.")

        if self.trainer.model.impl != "custom":
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires trainer.model.impl = 'custom'.")

        return self

    ### Auto-setup shared configs (before sub-config construction)

    @model_validator(mode="before")
    @classmethod
    def auto_setup_shared_configs(cls, data: Any) -> Any:
        """Propagate shared top-level fields into sub-config dicts before sub-configs
        are constructed. See ``validation.propagate_shared_fields`` for the full
        propagation table, transforms, and the mutex rule.
        """
        return propagate_shared_fields(data)

    @model_validator(mode="after")
    def auto_setup_run_dir(self):
        """Point trainer and orchestrator at the run directory (``output_dir / run.name``).

        The sub-configs' ``output_dir`` is fully derived here: sub-processes spawned by
        the launcher receive the resolved run directory and never re-derive it.
        """
        self._resolve_run_name()
        run_dir = self.run_dir
        for sub in (self.trainer, self.orchestrator):
            if "output_dir" in sub.model_fields_set and sub.output_dir != run_dir:
                raise ValueError(
                    f"{type(sub).__name__}.output_dir ({sub.output_dir}) conflicts with the run directory "
                    f"({run_dir}). Under the rl entrypoint, sub-config output directories are derived from "
                    "output_dir / run.name — set those instead."
                )
            sub.output_dir = run_dir
        return self

    @model_validator(mode="after")
    def auto_setup_resume(self):
        """Propagate the top-level resume onto the sub-configs."""
        if self.resume is None:
            return self
        self.trainer.resume = self.resume.model_copy()
        self.orchestrator.resume = self.resume.model_copy()
        return self

    @model_validator(mode="after")
    def auto_setup_run_identity(self):
        """Default the W&B and Prime platform run names to ``run.name``.

        Explicit names always win: only unset names inherit. Runs after the
        orchestrator's own ``auto_setup_prime_monitor_run_name``, so an explicitly
        set W&B name still takes precedence for the platform run name. The run
        identity itself is runtime-only ($PRL_RUN_ID / $PRL_RUN_NAME, set by the
        ``rl`` entrypoint), never sub-config.
        """
        self._resolve_run_name()
        for wandb in (self.wandb, self.trainer.wandb, self.orchestrator.wandb):
            if wandb is not None and wandb.name is None:
                wandb.name = self.run.name
        prime_monitor = self.orchestrator.prime_monitor
        if prime_monitor is not None and prime_monitor.run_name is None:
            prime_monitor.run_name = self.run.name
        return self

    ### Validate shared configs (after sub-config construction)

    @model_validator(mode="after")
    def validate_shared_configs(self):
        """Validate consistency of shared configs across trainer, orchestrator, and inference."""
        validate_shared_model_name(self.trainer, self.orchestrator, self.inference)
        validate_shared_tokenizer(self.trainer, self.orchestrator, self.inference)
        validate_shared_max_steps(self.trainer, self.orchestrator)
        validate_shared_seq_len(self.trainer, self.orchestrator)
        validate_shared_ckpt_config(self.trainer, self.orchestrator)
        validate_shared_wandb_config(self.trainer, self.orchestrator)
        return self

    @model_validator(mode="after")
    def auto_setup_weight_broadcast(self):
        """Auto-setup shared weight broadcast config for trainer, orchestrator, and inference.

        Defaults to NCCL broadcast when no ``weight_broadcast`` is configured. Falls back to
        filesystem when LoRA is enabled (not yet supported by in-memory transfer) or when no
        inference server is configured.
        """
        if self.weight_broadcast is None:
            if self.trainer.model.lora is not None or self.inference is None:
                self.weight_broadcast = SharedFileSystemWeightBroadcastConfig()
            else:
                self.weight_broadcast = SharedNCCLWeightBroadcastConfig()
        if self.weight_broadcast.type != "filesystem" and self.trainer.model.lora is not None:
            raise ValueError(
                "LoRA training is not yet supported with in-memory weight broadcast. "
                "Set weight_broadcast.type = 'filesystem'."
            )
        if self.weight_broadcast.type in ("nccl", "nixl"):
            inference_world_size = (
                self.inference.vllm.data_parallel_size * self.inference.vllm.tensor_parallel_size
                if self.inference
                else 1
            )
            common_config = dict(
                host=self.weight_broadcast.host,
                port=self.weight_broadcast.port,
                timeout=self.weight_broadcast.timeout,
                inference_world_size=inference_world_size,
            )
            if self.weight_broadcast.type == "nccl":
                transport_config = dict(
                    quantize_in_weight_transfer=self.weight_broadcast.quantize_in_weight_transfer,
                )
                trainer_config_type = TrainerNCCLWeightBroadcastConfig
                orchestrator_config_type = OrchestratorNCCLWeightBroadcastConfig
            else:
                transport_config = dict(session_id=self.weight_broadcast.session_id)
                trainer_config_type = TrainerNIXLWeightBroadcastConfig
                orchestrator_config_type = OrchestratorNIXLWeightBroadcastConfig
            self.trainer.weight_broadcast = trainer_config_type(**common_config, **transport_config)
            self.orchestrator.weight_broadcast = orchestrator_config_type(**common_config, **transport_config)
        elif self.weight_broadcast.type == "filesystem":
            self.trainer.weight_broadcast = TrainerFileSystemWeightBroadcastConfig()
            self.orchestrator.weight_broadcast = OrchestratorFileSystemWeightBroadcastConfig()
        if self.inference is not None:
            self.inference.weight_broadcast = InferenceWeightBroadcastConfig(type=self.weight_broadcast.type)

        validate_shared_weight_broadcast(self.trainer, self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def auto_setup_rollout_transport(self):
        """Resolve the shared ``rollout_transport`` from the sub-configs so the launcher can
        gate multi-node ZMQ host injection on it (mirrors ``auto_setup_weight_broadcast``).

        ``rollout_transport`` may be set either as the shared block (propagated down to both
        sub-configs by ``propagate_shared_fields``) or directly on
        ``trainer.rollout_transport`` / ``orchestrator.rollout_transport`` (the documented
        fallback). Either way the shared field must reflect the resolved per-component
        transport, otherwise the launcher would leave ZMQ trainers connecting to localhost.
        """
        if self.trainer.rollout_transport.type != self.orchestrator.rollout_transport.type:
            raise ValueError(
                "trainer.rollout_transport.type "
                f"({self.trainer.rollout_transport.type!r}) != orchestrator.rollout_transport.type "
                f"({self.orchestrator.rollout_transport.type!r}); set the shared [rollout_transport] "
                "block or make both sub-configs the same type."
            )
        if self.rollout_transport is None:
            self.rollout_transport = self.trainer.rollout_transport
        return self

    @model_validator(mode="after")
    def validate_eplb_requires_quantized_weight_transfer(self):
        if self.inference is None or not self.inference.vllm.enable_eplb:
            return self

        # TODO(matej): check if weight reloading works itself before supporting EPLB without quantized transfer.
        trainer_weight_broadcast = self.trainer.weight_broadcast
        if trainer_weight_broadcast.type != "nccl" or not trainer_weight_broadcast.quantize_in_weight_transfer:
            raise ValueError(
                "inference.vllm.enable_eplb requires weight_broadcast.type = 'nccl' and "
                "weight_broadcast.quantize_in_weight_transfer = true."
            )

        return self

    @model_validator(mode="after")
    def auto_setup_lora(self):
        if self.trainer.model.lora is not None:
            if self.orchestrator.model.lora is None:
                from prime_rl.configs.orchestrator import LoRAConfig

                self.orchestrator.model.lora = LoRAConfig()

            if (
                self.orchestrator.model.lora.rank is not None
                and self.orchestrator.model.lora.rank != self.trainer.model.lora.rank
            ):
                raise ValueError(
                    f"orchestrator.model.lora.rank ({self.orchestrator.model.lora.rank}) conflicts with "
                    f"trainer.model.lora.rank ({self.trainer.model.lora.rank}). "
                    f"Remove orchestrator.model.lora.rank to inherit from trainer, or update trainer.model.lora.rank to match."
                )

            if (
                self.orchestrator.model.lora.alpha is not None
                and self.orchestrator.model.lora.alpha != self.trainer.model.lora.alpha
            ):
                raise ValueError(
                    f"orchestrator.model.lora.alpha ({self.orchestrator.model.lora.alpha}) conflicts with "
                    f"trainer.model.lora.alpha ({self.trainer.model.lora.alpha}). "
                    f"Remove orchestrator.model.lora.alpha to inherit from trainer, or update trainer.model.lora.alpha to match."
                )

            if self.orchestrator.model.lora.rank is None:
                self.orchestrator.model.lora.rank = self.trainer.model.lora.rank

            if self.orchestrator.model.lora.alpha is None:
                self.orchestrator.model.lora.alpha = self.trainer.model.lora.alpha

            if self.orchestrator.model.lora.name is None:
                self.orchestrator.model.lora.name = (
                    f"r{self.orchestrator.model.lora.rank}-a{self.orchestrator.model.lora.alpha}"
                )

            if self.inference is not None:
                self.inference.vllm.enable_lora = True
                self.inference.vllm.max_lora_rank = self.trainer.model.lora.rank
            else:
                warnings.warn(
                    "LoRA is enabled, but inference is not configured. When manually starting the inference server, "
                    "make sure to set --enable_lora and --max-lora-rank.",
                    stacklevel=2,
                )

        return self

    @model_validator(mode="after")
    def auto_setup_router_replay(self):
        if self.trainer.enable_router_replay:
            if self.inference is not None:
                if self.inference.vllm.enable_return_routed_experts is False:
                    warnings.warn(
                        "Router replay is enabled, but inference.vllm.enable_return_routed_experts is False. Setting to True.",
                        stacklevel=2,
                    )
                self.inference.vllm.enable_return_routed_experts = True
            else:
                warnings.warn(
                    "Router replay is enabled, but inference is not configured. When manually starting the inference server, make sure to pass `--enable-return-routed-experts` to the vLLM server.",
                    stacklevel=2,
                )
        return self

    @model_validator(mode="after")
    def validate_llmd_no_routed_experts(self):
        """Reject routed-expert return with the llm-d router (breaks P/D, unverified for multi-node).

        Runs after ``auto_setup_router_replay`` so it also catches the
        ``trainer.enable_router_replay`` path, which sets the inference flag here
        (after InferenceConfig's own validators, which therefore miss it).
        """
        if self.inference is not None and self.inference.vllm.enable_return_routed_experts:
            router = self.inference.router
            if router is not None and router.type == "llm-d":
                raise ValueError(
                    "The llm-d router backend does not support routed-expert return "
                    "(inference.vllm.enable_return_routed_experts / trainer.enable_router_replay): it "
                    "breaks P/D and is unverified for multi-node. Use router type 'vllm-router' "
                    "for router-replay runs."
                )
        return self

    @model_validator(mode="after")
    def validate_multi_node_requires_router(self):
        if self.deployment.type == "multi_node" and self.inference is not None and self.inference.router is None:
            raise ValueError("Multi-node deployments require inference.router to front the per-rank engines.")
        return self

    @model_validator(mode="after")
    def validate_router_replay_without_kv_offload(self):
        if (
            self.trainer.enable_router_replay
            and self.inference is not None
            and self.inference.kv_cache_offload is not None
        ):
            raise ValueError(
                "Router replay with inference.kv_cache_offload is not supported. "
                "External KV cache hits do not carry routed-expert decisions."
            )
        return self

    @model_validator(mode="after")
    def validate_mooncake_offload_requires_slurm(self):
        if (
            self.slurm is None
            and self.inference is not None
            and self.inference.kv_cache_offload is not None
            and self.inference.kv_cache_offload.type == "mooncake"
        ):
            raise ValueError(
                "Mooncake KV offload requires SLURM — the per-node store is launched by the sbatch "
                "template. Use inference.kv_cache_offload.type='native' for local runs."
            )
        return self

    @model_validator(mode="after")
    def auto_setup_deployment(self):
        self.orchestrator.pad_to_multiple_of = self.trainer.model.cp
        if self.deployment.type == "single_node":  # single-node
            # set num_train_workers to the number of data replicas
            non_data_parallel_size = self.trainer.model.cp
            if self.deployment.num_train_gpus > 1:
                self.orchestrator.num_train_workers = self.deployment.num_train_gpus // non_data_parallel_size

            # fill up inference capacity with dp ranks
            if self.inference is not None:
                num_infer_gpus = self.deployment.num_infer_gpus
                if num_infer_gpus != self.inference.vllm.data_parallel_size * self.inference.vllm.tensor_parallel_size:
                    assert num_infer_gpus % self.inference.vllm.tensor_parallel_size == 0, (
                        "Number of inference GPUs must be divisible by the tensor parallel size"
                    )
                    self.inference.vllm.data_parallel_size = num_infer_gpus // self.inference.vllm.tensor_parallel_size
                # Ensure api_server_count matches DP so all workers are created.
                # Without this, in-memory weight transfer expects dp*tp workers
                # but only api_server_count*tp exist, causing a deadlock.
                dp = self.inference.vllm.data_parallel_size
                if self.inference.vllm.api_server_count < dp and not self.inference.vllm.enable_lora:
                    self.inference.vllm.api_server_count = dp

        elif self.deployment.type == "multi_node":  # multi-node
            self.orchestrator.num_train_workers = (
                self.deployment.num_train_nodes * self.deployment.gpus_per_node // self.trainer.model.cp
            )

            if self.deployment.nodes_per_fsdp_group is not None:
                if self.deployment.num_train_nodes % self.deployment.nodes_per_fsdp_group != 0:
                    raise ValueError(
                        f"deployment.num_train_nodes ({self.deployment.num_train_nodes}) must be divisible by "
                        f"deployment.nodes_per_fsdp_group ({self.deployment.nodes_per_fsdp_group})"
                    )
                self.trainer.model.dp_replicate = (
                    self.deployment.num_train_nodes // self.deployment.nodes_per_fsdp_group
                )

            if (
                self.inference is not None
                and self.inference.vllm.enable_expert_parallel
                and self.inference.deployment.type != "disaggregated"
            ):
                inference_tp = self.inference.vllm.tensor_parallel_size
                if self.deployment.gpus_per_node % inference_tp != 0:
                    raise ValueError(
                        "deployment.gpus_per_node must be divisible by inference.vllm.tensor_parallel_size "
                        "when inference.vllm.enable_expert_parallel is enabled in multi-node deployment."
                    )

                inferred_dp_local = self.deployment.gpus_per_node // inference_tp
                total_infer_gpus = self.deployment.infer_nodes_per_replica * self.deployment.gpus_per_node
                expected_global_world_size = self.inference.vllm.data_parallel_size * inference_tp
                if expected_global_world_size != total_infer_gpus:
                    raise ValueError(
                        "For multi-node expert parallel inference, inference.vllm.data_parallel_size * inference.vllm.tensor_parallel_size "
                        f"must match total inference GPUs ({total_infer_gpus}), got {expected_global_world_size}."
                    )

                if self.inference.vllm.data_parallel_size_local is None:
                    self.inference.vllm.data_parallel_size_local = inferred_dp_local
                elif self.inference.vllm.data_parallel_size_local != inferred_dp_local:
                    raise ValueError(
                        "inference.vllm.data_parallel_size_local must equal deployment.gpus_per_node / inference.vllm.tensor_parallel_size "
                        f"({inferred_dp_local}) when inference.vllm.enable_expert_parallel is enabled in multi-node deployment."
                    )

                if (
                    not self.inference.vllm.enable_lora
                    and self.inference.vllm.api_server_count == self.inference.vllm.data_parallel_size
                ):
                    self.inference.vllm.api_server_count = inferred_dp_local

            # Auto-infer DP and api_server_count for standard multi-node inference.
            # Without EP, vLLM only creates api_server_count * tp workers per node,
            # not gpus_per_node workers. If DP isn't set, the broadcast group expects
            # more workers than exist, deadlocking in-memory transfer initialization.
            if (
                self.inference is not None
                and not self.inference.vllm.enable_expert_parallel
                and self.inference.deployment.type != "disaggregated"
            ):
                dp_per_node = self.deployment.gpus_per_node // self.inference.vllm.tensor_parallel_size
                if self.inference.vllm.data_parallel_size == 1 and dp_per_node > 1:
                    self.inference.vllm.data_parallel_size = dp_per_node
                if self.inference.vllm.data_parallel_size_local is None and dp_per_node > 1:
                    self.inference.vllm.data_parallel_size_local = dp_per_node
                if self.inference.vllm.api_server_count == 1 and dp_per_node > 1:
                    self.inference.vllm.api_server_count = dp_per_node

            if self.weight_broadcast is not None and self.weight_broadcast.type in ("nccl", "nixl"):
                # Every allocated inference GPU is an in-memory transfer worker.
                # The external-LB launcher starts dp_per_node (= gpus_per_node / tp)
                # TP-sharded servers per node, i.e. gpus_per_node workers per node, so use
                # the GPU count directly. Deriving it from api_server_count double-counts:
                # api_server_count can resolve to the *global* DP size, making the node
                # factor count twice and wait for ranks that never connect. Matches
                # the disaggregated path below.
                total_infer_workers = self.deployment.total_infer_nodes * self.deployment.gpus_per_node
                assert self.trainer.weight_broadcast.type in ("nccl", "nixl")
                if self.trainer.weight_broadcast.type == "nccl":
                    self.trainer.weight_broadcast.host = "0.0.0.0"
                self.trainer.weight_broadcast.inference_world_size = total_infer_workers
                assert self.orchestrator.weight_broadcast.type in ("nccl", "nixl")
                self.orchestrator.weight_broadcast.inference_world_size = total_infer_workers

        return self

    @model_validator(mode="after")
    def auto_setup_disaggregated_inference(self):
        """Auto-setup for disaggregated P/D inference within a multi-node deployment."""
        if self.inference is None or self.inference.deployment.type != "disaggregated":
            return self
        if self.deployment.type != "multi_node":
            return self

        infer_deploy = self.inference.deployment
        expected_infer_nodes = infer_deploy.num_nodes
        if self.deployment.infer_nodes_per_replica != expected_infer_nodes:
            raise ValueError(
                f"deployment.num_infer_nodes ({self.deployment.num_infer_nodes}) must equal the derived "
                f"disaggregated inference nodes per replica ({expected_infer_nodes})."
            )

        total_infer_gpus = self.deployment.total_infer_nodes * self.deployment.gpus_per_node
        if "inference_metrics_roles" not in self.orchestrator.model_fields_set:
            # External-LB: one admin client per DP rank, so roles expand per rank
            # (stride = dp_local = gpus_per_node / tp). ADMIN_URLS lists all prefill
            # ranks, then all decode ranks, per replica — match that order.
            stride = self.deployment.gpus_per_node // self.inference.vllm.tensor_parallel_size
            role_order = ["prefill"] * (infer_deploy.num_prefill_nodes * stride) + ["decode"] * (
                infer_deploy.num_decode_nodes * stride
            )
            self.orchestrator.inference_metrics_roles = role_order * self.deployment.num_infer_replicas
        if self.weight_broadcast is not None and self.weight_broadcast.type in ("nccl", "nixl"):
            assert self.trainer.weight_broadcast.type in ("nccl", "nixl")
            self.trainer.weight_broadcast.inference_world_size = total_infer_gpus
            assert self.orchestrator.weight_broadcast.type in ("nccl", "nixl")
            self.orchestrator.weight_broadcast.inference_world_size = total_infer_gpus

        return self

    @model_validator(mode="after")
    def auto_setup_inference_client(self):
        """Auto-configure the orchestrator policy client from the inference server config.

        When no train env samples from the policy (e.g. sft_distill), set
        base_url. Policy-sourced algorithms rely on the ClientConfig default
        (``["http://localhost:8000/v1"]``), which already matches the
        auto-launched policy router at inference.server.port = 8000.
        """
        if self.inference is None:
            return self
        client = self.orchestrator.model.client
        if not self.orchestrator.any_policy_sourced and "base_url" not in client.model_fields_set:
            host = self.inference.server.host or "localhost"
            port = self.inference.server.port
            client.base_url = f"http://{host}:{port}/v1"
        if (
            self.deployment.type == "single_node"
            and self.inference.router is not None
            and "admin_base_url" not in client.model_fields_set
        ):
            # Admin ops (pause/update_weights/resume) must bypass the router and hit
            # the engine directly; multi-node runs get ADMIN_URLS from the sbatch.
            host = self.inference.server.host or "localhost"
            client.admin_base_url = [f"http://{host}:{self.inference.backend_port}/v1"]
        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        """Auto-setup the default single-node/multi-node SLURM template if no custom template is provided."""
        if self.slurm is not None and self.slurm.template_path is None:
            templates_dir = find_package_resource("templates")
            if templates_dir is not None:
                if self.deployment.type == "single_node":
                    self.slurm.template_path = templates_dir / "single_node_rl.sbatch.j2"
                else:
                    self.slurm.template_path = templates_dir / "multi_node_rl.sbatch.j2"
        return self

    ### Warnings

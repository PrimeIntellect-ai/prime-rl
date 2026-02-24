import warnings
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.inference import WeightBroadcastConfig as InferenceWeightBroadcastConfig
from prime_rl.configs.orchestrator import (
    CheckpointConfig as OrchestratorCheckpointConfig,
)
from prime_rl.configs.orchestrator import (
    FileSystemWeightBroadcastConfig as OrchestratorFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.orchestrator import (
    NCCLWeightBroadcastConfig as OrchestratorNCCLWeightBroadcastConfig,
)
from prime_rl.configs.orchestrator import (
    OrchestratorConfig,
)
from prime_rl.configs.shared import (
    SlurmConfig,
    WandbConfig,
    WandbWithExtrasConfig,
)
from prime_rl.configs.trainer import (
    BenchConfig,
    FakeDataLoaderConfig,
    TrainerConfig,
)
from prime_rl.configs.trainer import (
    CheckpointConfig as TrainerCheckpointConfig,
)
from prime_rl.configs.trainer import (
    FileSystemWeightBroadcastConfig as TrainerFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.trainer import (
    NCCLWeightBroadcastConfig as TrainerNCCLWeightBroadcastConfig,
)
from prime_rl.utils.pydantic_config import BaseSettings
from prime_rl.utils.validation import (
    validate_shared_ckpt_config,
    validate_shared_max_async_level,
    validate_shared_max_steps,
    validate_shared_model_name,
    validate_shared_output_dir,
    validate_shared_wandb_config,
    validate_shared_weight_broadcast,
)


class SharedLogConfig(BaseSettings):
    """Configures shared logging."""

    level: Annotated[str | None, Field(description="The log level to use.")] = "info"

    file: Annotated[bool | None, Field(description="Whether to log to a file.")] = True

    json_logging: Annotated[
        bool,
        Field(description="Emit JSON logs (newline-delimited) for log aggregation (Loki, Grafana, etc.)."),
    ] = False


class SharedWandbConfig(BaseSettings):
    """Configures shared W&B configs."""

    project: Annotated[str | None, Field(description="The W&B project to use.")] = "prime-rl"

    name: Annotated[str | None, Field(description="The W&B run name to use.")] = None

    offline: Annotated[bool | None, Field(description="Whether to run W&B in offline mode.")] = False


class SharedCheckpointConfig(BaseSettings):
    """Configures shared checkpoint configs."""

    interval: Annotated[int | None, Field(description="The interval at which to save checkpoints.")] = None

    resume_step: Annotated[
        int | None, Field(description="The step to resume from. If None, will not resume from a checkpoint.")
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


class SharedModelConfig(BaseSettings):
    """Configures shared model settings."""

    name: Annotated[
        str,
        Field(description="The name of the model to use."),
    ] = "Qwen/Qwen3-0.6B"


class SharedWeightBroadcastConfig(BaseSettings):
    """Configures shared weight broadcast settings."""

    type: Annotated[Literal["nccl", "filesystem"], Field(description="The type of weight broadcast to use.")] = (
        "filesystem"
    )

    port: Annotated[int, Field(description="The port to use for NCCL weight broadcast.")] = 29501
    timeout: Annotated[int, Field(description="The timeout in seconds for NCCL weight broadcast.")] = 1200


class BaseDeploymentConfig(BaseModel):
    """Configures a base deployment."""

    model_config = ConfigDict(extra="forbid")

    gpus_per_node: Annotated[int, Field(description="Number of GPUs per node.")] = 8


class SingleNodeDeploymentConfig(BaseDeploymentConfig):
    """Configures a single node deployment."""

    type: Literal["single_node"] = "single_node"

    num_train_gpus: Annotated[int, Field(description="Number of training GPUs")] = 1
    num_infer_gpus: Annotated[int, Field(description="Number of inference GPUs")] = 1
    num_teacher_gpus: Annotated[int | None, Field(description="Number of teacher inference GPUs")] = None

    @model_validator(mode="after")
    def validate_gpu_count(self):
        total = self.num_train_gpus + self.num_infer_gpus + (self.num_teacher_gpus or 0)
        if total > self.gpus_per_node:
            raise ValueError(
                f"Total GPU count ({total} = {self.num_train_gpus} train + {self.num_infer_gpus} infer"
                f" + {self.num_teacher_gpus or 0} teacher) exceeds gpus_per_node ({self.gpus_per_node})."
            )
        return self


class MultiNodeDeploymentConfig(BaseDeploymentConfig):
    """Configures a multi node deployment."""

    type: Literal["multi_node"] = "multi_node"

    num_train_nodes: Annotated[int, Field(description="Number of training nodes.")]
    num_infer_nodes: Annotated[int, Field(description="Number of inference nodes.")]
    num_teacher_nodes: Annotated[int | None, Field(description="Number of teacher inference nodes.")] = None

    nodes_per_fsdp_group: Annotated[
        int | None,
        Field(
            description="Number of training nodes per FSDP island. Auto-sets trainer.dp_replicate = num_train_nodes / nodes_per_fsdp_group."
        ),
    ] = None

    @model_validator(mode="after")
    def teacher_inference_not_supported(self):
        if self.num_teacher_nodes is not None:
            raise ValueError("Teacher inference is not yet supported in multi node deployment.")
        return self


DeploymentConfig: TypeAlias = Annotated[
    SingleNodeDeploymentConfig | MultiNodeDeploymentConfig, Field(discriminator="type")
]


class RLConfig(BaseSettings):
    """Configures an RL training run."""

    trainer: TrainerConfig
    orchestrator: OrchestratorConfig
    inference: Annotated[
        InferenceConfig | None,
        Field(
            description="The inference config. If None, the rl entrypoint will not start an inference server (useful for elastic inference pools or manually started servers)."
        ),
    ] = None

    teacher_inference: Annotated[
        InferenceConfig | None,
        Field(
            description="Teacher inference config. If None, will use the same config as inference or a default config. Only used when teacher GPUs or nodes are set."
        ),
    ] = None

    output_dir: Annotated[
        Path,
        Field(
            description="The directory to store the outputs. Should be set to a unique directory identifying the experiment."
        ),
    ] = Path("outputs")

    clean_output_dir: Annotated[
        bool,
        Field(
            description="If true, delete the output directory before starting training. Required to overwrite an output directory that contains checkpoints from a previous run when not resuming.",
        ),
    ] = False

    deployment: DeploymentConfig = SingleNodeDeploymentConfig()

    slurm: Annotated[SlurmConfig | None, Field(description="SLURM configuration. If None, will run locally.")] = None

    ### Shared configurations

    log: Annotated[
        SharedLogConfig,
        Field(
            description="Shared log configs. If None, will fallback to the log configs specified on submodule configs."
        ),
    ] = SharedLogConfig()

    ckpt: Annotated[
        SharedCheckpointConfig | None,
        Field(
            description="Shared checkpoint configs. If None, will fallback to the checkpoint configs specified on submodule configs."
        ),
    ] = None

    wandb: Annotated[
        SharedWandbConfig | None,
        Field(
            description="Shared W&B configs. If None, will fallback to the W&B configs specified on submodule configs."
        ),
    ] = None

    model: Annotated[
        SharedModelConfig | None,
        Field(
            description="Shared model configs. If None, will fallback to the model configs specified on submodule configs."
        ),
    ] = None

    max_steps: Annotated[
        int | None,
        Field(
            description="The maximum number of steps to train for. If None, will fallback to the max steps specified on submodule configs."
        ),
    ] = None

    max_model_len: Annotated[
        int | None,
        Field(
            description="The maximum model length to use. If None, will fallback to the max model length specified on submodule configs."
        ),
    ] = None

    seq_len: Annotated[
        int | None,
        Field(
            description="Shared sequence length. Propagates to trainer.model.seq_len and orchestrator.seq_len, "
            "but only for those not explicitly set in the config. "
            "Explicitly set per-component values always take precedence."
        ),
    ] = None

    max_async_level: Annotated[
        int | None,
        Field(
            description="The async level to use. If None, will fallback to the async level specified on submodule configs."
        ),
    ] = None

    weight_broadcast: Annotated[
        SharedWeightBroadcastConfig | None, Field(description="The weight broadcast config.")
    ] = None

    ### Local-only fields

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. Automatically sets the trainer and orchestrator to benchmark mode and, if present, suffixes the W&B project with `-bench`.",
        ),
    ] = False

    dump_config: Annotated[
        Path | None,
        Field(
            description="If set, dump resolved subconfigs (trainer, orchestrator, inference) to this directory and exit without starting any processes."
        ),
    ] = None

    ### Validate configs (e.g. raise for unsupported (combinations of) configs)

    @model_validator(mode="after")
    def validate_deployment(self):
        if self.deployment.type == "multi_node":
            if self.slurm is None:
                raise ValueError("Must use SLURM for multi-node deployment.")
            if not self.inference:
                raise ValueError("Must configure inference when using multi-node deployment.")
        return self

    # TODO: fix this
    @model_validator(mode="after")
    def validate_no_teacher_in_multinode(self):
        if self.deployment.type == "multi_node" and self.teacher_inference is not None:
            raise ValueError(
                "Teacher inference is not supported in multi-node deployment. "
                "The SLURM template only handles inference and training nodes."
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
    def validate_teacher_model(self):
        if (
            self.trainer.loss.type == "default" and self.trainer.loss.teacher_tau > 0
        ) and not self.orchestrator.teacher_model:
            raise ValueError(
                "teacher_model must be configured when teacher_tau > 0. "
                "Either set teacher_tau = 0, set deployment.num_teacher_gpus, or configure teacher_model manually."
            )
        return self

    ### Auto-setup and validate shared configs

    @model_validator(mode="after")
    def auto_setup_output_dir(self):
        """Auto-setup shared output directory for trainer and orchestrator."""
        self.trainer.output_dir = self.output_dir
        self.orchestrator.output_dir = self.output_dir / "run_default"

        validate_shared_output_dir(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_logs(self):
        """Auto-setup shared log config for trainer and orchestrator."""
        if self.log is not None:
            if self.log.level is not None:
                self.trainer.log.level = self.log.level
                self.orchestrator.log.level = self.log.level
            if self.log.file is not None:
                self.trainer.log.file = self.log.file
                self.orchestrator.log.file = self.log.file
            self.trainer.log.json_logging = self.log.json_logging
            self.orchestrator.log.json_logging = self.log.json_logging

        return self

    @model_validator(mode="after")
    def auto_setup_ckpt(self):
        """Auto-setup shared checkpoint config for trainer and orchestrator."""
        if self.ckpt is not None:
            # Create checkpoint configs if not specified
            if self.trainer.ckpt is None:
                self.trainer.ckpt = TrainerCheckpointConfig()
            if self.orchestrator.ckpt is None:
                self.orchestrator.ckpt = OrchestratorCheckpointConfig()

            # If specified, use the same ckpt interval
            if self.ckpt.interval is not None:
                self.trainer.ckpt.interval = self.ckpt.interval
                self.orchestrator.ckpt.interval = self.ckpt.interval

            # If resuming training, ensure orchestrator resume from the same step
            if self.ckpt.resume_step is not None:
                self.trainer.ckpt.resume_step = self.ckpt.resume_step
                self.orchestrator.ckpt.resume_step = self.ckpt.resume_step

            # If specified, propagate keep policy
            if self.ckpt.keep_last is not None:
                self.trainer.ckpt.keep_last = self.ckpt.keep_last
                self.orchestrator.ckpt.keep_last = self.ckpt.keep_last

            if self.ckpt.keep_interval is not None:
                self.trainer.ckpt.keep_interval = self.ckpt.keep_interval
                self.orchestrator.ckpt.keep_interval = self.ckpt.keep_interval

        validate_shared_ckpt_config(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_wandb(self):
        """Auto-setup shared W&B config for trainer and orchestrator."""
        if self.wandb is not None:
            if not self.trainer.wandb:
                self.trainer.wandb = WandbConfig()
            if not self.orchestrator.wandb:
                self.orchestrator.wandb = WandbWithExtrasConfig()

            if self.wandb.project:
                self.trainer.wandb.project = self.wandb.project
                self.orchestrator.wandb.project = self.wandb.project

            # If specified, automatically use shared W&B name for orchestrator and trainer with suffixes
            if self.wandb.name:
                self.trainer.wandb.name = f"{self.wandb.name}-trainer"
                self.orchestrator.wandb.name = f"{self.wandb.name}-orchestrator"

            # If specified, automatically use shared W&B offline mode for orchestrator and trainer
            if self.wandb.offline:
                self.trainer.wandb.offline = self.wandb.offline
                self.orchestrator.wandb.offline = self.wandb.offline

        validate_shared_wandb_config(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_model(self):
        """Auto-setup shared model config for trainer, orchestrator, and inference."""
        if self.model is not None:
            self.trainer.model.name = self.model.name
            self.orchestrator.model.name = self.model.name
            if self.inference is not None:
                self.inference.model.name = self.model.name
                self.inference.model.tool_call_parser = None
                self.inference.model.enable_auto_tool_choice = False
                self.inference.model.resolve_tool_call_parser()

        validate_shared_model_name(self.trainer, self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def auto_setup_max_steps(self):
        """Auto-setup shared max steps for trainer and orchestrator."""
        if self.max_steps is not None:
            self.trainer.max_steps = self.max_steps
            self.orchestrator.max_steps = self.max_steps

        validate_shared_max_steps(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_async_level(self):
        """Auto-setup shared async level for trainer and orchestrator."""
        if self.max_async_level is not None:
            self.trainer.max_async_level = self.max_async_level
            self.orchestrator.max_async_level = self.max_async_level

        validate_shared_max_async_level(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_seq_len(self):
        """Auto-setup shared seq_len for trainer and orchestrator.

        Only propagates to components that weren't explicitly set in the config.
        Uses model_fields_set to detect explicit assignment.
        """
        if self.seq_len is not None:
            if "seq_len" not in self.trainer.model.model_fields_set:
                self.trainer.model.seq_len = self.seq_len
            if "seq_len" not in self.orchestrator.model_fields_set:
                self.orchestrator.seq_len = self.seq_len

        if self.trainer.model.seq_len < self.orchestrator.seq_len:
            raise ValueError(
                f"Trainer model seq_len ({self.trainer.model.seq_len}) must be >= orchestrator seq_len ({self.orchestrator.seq_len}). "
                f"The trainer needs to be able to handle sequences at least as long as those produced by the orchestrator."
            )

        return self

    @model_validator(mode="after")
    def auto_setup_weight_broadcast(self):
        """Auto-setup shared weight broadcast config for trainer, orchestrator, and inference."""
        if self.weight_broadcast is not None:
            if self.weight_broadcast.type == "nccl":
                inference_world_size = self.inference.parallel.dp * self.inference.parallel.tp if self.inference else 1
                self.trainer.weight_broadcast = TrainerNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type,
                    inference_world_size=inference_world_size,
                    port=self.weight_broadcast.port,
                    timeout=self.weight_broadcast.timeout,
                )
                self.orchestrator.weight_broadcast = OrchestratorNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type,
                    port=self.weight_broadcast.port,
                    timeout=self.weight_broadcast.timeout,
                )
            elif self.weight_broadcast.type == "filesystem":
                self.trainer.weight_broadcast = TrainerFileSystemWeightBroadcastConfig()
                self.orchestrator.weight_broadcast = OrchestratorFileSystemWeightBroadcastConfig()
            if self.inference is not None:
                self.inference.weight_broadcast = InferenceWeightBroadcastConfig(type=self.weight_broadcast.type)

        validate_shared_weight_broadcast(self.trainer, self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.trainer.bench = BenchConfig()
            self.orchestrator.bench = True
            self.trainer.data.fake = FakeDataLoaderConfig(
                batch_size=self.orchestrator.batch_size,
            )

        trainer_bench_enabled = self.trainer.bench is not None
        if trainer_bench_enabled != self.orchestrator.bench:
            raise ValueError(
                f"Trainer benchmark mode ({self.trainer.bench}) and orchestrator benchmark mode "
                f"({self.orchestrator.bench}) must match. Use the top-level bench = true to set both."
            )

        return self

    @model_validator(mode="after")
    def auto_setup_lora(self):
        if self.trainer.model.lora is not None:
            if self.trainer.weight_broadcast.type == "nccl":
                raise ValueError("NCCL weight broadcast does not support LoRA yet.")

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
                self.inference.enable_lora = True
                self.inference.max_lora_rank = self.trainer.model.lora.rank
            else:
                warnings.warn(
                    "LoRA is enabled, but inference is not configured. When manually starting the inference server, "
                    "make sure to set --enable_lora and --max-lora-rank."
                )

        return self

    @model_validator(mode="after")
    def auto_setup_router_replay(self):
        if self.trainer.enable_router_replay:
            if self.inference is not None:
                if self.inference.enable_return_routed_experts is False:
                    warnings.warn(
                        "Router replay is enabled, but inference.enable_return_routed_experts is False. Setting to True."
                    )
                self.inference.enable_return_routed_experts = True
            else:
                warnings.warn(
                    "Router replay is enabled, but inference is not configured. When manually starting the inference server, make sure to pass `--enable-return-routed-experts` to the vLLM server."
                )
        return self

    @model_validator(mode="after")
    def auto_setup_deployment(self):
        if self.deployment.type == "single_node":  # single-node
            # set num_train_workers to the number of data replicas
            non_data_parallel_size = self.trainer.model.cp * self.trainer.model.tp
            if self.deployment.num_train_gpus > 1:
                self.orchestrator.num_train_workers = self.deployment.num_train_gpus // non_data_parallel_size

            # fill up inference capacity with dp ranks
            if self.inference is not None:
                num_infer_gpus = self.deployment.num_infer_gpus
                if num_infer_gpus != self.inference.parallel.dp * self.inference.parallel.tp:
                    assert num_infer_gpus % self.inference.parallel.tp == 0, (
                        "Number of inference GPUs must be divisible by the tensor parallel size"
                    )
                    self.inference.parallel.dp = num_infer_gpus // self.inference.parallel.tp

        elif self.deployment.type == "multi_node":  # multi-node
            self.orchestrator.num_train_workers = self.deployment.num_train_nodes * self.deployment.gpus_per_node

            if self.deployment.nodes_per_fsdp_group is not None:
                if self.deployment.num_train_nodes % self.deployment.nodes_per_fsdp_group != 0:
                    raise ValueError(
                        f"deployment.num_train_nodes ({self.deployment.num_train_nodes}) must be divisible by "
                        f"deployment.nodes_per_fsdp_group ({self.deployment.nodes_per_fsdp_group})"
                    )
                self.trainer.model.dp_replicate = (
                    self.deployment.num_train_nodes // self.deployment.nodes_per_fsdp_group
                )

            if self.weight_broadcast is not None and self.weight_broadcast.type == "nccl":
                assert self.trainer.weight_broadcast.type == "nccl"
                self.trainer.weight_broadcast.host = "0.0.0.0"
                self.trainer.weight_broadcast.inference_world_size = (
                    self.deployment.gpus_per_node * self.deployment.num_infer_nodes
                )

        return self

    @model_validator(mode="after")
    def auto_setup_teacher_inference(self):
        """Auto-configure teacher inference server and orchestrator teacher_model client."""
        if self.deployment.type != "single_node":
            return self
        if self.deployment.num_teacher_gpus is None or self.deployment.num_teacher_gpus == 0:
            return self

        import copy

        from prime_rl.configs.orchestrator import TeacherModelConfig

        if self.teacher_inference is None:
            if self.inference is None:
                self.teacher_inference = InferenceConfig()
            else:
                self.teacher_inference = copy.deepcopy(self.inference)
            self.teacher_inference.server.port = (self.inference.server.port if self.inference else 8000) + 1
        elif self.inference is not None and self.teacher_inference.server.port == self.inference.server.port:
            raise ValueError(
                f"teacher_inference.server.port ({self.teacher_inference.server.port}) conflicts with "
                f"inference.server.port ({self.inference.server.port}). "
                "Either use different ports or let teacher_inference be auto-configured."
            )

        tp = self.teacher_inference.parallel.tp
        num_teacher_gpus = self.deployment.num_teacher_gpus
        if num_teacher_gpus != self.teacher_inference.parallel.dp * tp:
            assert num_teacher_gpus % tp == 0, "Number of teacher GPUs must be divisible by tensor parallel size"
            assert num_teacher_gpus > 0, "num_teacher_gpus cannot be zero"
            self.teacher_inference.parallel.dp = num_teacher_gpus // tp

        if self.orchestrator.teacher_model is None:
            self.orchestrator.teacher_model = TeacherModelConfig()
        host = self.teacher_inference.server.host or "localhost"
        port = self.teacher_inference.server.port
        self.orchestrator.teacher_model.client.base_url = [f"http://{host}:{port}/v1"]
        self.orchestrator.teacher_model.model.name = self.teacher_inference.model.name

        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        """Auto-setup the default single-node/multi-node SLURM template if no custom template is provided."""
        if self.slurm is not None and self.slurm.template_path is None:
            import prime_rl

            templates_dir = Path(prime_rl.__file__).parent / "templates"
            if self.deployment.type == "single_node":
                self.slurm.template_path = templates_dir / "single_node_rl.sbatch.j2"
            else:
                self.slurm.template_path = templates_dir / "multi_node_rl.sbatch.j2"
        return self

    ### Warnings

    @model_validator(mode="after")
    def warn_wandb_resume_id_missing(self):
        if self.trainer.ckpt is not None and self.trainer.ckpt.resume_step is not None:
            if self.trainer.wandb and not self.trainer.wandb.id:
                warnings.warn(
                    "W&B run ID is not set for trainer even though resuming training. The current run will be created as a new run."
                )
        if self.orchestrator.ckpt is not None and self.orchestrator.ckpt.resume_step is not None:
            if self.orchestrator.wandb and not self.orchestrator.wandb.id:
                warnings.warn(
                    "W&B run ID is not set for orchestrator even though resuming training. The current run will be created as a new run."
                )
        return self

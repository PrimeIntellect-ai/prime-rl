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
    VLMConfig,
    WandbConfig,
    WandbWithExtrasConfig,
)
from prime_rl.configs.trainer import (
    BenchConfig,
    FakeDataLoaderConfig,
    TokenizerConfig,
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
from prime_rl.utils.config import BaseConfig, find_package_resource
from prime_rl.utils.validation import (
    validate_shared_ckpt_config,
    validate_shared_max_async_level,
    validate_shared_max_steps,
    validate_shared_model_name,
    validate_shared_output_dir,
    validate_shared_tokenizer,
    validate_shared_wandb_config,
    validate_shared_weight_broadcast,
)


class RLExperimentalConfig(BaseConfig):
    """Experimental features for RL training."""


class SharedLogConfig(BaseConfig):
    """Configures shared logging."""

    level: Annotated[
        str | None,
        Field(
            description="The log level to use. When unset, the trainer and orchestrator log levels are used as-is (which themselves default to the PRIME_LOG_LEVEL env var if set, else 'info').",
        ),
    ] = None

    json_logging: Annotated[
        bool,
        Field(description="Emit JSON logs (newline-delimited) for log aggregation (Loki, Grafana, etc.)."),
    ] = False


class SharedWandbConfig(BaseConfig):
    """Configures shared W&B configs."""

    project: Annotated[str | None, Field(description="The W&B project to use.")] = "prime-rl"

    entity: Annotated[str | None, Field(description="The W&B entity to use.")] = None

    name: Annotated[str | None, Field(description="The W&B run name to use.")] = None

    group: Annotated[str | None, Field(description="The W&B group to use.")] = None

    tags: Annotated[list[str] | None, Field(description="The W&B tags to attach to the run.")] = None

    offline: Annotated[bool | None, Field(description="Whether to run W&B in offline mode.")] = False

    shared: Annotated[
        bool,
        Field(
            description="Use shared W&B mode to log trainer and orchestrator metrics to a single run. "
            "Requires wandb SDK >= 0.19.9. Incompatible with offline mode.",
        ),
    ] = True

    @model_validator(mode="after")
    def validate_shared_not_offline(self):
        if self.shared and self.offline:
            raise ValueError("W&B shared mode requires server connectivity and is incompatible with offline mode")
        return self


class SharedCheckpointConfig(BaseConfig):
    """Configures shared checkpoint configs."""

    output_dir: Annotated[
        Path | None,
        Field(
            description="Override directory for checkpoints and weights. When set, checkpoints and weight snapshots are written here instead of under the trainer output_dir.",
        ),
    ] = None

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


class SharedModelConfig(BaseConfig):
    """Configures shared model settings."""

    name: Annotated[
        str,
        Field(description="The name of the model to use."),
    ] = "Qwen/Qwen3-0.6B"

    vlm: Annotated[
        "VLMConfig | None",
        Field(description="VLM configuration. Set to enable vision-language model support."),
    ] = None


class SharedWeightBroadcastConfig(BaseConfig):
    """Configures shared weight broadcast settings."""

    type: Annotated[Literal["nccl", "filesystem"], Field(description="The type of weight broadcast to use.")] = (
        "filesystem"
    )

    port: Annotated[int, Field(description="The port to use for NCCL weight broadcast.")] = 29501
    timeout: Annotated[int, Field(description="The timeout in seconds for NCCL weight broadcast.")] = 1200
    quantize_in_weight_transfer: Annotated[
        bool,
        Field(
            description=(
                "Use kernel-format FP8 quantized NCCL transfer for weight updates. "
                "When disabled, uses default HF checkpoint-format transfer."
            ),
        ),
    ] = False


class BaseDeploymentConfig(BaseModel):
    """Configures a base deployment."""

    model_config = ConfigDict(extra="forbid")

    gpus_per_node: Annotated[int, Field(description="Number of GPUs per node.")] = 8


class SingleNodeDeploymentConfig(BaseDeploymentConfig):
    """Configures a single node deployment."""

    type: Literal["single_node"] = "single_node"

    num_train_gpus: Annotated[int, Field(description="Number of training GPUs")] = 1
    num_infer_gpus: Annotated[
        int,
        Field(
            description=(
                "Number of GPUs allocated to the inference pool. When there is a single [[inference]] "
                "entry this sizes its parallel.dp directly; with multiple entries each entry's GPU "
                "count comes from its own parallel.dp * parallel.tp."
            ),
        ),
    ] = 1

    @model_validator(mode="after")
    def validate_gpu_count(self):
        total = self.num_train_gpus + self.num_infer_gpus
        if total > self.gpus_per_node:
            raise ValueError(
                f"num_train_gpus ({self.num_train_gpus}) + num_infer_gpus ({self.num_infer_gpus}) "
                f"exceeds gpus_per_node ({self.gpus_per_node}). Either reduce the inference pool, "
                f"raise gpus_per_node, or split deployments across nodes."
            )
        return self


class MultiNodeDeploymentConfig(BaseDeploymentConfig):
    """Configures a multi node deployment."""

    type: Literal["multi_node"] = "multi_node"

    num_train_nodes: Annotated[int, Field(description="Number of training nodes.")]
    num_infer_nodes: Annotated[
        int,
        Field(
            ge=0,
            description="Number of inference nodes per replica. Set to 0 to skip inference and orchestrator (requires fake data).",
        ),
    ]
    num_infer_replicas: Annotated[
        int,
        Field(
            ge=1,
            description="Number of independent inference replicas. Total inference nodes = num_infer_nodes * num_infer_replicas.",
        ),
    ] = 1

    nodes_per_fsdp_group: Annotated[
        int | None,
        Field(
            description="Number of training nodes per FSDP island. Auto-sets trainer.dp_replicate = num_train_nodes / nodes_per_fsdp_group."
        ),
    ] = None

    @property
    def total_infer_nodes(self) -> int:
        return self.num_infer_nodes * self.num_infer_replicas


DeploymentConfig: TypeAlias = Annotated[
    SingleNodeDeploymentConfig | MultiNodeDeploymentConfig, Field(discriminator="type")
]


class RLConfig(BaseConfig):
    """Configures an RL training run."""

    trainer: TrainerConfig
    orchestrator: OrchestratorConfig
    inference: Annotated[
        list[InferenceConfig],
        Field(
            description=(
                "Inference deployments to launch alongside this RL run, written as repeated "
                "[[inference]] TOML blocks. Each entry is tagged (`tag` field, defaults to "
                "`student`); the launcher brings up one vLLM subprocess per entry, and the "
                "orchestrator routes student/teacher requests by tag. An empty list means no "
                "inference server is launched here (useful for elastic pools or manually "
                "managed servers). For back-compat, a single `[inference]` block is auto-wrapped "
                'into a list with tag="student".'
            ),
        ),
    ] = []

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

    tokenizer: Annotated[
        TokenizerConfig | None,
        Field(
            description="Shared tokenizer config. Propagated to trainer, orchestrator, and inference. "
            "If None, each component uses its own tokenizer config (defaulting to model name).",
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

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. Automatically sets the trainer and orchestrator to benchmark mode and, if present, suffixes the W&B project with `-bench`.",
        ),
    ] = False

    deployment: DeploymentConfig = SingleNodeDeploymentConfig()

    slurm: Annotated[SlurmConfig | None, Field(description="SLURM configuration. If None, will run locally.")] = None

    dry_run: Annotated[bool, Field(description="Only validate and dump resolved configs and exit early.")] = False

    experimental: Annotated[
        RLExperimentalConfig,
        Field(description="Experimental features for RL training."),
    ] = RLExperimentalConfig()

    ### Tagged inference helpers

    def get_inference(self, tag: str) -> InferenceConfig | None:
        for entry in self.inference:
            if entry.tag == tag:
                return entry
        return None

    @property
    def student_inference(self) -> InferenceConfig | None:
        return self.get_inference("student")

    @property
    def teacher_inference(self) -> InferenceConfig | None:
        return self.get_inference("teacher")

    ### Validate configs (e.g. raise for unsupported (combinations of) configs)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_inference(cls, data):
        """Translate the pre-#2554 single-deployment schema into the tagged
        `[[inference]]` list:

        1. `[inference]` (dict) → `[{tag="student", ...}]`.
        2. `[teacher_inference]` (dict) → second list entry with `tag="teacher"`,
           sized from `deployment.num_teacher_gpus` (mapped onto `parallel.dp`).
        3. `deployment.num_teacher_gpus` set without a `[teacher_inference]`
           block → synthesizes a teacher entry whose `parallel.dp` matches.

        Once migrated, `num_teacher_gpus` is dropped from `deployment` (the
        field was removed from the schema in this PR). GPU placement is then
        sequential across inference entries — see the RL launcher.
        """
        if not isinstance(data, dict):
            return data

        inference = data.get("inference")
        teacher_inference = data.pop("teacher_inference", None)
        deployment = data.get("deployment") if isinstance(data.get("deployment"), dict) else None
        num_teacher_gpus = deployment.pop("num_teacher_gpus", None) if deployment is not None else None

        if isinstance(inference, dict):
            entry = {**inference}
            entry.setdefault("tag", "student")
            inference_list = [entry]
            data["inference"] = inference_list
        elif isinstance(inference, list):
            inference_list = inference
        elif inference is None:
            if teacher_inference is None and not num_teacher_gpus:
                return data
            inference_list = []
            data["inference"] = inference_list
        else:
            return data

        if isinstance(teacher_inference, dict):
            entry = {**teacher_inference}
            entry.setdefault("tag", "teacher")
            if num_teacher_gpus:
                parallel = entry.setdefault("parallel", {})
                parallel.setdefault("dp", num_teacher_gpus)
            inference_list.append(entry)
        elif num_teacher_gpus:
            inference_list.append({"tag": "teacher", "parallel": {"dp": num_teacher_gpus}})

        return data

    @model_validator(mode="before")
    @classmethod
    def _stub_orchestrator_teacher_for_auto_setup(cls, data):
        """When the inference list contains a `teacher` deployment and the user
        didn't write an `[orchestrator.teacher]` block, inject an empty one so
        `OrchestratorConfig.validate_training_mode` (which fires during nested
        validation) doesn't reject `training_mode = "opd"` for "missing teacher".
        `auto_setup_inference_clients` (after) then fills in client.base_url and
        model.name from the auto-launched teacher inference server."""
        if not isinstance(data, dict):
            return data
        inference = data.get("inference")
        if not isinstance(inference, list):
            return data
        if not any(isinstance(i, dict) and i.get("tag") == "teacher" for i in inference):
            return data
        orch = data.setdefault("orchestrator", {})
        if isinstance(orch, dict) and "teacher" not in orch and "teacher_model" not in orch:
            orch["teacher"] = {}
        return data

    @model_validator(mode="after")
    def validate_deployment(self):
        if self.deployment.type == "multi_node":
            if self.slurm is None:
                raise ValueError("Must use SLURM for multi-node deployment.")
            if self.deployment.num_infer_nodes > 0 and not self.inference:
                raise ValueError("Must configure inference when using multi-node deployment with inference nodes.")
            if self.deployment.num_infer_nodes == 0 and self.inference:
                raise ValueError(
                    "Cannot configure inference with num_infer_nodes = 0. "
                    "Either set num_infer_nodes > 0 or remove all [[inference]] blocks."
                )
            if self.deployment.num_infer_nodes == 0 and not self.trainer.data.fake and not self.bench:
                raise ValueError(
                    "Must use fake data (trainer.data.fake or bench = true) when num_infer_nodes = 0, "
                    "since no orchestrator or inference server will be running."
                )
        return self

    @model_validator(mode="after")
    def validate_inference_tags(self):
        tags = [i.tag for i in self.inference]
        if any(not t for t in tags):
            raise ValueError("Every [[inference]] entry must set a non-empty `tag`.")
        if len(tags) != len(set(tags)):
            raise ValueError(f"Duplicate [[inference]] tags: {tags}. Each `tag` must be unique within an RL run.")
        if self.inference and "student" not in tags:
            raise ValueError(
                'The [[inference]] list must include an entry with `tag = "student"` (the weight-broadcast target). '
                f"Got tags: {tags}."
            )
        return self

    @model_validator(mode="after")
    def validate_no_teacher_in_multinode(self):
        if self.deployment.type == "multi_node" and self.teacher_inference is not None:
            raise ValueError(
                "Teacher inference is not yet supported in multi-node deployment. "
                "The SLURM template only handles a single inference deployment per run."
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
        if self.weight_broadcast is None or not self.weight_broadcast.quantize_in_weight_transfer:
            return self

        if self.weight_broadcast.type != "nccl":
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires weight_broadcast.type = 'nccl'.")

        if not self.inference:
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires an inference config.")

        if self.trainer.model.impl != "custom":
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires trainer.model.impl = 'custom'.")

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

            # If specified, override checkpoint output directory
            if self.ckpt.output_dir is not None:
                self.trainer.ckpt.output_dir = self.ckpt.output_dir

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

            if self.wandb.entity:
                self.trainer.wandb.entity = self.wandb.entity
                self.orchestrator.wandb.entity = self.wandb.entity

            if self.wandb.shared:
                if self.wandb.name:
                    self.trainer.wandb.name = self.wandb.name
                    self.orchestrator.wandb.name = self.wandb.name
            else:
                if self.wandb.name:
                    self.trainer.wandb.name = f"{self.wandb.name}-trainer"
                    self.orchestrator.wandb.name = f"{self.wandb.name}-orchestrator"

            if self.wandb.group:
                self.trainer.wandb.group = self.wandb.group
                self.orchestrator.wandb.group = self.wandb.group

            if self.wandb.tags:
                self.trainer.wandb.tags = self.wandb.tags.copy()
                self.orchestrator.wandb.tags = self.wandb.tags.copy()

            if self.wandb.offline:
                self.trainer.wandb.offline = self.wandb.offline
                self.orchestrator.wandb.offline = self.wandb.offline

        validate_shared_wandb_config(self.trainer, self.orchestrator)

        if self.orchestrator.prime_monitor is not None and self.orchestrator.prime_monitor.run_name is None:
            if self.wandb and self.wandb.name:
                self.orchestrator.prime_monitor.run_name = self.wandb.name

        return self

    @model_validator(mode="after")
    def auto_setup_model(self):
        """Auto-setup shared model config for trainer, orchestrator, and inference."""
        if self.model is not None:
            self.trainer.model.name = self.model.name
            student_inf = self.student_inference
            if student_inf is not None:
                inference_model_explicitly_set = "name" in student_inf.model.model_fields_set
                if not inference_model_explicitly_set:
                    student_inf.model.name = self.model.name
                self.orchestrator.student.model.name = student_inf.model.name
            else:
                self.orchestrator.student.model.name = self.model.name

            if self.model.vlm is not None:
                self.trainer.model.vlm = self.model.vlm
                self.orchestrator.student.model.vlm = self.model.vlm
                if student_inf is not None:
                    student_inf.model.vlm = self.model.vlm

        validate_shared_model_name(self.trainer, self.orchestrator, self.student_inference)

        return self

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        """Auto-setup shared tokenizer config for trainer, orchestrator, and inference."""
        if self.tokenizer is not None:
            # Shared tokenizer config: propagate to all components, then fill
            # in name/trust_remote_code from model config where still unset.
            self.trainer.tokenizer = self.tokenizer.model_copy()
            self.orchestrator.tokenizer = self.tokenizer.model_copy()
            if self.trainer.tokenizer.name is None:
                self.trainer.tokenizer.name = self.trainer.model.name
            if self.trainer.tokenizer.trust_remote_code is None:
                self.trainer.tokenizer.trust_remote_code = self.trainer.model.trust_remote_code
            if self.orchestrator.tokenizer.name is None:
                self.orchestrator.tokenizer.name = self.orchestrator.student.model.name
            if self.orchestrator.tokenizer.trust_remote_code is None:
                self.orchestrator.tokenizer.trust_remote_code = self.orchestrator.student.model.trust_remote_code
        else:
            # No shared tokenizer: re-derive from (now-correct) model names,
            # since auto_setup_tokenizer on sub-configs already ran with defaults.
            self.trainer.tokenizer.name = self.trainer.model.name
            self.trainer.tokenizer.trust_remote_code = self.trainer.model.trust_remote_code
            self.orchestrator.tokenizer.name = self.orchestrator.student.model.name
            self.orchestrator.tokenizer.trust_remote_code = self.orchestrator.student.model.trust_remote_code

        # Propagate chat_template to the student inference deployment (vLLM
        # --chat-template). Other tags (e.g. teacher) keep their own template,
        # which can differ when the teacher is a distinct model family.
        chat_template = self.trainer.tokenizer.chat_template
        if chat_template is not None:
            student_inf = self.student_inference
            if student_inf is not None and student_inf.model.chat_template is None:
                student_inf.model.chat_template = chat_template

        validate_shared_tokenizer(self.trainer, self.orchestrator, self.student_inference)

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
        """Auto-setup shared weight broadcast config for trainer, orchestrator, and inference.

        NCCL broadcast targets the `student` deployment only (it's the weight-sync
        destination). Other tags (e.g. teacher) keep their own weight_broadcast
        config — typically `filesystem`/no-op since they aren't updated mid-run.
        """
        if self.weight_broadcast is not None:
            student_inf = self.student_inference
            if self.weight_broadcast.type == "nccl":
                inference_world_size = (
                    student_inf.parallel.dp * student_inf.parallel.tp if student_inf is not None else 1
                )
                self.trainer.weight_broadcast = TrainerNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type,
                    inference_world_size=inference_world_size,
                    port=self.weight_broadcast.port,
                    timeout=self.weight_broadcast.timeout,
                    quantize_in_weight_transfer=self.weight_broadcast.quantize_in_weight_transfer,
                )
                self.orchestrator.weight_broadcast = OrchestratorNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type,
                    port=self.weight_broadcast.port,
                    timeout=self.weight_broadcast.timeout,
                    inference_world_size=inference_world_size,
                    quantize_in_weight_transfer=self.weight_broadcast.quantize_in_weight_transfer,
                )
            elif self.weight_broadcast.type == "filesystem":
                self.trainer.weight_broadcast = TrainerFileSystemWeightBroadcastConfig()
                self.orchestrator.weight_broadcast = OrchestratorFileSystemWeightBroadcastConfig()
            if student_inf is not None:
                student_inf.weight_broadcast = InferenceWeightBroadcastConfig(type=self.weight_broadcast.type)

        validate_shared_weight_broadcast(self.trainer, self.orchestrator, self.student_inference)

        return self

    @model_validator(mode="after")
    def validate_eplb_requires_quantized_weight_transfer(self):
        student_inf = self.student_inference
        if student_inf is None or not student_inf.enable_eplb:
            return self

        # TODO(matej): check if weight reloading works itself before supporting EPLB without quantized transfer.
        trainer_weight_broadcast = self.trainer.weight_broadcast
        if trainer_weight_broadcast.type != "nccl" or not trainer_weight_broadcast.quantize_in_weight_transfer:
            raise ValueError(
                "inference.enable_eplb requires weight_broadcast.type = 'nccl' and "
                "weight_broadcast.quantize_in_weight_transfer = true."
            )

        return self

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.trainer.bench = BenchConfig()
            self.orchestrator.bench = True
            self.trainer.data.fake = FakeDataLoaderConfig(
                batch_size=self.orchestrator.batch_size or 32,
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

            if self.orchestrator.student.model.lora is None:
                from prime_rl.configs.orchestrator import LoRAConfig

                self.orchestrator.student.model.lora = LoRAConfig()

            if (
                self.orchestrator.student.model.lora.rank is not None
                and self.orchestrator.student.model.lora.rank != self.trainer.model.lora.rank
            ):
                raise ValueError(
                    f"orchestrator.student.model.lora.rank ({self.orchestrator.student.model.lora.rank}) conflicts with "
                    f"trainer.model.lora.rank ({self.trainer.model.lora.rank}). "
                    f"Remove orchestrator.student.model.lora.rank to inherit from trainer, or update trainer.model.lora.rank to match."
                )

            if (
                self.orchestrator.student.model.lora.alpha is not None
                and self.orchestrator.student.model.lora.alpha != self.trainer.model.lora.alpha
            ):
                raise ValueError(
                    f"orchestrator.student.model.lora.alpha ({self.orchestrator.student.model.lora.alpha}) conflicts with "
                    f"trainer.model.lora.alpha ({self.trainer.model.lora.alpha}). "
                    f"Remove orchestrator.student.model.lora.alpha to inherit from trainer, or update trainer.model.lora.alpha to match."
                )

            if self.orchestrator.student.model.lora.rank is None:
                self.orchestrator.student.model.lora.rank = self.trainer.model.lora.rank

            if self.orchestrator.student.model.lora.alpha is None:
                self.orchestrator.student.model.lora.alpha = self.trainer.model.lora.alpha

            if self.orchestrator.student.model.lora.name is None:
                self.orchestrator.student.model.lora.name = (
                    f"r{self.orchestrator.student.model.lora.rank}-a{self.orchestrator.student.model.lora.alpha}"
                )

            student_inf = self.student_inference
            if student_inf is not None:
                student_inf.enable_lora = True
                student_inf.max_lora_rank = self.trainer.model.lora.rank
            else:
                warnings.warn(
                    "LoRA is enabled, but inference is not configured. When manually starting the inference server, "
                    "make sure to set --enable_lora and --max-lora-rank.",
                    stacklevel=2,
                )

        return self

    @model_validator(mode="after")
    def auto_setup_session_headers(self):
        """Ensure X-Session-ID header is always set for sticky DP-aware routing at the inference router."""
        self.orchestrator.student.client.extra_headers_from_state.setdefault("X-Session-ID", "example_id")
        return self

    @model_validator(mode="after")
    def auto_setup_router_replay(self):
        if self.trainer.enable_router_replay:
            student_inf = self.student_inference
            if student_inf is not None:
                if student_inf.enable_return_routed_experts is False:
                    warnings.warn(
                        "Router replay is enabled, but inference.enable_return_routed_experts is False. Setting to True.",
                        stacklevel=2,
                    )
                student_inf.enable_return_routed_experts = True
            else:
                warnings.warn(
                    "Router replay is enabled, but inference is not configured. When manually starting the inference server, make sure to pass `--enable-return-routed-experts` to the vLLM server.",
                    stacklevel=2,
                )
        return self

    @model_validator(mode="after")
    def auto_setup_deployment(self):
        # Multi-node and the disaggregated/expert-parallel knobs below only apply
        # to the student deployment; other tags are gated out by
        # validate_no_teacher_in_multinode.
        student_inf = self.student_inference
        if self.deployment.type == "single_node":  # single-node
            # set num_train_workers to the number of data replicas
            non_data_parallel_size = self.trainer.model.cp
            if self.deployment.num_train_gpus > 1:
                self.orchestrator.num_train_workers = self.deployment.num_train_gpus // non_data_parallel_size

            # Single-deployment legacy: num_infer_gpus sizes parallel.dp directly
            # so existing `[inference]` configs keep working without setting dp.
            # With multiple deployments each entry's parallel.dp/tp is authoritative.
            if len(self.inference) == 1:
                entry = self.inference[0]
                num_infer_gpus = self.deployment.num_infer_gpus
                if num_infer_gpus != entry.parallel.dp * entry.parallel.tp:
                    assert num_infer_gpus % entry.parallel.tp == 0, (
                        f"Number of inference GPUs ({num_infer_gpus}) must be divisible by "
                        f"the tensor parallel size ({entry.parallel.tp})."
                    )
                    entry.parallel.dp = num_infer_gpus // entry.parallel.tp
            for entry in self.inference:
                dp = entry.parallel.dp
                if entry.api_server_count < dp and not entry.enable_lora:
                    entry.api_server_count = dp

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

            if (
                student_inf is not None
                and student_inf.enable_expert_parallel
                and student_inf.deployment.type != "disaggregated"
            ):
                inference_tp = student_inf.parallel.tp
                if self.deployment.gpus_per_node % inference_tp != 0:
                    raise ValueError(
                        "deployment.gpus_per_node must be divisible by inference.parallel.tp "
                        "when inference.enable_expert_parallel is enabled in multi-node deployment."
                    )

                inferred_dp_local = self.deployment.gpus_per_node // inference_tp
                total_infer_gpus = self.deployment.num_infer_nodes * self.deployment.gpus_per_node
                expected_global_world_size = student_inf.parallel.dp * inference_tp
                if expected_global_world_size != total_infer_gpus:
                    raise ValueError(
                        "For multi-node expert parallel inference, inference.parallel.dp * inference.parallel.tp "
                        f"must match total inference GPUs ({total_infer_gpus}), got {expected_global_world_size}."
                    )

                if student_inf.data_parallel_size_local is None:
                    student_inf.data_parallel_size_local = inferred_dp_local
                elif student_inf.data_parallel_size_local != inferred_dp_local:
                    raise ValueError(
                        "inference.data_parallel_size_local must equal deployment.gpus_per_node / inference.parallel.tp "
                        f"({inferred_dp_local}) when inference.enable_expert_parallel is enabled in multi-node deployment."
                    )

                if not student_inf.enable_lora and student_inf.api_server_count == student_inf.parallel.dp:
                    student_inf.api_server_count = inferred_dp_local

            # Auto-infer DP and api_server_count for standard multi-node inference.
            # Without EP, vLLM only creates api_server_count * tp workers per node,
            # not gpus_per_node workers. If DP isn't set, the broadcast group expects
            # more workers than exist, deadlocking NCCL init.
            if (
                student_inf is not None
                and not student_inf.enable_expert_parallel
                and student_inf.deployment.type != "disaggregated"
            ):
                dp_per_node = self.deployment.gpus_per_node // student_inf.parallel.tp
                if student_inf.parallel.dp == 1 and dp_per_node > 1:
                    student_inf.parallel.dp = dp_per_node
                if student_inf.data_parallel_size_local is None and dp_per_node > 1:
                    student_inf.data_parallel_size_local = dp_per_node
                if student_inf.api_server_count == 1 and dp_per_node > 1:
                    student_inf.api_server_count = dp_per_node

            if self.weight_broadcast is not None and self.weight_broadcast.type == "nccl":
                # Compute inference_world_size from actual worker count per server:
                # each api_server runs tp workers that participate in collective_rpc.
                api_server_count = student_inf.api_server_count if student_inf else 1
                tp = student_inf.parallel.tp if student_inf else 1
                total_infer_workers = self.deployment.total_infer_nodes * api_server_count * tp
                assert self.trainer.weight_broadcast.type == "nccl"
                self.trainer.weight_broadcast.host = "0.0.0.0"
                self.trainer.weight_broadcast.inference_world_size = total_infer_workers
                assert self.orchestrator.weight_broadcast.type == "nccl"
                self.orchestrator.weight_broadcast.inference_world_size = total_infer_workers

        return self

    @model_validator(mode="after")
    def auto_setup_disaggregated_inference(self):
        """Auto-setup for disaggregated P/D inference within a multi-node deployment.

        Disaggregated P/D only applies to the student deployment today; other
        tags are blocked by ``validate_no_teacher_in_multinode``.
        """
        student_inf = self.student_inference
        if student_inf is None or student_inf.deployment.type != "disaggregated":
            return self
        if self.deployment.type != "multi_node":
            return self

        infer_deploy = student_inf.deployment
        expected_infer_nodes = infer_deploy.num_prefill_nodes + infer_deploy.num_decode_nodes
        if self.deployment.num_infer_nodes != expected_infer_nodes:
            raise ValueError(
                f"deployment.num_infer_nodes ({self.deployment.num_infer_nodes}) must equal "
                f"inference.deployment.num_prefill_nodes ({infer_deploy.num_prefill_nodes}) + "
                f"inference.deployment.num_decode_nodes ({infer_deploy.num_decode_nodes}) = {expected_infer_nodes}"
            )

        total_infer_gpus = self.deployment.total_infer_nodes * self.deployment.gpus_per_node
        if self.weight_broadcast is not None and self.weight_broadcast.type == "nccl":
            assert self.trainer.weight_broadcast.type == "nccl"
            self.trainer.weight_broadcast.inference_world_size = total_infer_gpus
            assert self.orchestrator.weight_broadcast.type == "nccl"
            self.orchestrator.weight_broadcast.inference_world_size = total_infer_gpus

        return self

    @model_validator(mode="after")
    def auto_setup_inference_clients(self):
        """Auto-configure orchestrator student/teacher clients from the tagged inference list.

        - Student: sets ``dp_rank_count`` from the student deployment's DP size. For SFT mode,
          also sets ``base_url`` (rl/opd rely on the ClientConfig default of
          ``["http://localhost:8000/v1"]`` which already matches the auto-launched student vLLM
          at ``inference.server.port = 8000``).
        - Teacher: sets ``orchestrator.teacher.client.base_url`` to the teacher deployment's
          ``host:port`` (so the orchestrator points at the right server even when student and
          teacher share a host with different ports), and copies the teacher model name from
          inference config when ``orchestrator.teacher.model.name`` was not explicitly set.

        Also enforces that no two inference deployments bind the same ``server.port``.
        """
        ports = [(entry.tag, entry.server.port) for entry in self.inference]
        seen: dict[int, str] = {}
        for tag, port in ports:
            if port in seen:
                raise ValueError(
                    f"Inference deployments {seen[port]!r} and {tag!r} both bind server.port={port}. "
                    "Each [[inference]] entry needs a unique port."
                )
            seen[port] = tag

        student_inf = self.student_inference
        if student_inf is not None:
            client = self.orchestrator.student.client
            if "dp_rank_count" not in client.model_fields_set:
                client.dp_rank_count = student_inf.data_parallel_size_local or student_inf.parallel.dp
            if self.orchestrator.training_mode == "sft" and "base_url" not in client.model_fields_set:
                host = student_inf.server.host or "localhost"
                port = student_inf.server.port
                client.base_url = [f"http://{host}:{port}/v1"]

        teacher_inf = self.teacher_inference
        if teacher_inf is not None:
            from prime_rl.configs.orchestrator import RolloutModelConfig

            if self.orchestrator.teacher is None:
                self.orchestrator.teacher = RolloutModelConfig()
            host = teacher_inf.server.host or "localhost"
            port = teacher_inf.server.port
            if "base_url" not in self.orchestrator.teacher.client.model_fields_set:
                self.orchestrator.teacher.client.base_url = [f"http://{host}:{port}/v1"]
            if "name" not in self.orchestrator.teacher.model.model_fields_set:
                self.orchestrator.teacher.model.name = teacher_inf.model.name

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

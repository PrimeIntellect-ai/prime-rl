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
    OrchestratorConfig,
)
from prime_rl.configs.shared import (
    SlurmConfig,
    VLMConfig,
)
from prime_rl.configs.trainer import (
    BenchConfig,
    FakeDataLoaderConfig,
    TokenizerConfig,
    TrainerConfig,
)
from prime_rl.configs.trainer import (
    FileSystemWeightBroadcastConfig as TrainerFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.trainer import (
    NCCLWeightBroadcastConfig as TrainerNCCLWeightBroadcastConfig,
)
from prime_rl.utils.config import BaseConfig, find_package_resource
from prime_rl.utils.validation import (
    propagate_shared_fields,
    validate_shared_ckpt_config,
    validate_shared_max_async_level,
    validate_shared_max_steps,
    validate_shared_model_name,
    validate_shared_output_dir,
    validate_shared_seq_len,
    validate_shared_tokenizer,
    validate_shared_wandb_config,
    validate_shared_weight_broadcast,
)


class RLExperimentalConfig(BaseConfig):
    pass


class SharedLogConfig(BaseConfig):
    level: str | None = None
    """Log level for trainer and orchestrator. When unset, each sub-config's own log level applies (defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``)."""

    json_logging: bool = False
    """Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.)."""


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
    """Run W&B in offline mode."""

    shared: bool = True
    """Log trainer and orchestrator metrics to a single shared W&B run. Requires wandb SDK ≥ 0.19.9. Incompatible with offline mode."""

    @model_validator(mode="after")
    def validate_shared_not_offline(self):
        if self.shared and self.offline:
            raise ValueError("W&B shared mode requires server connectivity and is incompatible with offline mode")
        return self


class SharedCheckpointConfig(BaseConfig):
    output_dir: Path | None = None
    """Override directory for checkpoints and weights. When set, checkpoints and weight snapshots are written here instead of under the trainer ``output_dir``."""

    interval: int | None = None
    """Interval at which to save checkpoints."""

    resume_step: int | None = None
    """Step to resume from. If None, does not resume from a checkpoint."""

    keep_last: int | None = Field(None, ge=1)
    """Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency."""

    keep_interval: int | None = Field(None, ge=1)
    """Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping."""


class SharedModelConfig(BaseConfig):
    name: str = "Qwen/Qwen3-0.6B"
    """HF model name or local path."""

    vlm: "VLMConfig | None" = None
    """VLM configuration. Set this to enable vision-language model support."""


class SharedWeightBroadcastConfig(BaseConfig):
    type: Literal["nccl", "filesystem"] = "filesystem"
    """Weight broadcast transport."""

    port: int = 29501
    """Port for NCCL weight broadcast."""

    timeout: int = 1200
    """Timeout in seconds for NCCL weight broadcast."""

    quantize_in_weight_transfer: bool = False
    """Use kernel-format FP8 quantized NCCL transfer for weight updates. When disabled, uses default HF checkpoint-format transfer."""


class BaseDeploymentConfig(BaseConfig):
    gpus_per_node: int = 8
    """GPUs per node."""


class SingleNodeDeploymentConfig(BaseDeploymentConfig):
    type: Literal["single_node"] = "single_node"

    num_train_gpus: int = 1
    """GPUs allocated to the trainer."""

    num_infer_gpus: int = 1
    """GPUs allocated to the inference pool. With a single ``[[inference]]`` entry this sizes its ``parallel.dp`` directly; with multiple entries each entry's GPU count comes from its own ``parallel.dp * parallel.tp``."""

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
    type: Literal["multi_node"] = "multi_node"

    num_train_nodes: int
    """Training nodes."""

    num_infer_nodes: int = Field(ge=0)
    """Inference nodes per replica. Set to 0 to skip inference and orchestrator (requires fake data)."""

    num_infer_replicas: int = Field(1, ge=1)
    """Independent inference replicas. Total inference nodes = ``num_infer_nodes * num_infer_replicas``."""

    nodes_per_fsdp_group: int | None = None
    """Training nodes per FSDP island. Auto-sets ``trainer.dp_replicate = num_train_nodes / nodes_per_fsdp_group``."""

    @property
    def total_infer_nodes(self) -> int:
        return self.num_infer_nodes * self.num_infer_replicas


DeploymentConfig: TypeAlias = Annotated[
    SingleNodeDeploymentConfig | MultiNodeDeploymentConfig, Field(discriminator="type")
]


class RLConfig(BaseConfig):
    trainer: TrainerConfig

    orchestrator: OrchestratorConfig

    inference: list[InferenceConfig] = []
    """Inference deployments to launch alongside this RL run, written as repeated ``[[inference]]`` TOML blocks. Each entry is tagged (``tag`` field, defaults to ``student``); the launcher brings up one vLLM subprocess per entry, and the orchestrator routes student/teacher requests by tag. An empty list means no inference server is launched here (useful for elastic pools or manually managed servers). For back-compat, a single ``[inference]`` block is auto-wrapped into a list with ``tag = "student"``."""

    output_dir: Path = Path("outputs")
    """Output directory. Should be unique per experiment."""

    clean_output_dir: bool = False
    """Delete the output directory before starting training. Required to overwrite an output directory that contains checkpoints from a previous run when not resuming."""

    ### Shared configurations

    log: SharedLogConfig = SharedLogConfig()
    """Shared log config. Propagated to trainer and orchestrator."""

    ckpt: SharedCheckpointConfig | None = None
    """Shared checkpoint config. If None, falls back to the sub-config checkpoint settings."""

    wandb: SharedWandbConfig | None = None
    """Shared W&B config. If None, falls back to the sub-config W&B settings."""

    model: SharedModelConfig | None = None
    """Shared model config. If None, falls back to the sub-config model settings."""

    tokenizer: TokenizerConfig | None = None
    """Shared tokenizer config. Propagated to trainer, orchestrator, and inference. If None, each component uses its own tokenizer config (defaulting to model name)."""

    max_steps: int | None = None
    """Shared maximum training steps. If None, falls back to the sub-config ``max_steps``."""

    seq_len: int | None = None
    """Shared sequence length. Propagates to ``trainer.model.seq_len`` and ``orchestrator.seq_len`` only when those values were not explicitly set; explicit per-component values always win."""

    max_async_level: int | None = None
    """Shared async level. If None, falls back to the sub-config ``max_async_level``."""

    weight_broadcast: SharedWeightBroadcastConfig | None = None

    bench: bool = False
    """Benchmark mode. Sets trainer and orchestrator to benchmark mode and, when set, suffixes the W&B project with ``-bench``."""

    deployment: DeploymentConfig = SingleNodeDeploymentConfig()

    slurm: SlurmConfig | None = None
    """SLURM configuration. If None, runs locally."""

    dry_run: bool = False
    """Only validate and dump resolved configs, then exit early."""

    experimental: RLExperimentalConfig = RLExperimentalConfig()

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

    ### Auto-setup shared configs (before sub-config construction)

    @model_validator(mode="before")
    @classmethod
    def auto_setup_shared_configs(cls, data: Any) -> Any:
        """Propagate shared top-level fields into sub-config dicts before sub-configs
        are constructed. See ``validation.propagate_shared_fields`` for the full
        propagation table, transforms, and the mutex rule.
        """
        return propagate_shared_fields(data)

    ### Validate shared configs (after sub-config construction)

    @model_validator(mode="after")
    def validate_shared_configs(self):
        """Validate consistency of shared configs across trainer, orchestrator, and inference."""
        validate_shared_output_dir(self.trainer, self.orchestrator)
        validate_shared_model_name(self.trainer, self.orchestrator, self.student_inference)
        validate_shared_tokenizer(self.trainer, self.orchestrator, self.student_inference)
        validate_shared_max_steps(self.trainer, self.orchestrator)
        validate_shared_max_async_level(self.trainer, self.orchestrator)
        validate_shared_seq_len(self.trainer, self.orchestrator)
        validate_shared_ckpt_config(self.trainer, self.orchestrator)
        validate_shared_wandb_config(self.trainer, self.orchestrator)
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

from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.validation import (
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

    name: Annotated[str | None, Field(description="The W&B run name to use.")] = None

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
    num_teacher_nodes: Annotated[int | None, Field(description="Number of teacher inference nodes.")] = None

    nodes_per_fsdp_group: Annotated[
        int | None,
        Field(
            description="Number of training nodes per FSDP island. Auto-sets trainer.dp_replicate = num_train_nodes / nodes_per_fsdp_group."
        ),
    ] = None

    @property
    def total_infer_nodes(self) -> int:
        return self.num_infer_nodes * self.num_infer_replicas

    @model_validator(mode="after")
    def teacher_inference_not_supported(self):
        if self.num_teacher_nodes is not None:
            raise ValueError("Teacher inference is not yet supported in multi node deployment.")
        return self


DeploymentConfig: TypeAlias = Annotated[
    SingleNodeDeploymentConfig | MultiNodeDeploymentConfig, Field(discriminator="type")
]


class RLConfig(BaseConfig):
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

    ### Validate configs (e.g. raise for unsupported (combinations of) configs)

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
                    "Either set num_infer_nodes > 0 or remove the inference config."
                )
            if self.deployment.num_infer_nodes == 0 and not self.trainer.data.fake and not self.bench:
                raise ValueError(
                    "Must use fake data (trainer.data.fake or bench = true) when num_infer_nodes = 0, "
                    "since no orchestrator or inference server will be running."
                )
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
    def validate_quantize_in_weight_transfer(self):
        if self.weight_broadcast is None or not self.weight_broadcast.quantize_in_weight_transfer:
            return self

        if self.weight_broadcast.type != "nccl":
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires weight_broadcast.type = 'nccl'.")

        if self.inference is None:
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires an inference config.")

        if self.trainer.model.impl != "custom":
            raise ValueError("weight_broadcast.quantize_in_weight_transfer requires trainer.model.impl = 'custom'.")

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

    @model_validator(mode="after")
    def validate_external_rollout_mode(self):
        if self.orchestrator.teacher_rollout_model is None:
            return self

        if self.trainer.loss.type != "sft":
            raise ValueError('orchestrator.teacher_rollout_model is only supported when trainer.loss.type = "sft".')

        if self.inference is not None:
            raise ValueError(
                "inference must be omitted when orchestrator.teacher_rollout_model is configured. "
                "External rollout mode does not use the local inference server."
            )

        if self.orchestrator.use_token_client:
            raise ValueError(
                "orchestrator.use_token_client must be false when orchestrator.teacher_rollout_model is configured."
            )

        return self

    ### Auto-setup shared configs (before sub-config construction)

    @model_validator(mode="before")
    @classmethod
    def auto_setup_shared_configs(cls, data: Any) -> Any:
        """Propagate shared top-level config values into sub-config dicts.

        Runs before sub-configs are constructed so their own validators see the
        final values — e.g. a ModelConfig validator that auto-resolves the parser
        from the model name needs the propagated name at construction time, not
        after.

        Shared configs handled here (fill-if-absent: sub-configs win):
          - `[model]` fields                  → trainer / orchestrator / inference
          - `[log]` fields                    → trainer / orchestrator
          - `[ckpt]` fields                   → trainer / orchestrator
          - `[wandb]` fields + shared/-suffix → trainer / orchestrator
          - `[tokenizer]` fields              → trainer / orchestrator (+ chat_template to inference)
          - `max_steps`                       → trainer / orchestrator
          - `max_async_level`                 → trainer / orchestrator
          - `output_dir`                      → trainer / orchestrator (orchestrator derives a "run_default" subdir)
          - `seq_len`                         → trainer.model / orchestrator

        Sub-config values always take precedence; any mismatch between
        sub-configs themselves, is caught by `validate_shared_configs`
        (after-validator) which runs the full suite of `validate_shared_*`
        checks post-construction.
        """
        if not isinstance(data, dict):
            return data

        data = deepcopy(data)

        # tyro may pass already-constructed sub-config instances rather than
        # raw dicts (when merging CLI overrides on top of a TOML default). Dump
        # them back to dicts, dropping fields still at their class defaults so
        # `fill` can still tell "unset" apart from "set to a default". Discriminator
        # `type` keys are preserved at every level so pydantic can pick the right
        # union variant when it re-validates (otherwise `type="multi_node"` gets
        # stripped as "equals default" and the nested config falls back to the
        # wrong union variant).
        def _dump_preserving_discriminators(obj: Any) -> Any:
            if isinstance(obj, BaseModel):
                dumped = obj.model_dump(exclude_defaults=True)
                if "type" in type(obj).model_fields and "type" not in dumped:
                    dumped["type"] = getattr(obj, "type")
                for field_name in type(obj).model_fields:
                    if field_name in dumped:
                        dumped[field_name] = _dump_preserving_discriminators(getattr(obj, field_name))
                return dumped
            if isinstance(obj, list):
                return [_dump_preserving_discriminators(item) for item in obj]
            if isinstance(obj, dict):
                return {k: _dump_preserving_discriminators(v) for k, v in obj.items()}
            return obj

        for key, value in list(data.items()):
            if isinstance(value, BaseModel):
                data[key] = _dump_preserving_discriminators(value)

        def get(path: str) -> Any | None:
            """Read a dotted path (e.g. `model.name`) from raw config data. Returns
            None if any intermediate key is missing or not a dict."""
            node: Any = data
            for part in path.split("."):
                if not isinstance(node, dict) or part not in node:
                    return None
                node = node[part]
            return node

        def fill(path: str, value: Any) -> None:
            """Fill a value at dotted path in raw data, only if the slot is empty.

            Only creates intermediate dicts for nested keys (e.g. 'model' in
            'trainer.model.name' but not 'trainer').
            """
            parts = path.split(".")
            # Don't create the top-level sub-config if it doesn't exist
            if parts[0] not in data or not isinstance(data[parts[0]], dict):
                return
            node = data
            for part in parts[:-1]:
                if not isinstance(node, dict):
                    return
                node = node.setdefault(part, {})
            if isinstance(node, dict) and parts[-1] not in node:
                node[parts[-1]] = value

        # [model] → sub-config model dicts
        model_name = get("model.name")
        if model_name is not None:
            fill("trainer.model.name", model_name)
            fill("inference.model.name", model_name)
            # Orchestrator follows inference model name (which may differ from shared)
            fill("orchestrator.model.name", get("inference.model.name") or model_name)
        model_vlm = get("model.vlm")
        if model_vlm is not None:
            for target in ("trainer.model.vlm", "inference.model.vlm", "orchestrator.model.vlm"):
                fill(target, model_vlm)

        # [log] → trainer/orchestrator log dicts
        for key in ("level", "json_logging"):
            val = get(f"log.{key}")
            if val is not None:
                fill(f"trainer.log.{key}", val)
                fill(f"orchestrator.log.{key}", val)

        # [ckpt] → trainer/orchestrator ckpt dicts (output_dir is trainer-only).
        # Presence of shared [ckpt] (even empty) enables ckpt on both sub-configs.
        if get("ckpt") is not None:
            fill("trainer.ckpt", {})
            fill("orchestrator.ckpt", {})
        ckpt_output_dir = get("ckpt.output_dir")
        if ckpt_output_dir is not None:
            fill("trainer.ckpt.output_dir", ckpt_output_dir)
        for field in ("interval", "resume_step", "keep_last", "keep_interval"):
            val = get(f"ckpt.{field}")
            if val is not None:
                fill(f"trainer.ckpt.{field}", val)
                fill(f"orchestrator.ckpt.{field}", val)

        # [wandb] → trainer/orchestrator wandb dicts.
        # Presence of shared [wandb] (even empty) enables wandb on both sub-configs.
        if get("wandb") is not None:
            fill("trainer.wandb", {})
            fill("orchestrator.wandb", {})
        for field in ("project", "offline"):
            val = get(f"wandb.{field}")
            if val is not None:
                fill(f"trainer.wandb.{field}", val)
                fill(f"orchestrator.wandb.{field}", val)

        # W&B name: in shared mode (the default), both sub-configs use the same
        # name. In non-shared mode (wandb.shared = false), suffix with -trainer/
        # -orchestrator so the runs are distinguishable.
        wandb_name = get("wandb.name")
        if wandb_name:
            non_shared = get("wandb.shared") is False
            fill("trainer.wandb.name", f"{wandb_name}-trainer" if non_shared else wandb_name)
            fill("orchestrator.wandb.name", f"{wandb_name}-orchestrator" if non_shared else wandb_name)

        # wandb.name → orchestrator.prime_monitor.run_name, but only when
        # prime_monitor is already enabled (don't fabricate it).
        if wandb_name and get("orchestrator.prime_monitor") is not None:
            fill("orchestrator.prime_monitor.run_name", wandb_name)

        # [tokenizer] → trainer/orchestrator tokenizer dicts (fill-if-absent).
        # If tokenizer.name is left absent here, each sub-config's own
        # auto_setup_tokenizer fills it from model.name during construction.
        for field in ("name", "trust_remote_code", "chat_template"):
            val = get(f"tokenizer.{field}")
            if val is not None:
                fill(f"trainer.tokenizer.{field}", val)
                fill(f"orchestrator.tokenizer.{field}", val)
        # chat_template flows trainer.tokenizer → inference.model (vLLM --chat-template).
        # Read after the propagation above so we pick up shared → trainer flow.
        chat_template = get("trainer.tokenizer.chat_template")
        if chat_template is not None:
            fill("inference.model.chat_template", chat_template)

        # max_steps → trainer/orchestrator
        max_steps = get("max_steps")
        if max_steps is not None:
            fill("trainer.max_steps", max_steps)
            fill("orchestrator.max_steps", max_steps)

        # max_async_level → trainer/orchestrator
        max_async_level = get("max_async_level")
        if max_async_level is not None:
            fill("trainer.max_async_level", max_async_level)
            fill("orchestrator.max_async_level", max_async_level)

        # output_dir → trainer/orchestrator (orchestrator derives a "run_default" subdir)
        output_dir = get("output_dir")
        if output_dir is not None:
            fill("trainer.output_dir", output_dir)
            fill("orchestrator.output_dir", f"{output_dir}/run_default")

        # seq_len → trainer.model.seq_len and orchestrator.seq_len (fill-if-absent)
        seq_len = get("seq_len")
        if seq_len is not None:
            fill("trainer.model.seq_len", seq_len)
            fill("orchestrator.seq_len", seq_len)

        return data

    ### Validate shared configs (after sub-config construction)

    @model_validator(mode="after")
    def validate_shared_configs(self):
        """Validate consistency of shared configs across trainer, orchestrator, and inference."""
        validate_shared_output_dir(self.trainer, self.orchestrator)
        validate_shared_ckpt_config(self.trainer, self.orchestrator)
        validate_shared_wandb_config(self.trainer, self.orchestrator)
        validate_shared_model_name(self.trainer, self.orchestrator, self.inference)
        validate_shared_tokenizer(self.trainer, self.orchestrator, self.inference)
        validate_shared_max_steps(self.trainer, self.orchestrator)
        validate_shared_max_async_level(self.trainer, self.orchestrator)
        validate_shared_seq_len(self.trainer, self.orchestrator)
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
                get_logger().warning(
                    "LoRA is enabled, but inference is not configured. When manually starting the inference server, "
                    "make sure to set --enable_lora and --max-lora-rank."
                )

        return self

    @model_validator(mode="after")
    def auto_setup_router_replay(self):
        if self.trainer.enable_router_replay:
            if self.inference is not None:
                if self.inference.enable_return_routed_experts is False:
                    get_logger().warning(
                        "Router replay is enabled, but inference.enable_return_routed_experts is False. Setting to True."
                    )
                self.inference.enable_return_routed_experts = True
            else:
                get_logger().warning(
                    "Router replay is enabled, but inference is not configured. When manually starting the inference server, make sure to pass `--enable-return-routed-experts` to the vLLM server."
                )
        return self

    @model_validator(mode="after")
    def auto_setup_deployment(self):
        if self.deployment.type == "single_node":  # single-node
            # set num_train_workers to the number of data replicas
            non_data_parallel_size = self.trainer.model.cp
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
                # Ensure api_server_count matches DP so all workers are created.
                # Without this, the NCCL broadcast group expects dp*tp workers
                # but only api_server_count*tp exist, causing a deadlock.
                dp = self.inference.parallel.dp
                if self.inference.api_server_count < dp and not self.inference.enable_lora:
                    self.inference.api_server_count = dp

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
                self.inference is not None
                and self.inference.enable_expert_parallel
                and self.inference.deployment.type != "disaggregated"
            ):
                inference_tp = self.inference.parallel.tp
                if self.deployment.gpus_per_node % inference_tp != 0:
                    raise ValueError(
                        "deployment.gpus_per_node must be divisible by inference.parallel.tp "
                        "when inference.enable_expert_parallel is enabled in multi-node deployment."
                    )

                inferred_dp_local = self.deployment.gpus_per_node // inference_tp
                total_infer_gpus = self.deployment.num_infer_nodes * self.deployment.gpus_per_node
                expected_global_world_size = self.inference.parallel.dp * inference_tp
                if expected_global_world_size != total_infer_gpus:
                    raise ValueError(
                        "For multi-node expert parallel inference, inference.parallel.dp * inference.parallel.tp "
                        f"must match total inference GPUs ({total_infer_gpus}), got {expected_global_world_size}."
                    )

                if self.inference.data_parallel_size_local is None:
                    self.inference.data_parallel_size_local = inferred_dp_local
                elif self.inference.data_parallel_size_local != inferred_dp_local:
                    raise ValueError(
                        "inference.data_parallel_size_local must equal deployment.gpus_per_node / inference.parallel.tp "
                        f"({inferred_dp_local}) when inference.enable_expert_parallel is enabled in multi-node deployment."
                    )

                if not self.inference.enable_lora and self.inference.api_server_count == self.inference.parallel.dp:
                    self.inference.api_server_count = inferred_dp_local

            # Auto-infer DP and api_server_count for standard multi-node inference.
            # Without EP, vLLM only creates api_server_count * tp workers per node,
            # not gpus_per_node workers. If DP isn't set, the broadcast group expects
            # more workers than exist, deadlocking NCCL init.
            if (
                self.inference is not None
                and not self.inference.enable_expert_parallel
                and self.inference.deployment.type != "disaggregated"
            ):
                dp_per_node = self.deployment.gpus_per_node // self.inference.parallel.tp
                if self.inference.parallel.dp == 1 and dp_per_node > 1:
                    self.inference.parallel.dp = dp_per_node
                if self.inference.data_parallel_size_local is None and dp_per_node > 1:
                    self.inference.data_parallel_size_local = dp_per_node
                if self.inference.api_server_count == 1 and dp_per_node > 1:
                    self.inference.api_server_count = dp_per_node

            if self.weight_broadcast is not None and self.weight_broadcast.type == "nccl":
                # Compute inference_world_size from actual worker count per server:
                # each api_server runs tp workers that participate in collective_rpc.
                api_server_count = self.inference.api_server_count if self.inference else 1
                tp = self.inference.parallel.tp if self.inference else 1
                total_infer_workers = self.deployment.total_infer_nodes * api_server_count * tp
                assert self.trainer.weight_broadcast.type == "nccl"
                self.trainer.weight_broadcast.host = "0.0.0.0"
                self.trainer.weight_broadcast.inference_world_size = total_infer_workers
                assert self.orchestrator.weight_broadcast.type == "nccl"
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
    def auto_setup_dp_rank_count(self):
        """Auto-set orchestrator client dp_rank_count from inference DP size.

        Uses data_parallel_size_local (per-node DP) when set, since each base URL
        points to a single node whose API server only knows about its local ranks.
        Falls back to the global parallel.dp for single-node setups.
        """
        if self.inference is not None and "dp_rank_count" not in self.orchestrator.client.model_fields_set:
            self.orchestrator.client.dp_rank_count = (
                self.inference.data_parallel_size_local or self.inference.parallel.dp
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

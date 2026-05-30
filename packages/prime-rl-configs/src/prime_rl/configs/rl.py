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

    inference: InferenceConfig | None = None
    """Inference server configuration. If None, the rl entrypoint will not start an inference server (useful for elastic inference pools or manually started servers)."""

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

    weight_broadcast: SharedWeightBroadcastConfig | None = None

    bench: bool = False
    """Benchmark mode. Sets trainer and orchestrator to benchmark mode and, when set, suffixes the W&B project with ``-bench``."""

    deployment: DeploymentConfig = SingleNodeDeploymentConfig()

    slurm: SlurmConfig | None = None
    """SLURM configuration. If None, runs locally."""

    dry_run: bool = False
    """Only validate and dump resolved configs, then exit early."""

    experimental: RLExperimentalConfig = RLExperimentalConfig()

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
        validate_shared_model_name(self.trainer, self.orchestrator, self.inference)
        validate_shared_tokenizer(self.trainer, self.orchestrator, self.inference)
        validate_shared_max_steps(self.trainer, self.orchestrator)
        validate_shared_seq_len(self.trainer, self.orchestrator)
        validate_shared_ckpt_config(self.trainer, self.orchestrator)
        validate_shared_wandb_config(self.trainer, self.orchestrator)
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
    def validate_eplb_requires_quantized_weight_transfer(self):
        if self.inference is None or not self.inference.enable_eplb:
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

            if self.inference is not None:
                self.inference.enable_lora = True
                self.inference.max_lora_rank = self.trainer.model.lora.rank
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
                if self.inference.enable_return_routed_experts is False:
                    warnings.warn(
                        "Router replay is enabled, but inference.enable_return_routed_experts is False. Setting to True.",
                        stacklevel=2,
                    )
                self.inference.enable_return_routed_experts = True
            else:
                warnings.warn(
                    "Router replay is enabled, but inference is not configured. When manually starting the inference server, make sure to pass `--enable-return-routed-experts` to the vLLM server.",
                    stacklevel=2,
                )
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
        if "inference_metrics_roles" not in self.orchestrator.model_fields_set:
            role_order = ["prefill"] * infer_deploy.num_prefill_nodes + ["decode"] * infer_deploy.num_decode_nodes
            self.orchestrator.inference_metrics_roles = role_order * self.deployment.num_infer_replicas
        if self.weight_broadcast is not None and self.weight_broadcast.type == "nccl":
            assert self.trainer.weight_broadcast.type == "nccl"
            self.trainer.weight_broadcast.inference_world_size = total_infer_gpus
            assert self.orchestrator.weight_broadcast.type == "nccl"
            self.orchestrator.weight_broadcast.inference_world_size = total_infer_gpus

        return self

    @model_validator(mode="after")
    def auto_setup_inference_client(self):
        """Auto-configure orchestrator student client from the inference server config.

        For all modes, sets dp_rank_count from inference DP size. For SFT mode,
        also sets base_url - rl/opd rely on the ClientConfig default
        (``["http://localhost:8000/v1"]``) which already matches the auto-launched
        student vLLM at inference.server.port = 8000.
        """
        if self.inference is None:
            return self
        client = self.orchestrator.student.client
        if "dp_rank_count" not in client.model_fields_set:
            client.dp_rank_count = self.inference.data_parallel_size_local or self.inference.parallel.dp
        if self.orchestrator.training_mode == "sft" and "base_url" not in client.model_fields_set:
            host = self.inference.server.host or "localhost"
            port = self.inference.server.port
            client.base_url = [f"http://{host}:{port}/v1"]
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

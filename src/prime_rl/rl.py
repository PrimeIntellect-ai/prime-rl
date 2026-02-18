import copy
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread
from typing import Annotated, Literal

import pynvml
import tomli_w
from pydantic import Field, model_validator

from prime_rl.inference.config import All2AllBackend, InferenceConfig
from prime_rl.inference.config import WeightBroadcastConfig as InferenceWeightBroadcastConfig
from prime_rl.orchestrator.config import CheckpointConfig as OrchestratorCheckpointConfig
from prime_rl.orchestrator.config import FileSystemWeightBroadcastConfig as OrchestratorFileSystemWeightBroadcastConfig
from prime_rl.orchestrator.config import NCCLWeightBroadcastConfig as OrchestratorNCCLWeightBroadcastConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.trainer.config import CheckpointConfig as TrainerCheckpointConfig
from prime_rl.trainer.rl.config import FakeDataLoaderConfig
from prime_rl.trainer.rl.config import FileSystemWeightBroadcastConfig as TrainerFileSystemWeightBroadcastConfig
from prime_rl.trainer.rl.config import NCCLWeightBroadcastConfig as TrainerNCCLWeightBroadcastConfig
from prime_rl.trainer.rl.config import RLTrainerConfig as TrainerConfig
from prime_rl.utils.config import WandbConfig, WandbWithExtrasConfig
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pydantic_config import BaseSettings, parse_argv
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_free_port,
    get_log_dir,
    get_rollout_dir,
)
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


class PDDisaggConfig(BaseSettings):
    """Configures PD disaggregation with dedicated prefill/decode servers."""

    enabled: Annotated[
        bool,
        Field(description="Whether to enable prefill/decode disaggregation for inference."),
    ] = False

    prefill_gpu_ids: Annotated[
        list[int],
        Field(
            description="Flattened GPU IDs for prefill workers. IDs are grouped in contiguous chunks of `prefill_tp` (or `inference.parallel.tp` if unset), one TP group per prefill worker.",
        ),
    ] = Field(default_factory=list)

    decode_gpu_ids: Annotated[
        list[int],
        Field(
            description="Flattened GPU IDs for decode workers. IDs are grouped in contiguous chunks of `decode_tp` (or `inference.parallel.tp` if unset), one TP group per decode worker.",
        ),
    ] = Field(default_factory=list)

    prefill_tp: Annotated[
        int | None,
        Field(
            ge=1,
            description="Tensor-parallel size for prefill workers. If None, falls back to inference.parallel.tp.",
        ),
    ] = None

    decode_tp: Annotated[
        int | None,
        Field(
            ge=1,
            description="Tensor-parallel size for decode workers. If None, falls back to inference.parallel.tp.",
        ),
    ] = None

    host: Annotated[
        str | None,
        Field(
            description="Host used by the local PD proxy and backend URLs. If None, defaults to inference.server.host when set, otherwise 127.0.0.1.",
        ),
    ] = None

    proxy_port: Annotated[
        int,
        Field(description="Port for the PD proxy OpenAI-compatible endpoint."),
    ] = 8000

    prefill_port: Annotated[
        int,
        Field(description="Base HTTP port for prefill workers. Worker i uses `prefill_port + i`."),
    ] = 8100

    decode_port: Annotated[
        int,
        Field(description="Base HTTP port for decode workers. Worker i uses `decode_port + i`."),
    ] = 8200

    prefill_kv_port: Annotated[
        int,
        Field(description="Base KV transfer port for prefill workers. Worker i uses `prefill_kv_port + i`."),
    ] = 14579

    decode_kv_port: Annotated[
        int,
        Field(description="Base KV transfer port for decode workers. Worker i uses `decode_kv_port + i`."),
    ] = 14580

    kv_connector: Annotated[
        str,
        Field(description="vLLM KV connector to use for PD disaggregation."),
    ] = "P2pNcclConnector"

    kv_send_type: Annotated[
        Literal["PUT_ASYNC", "PUT", "GET"],
        Field(description="Transfer mode for P2pNcclConnector."),
    ] = "PUT_ASYNC"

    prefill_all2all_backend: Annotated[
        All2AllBackend | None,
        Field(
            description="Optional all2all backend override for prefill workers (used with expert parallelism). If None, uses inference.all2all_backend.",
        ),
    ] = None

    decode_all2all_backend: Annotated[
        All2AllBackend | None,
        Field(
            description="Optional all2all backend override for decode workers (used with expert parallelism). If None, uses inference.all2all_backend.",
        ),
    ] = None

    auto_ports: Annotated[
        bool,
        Field(
            description="Automatically pick available, non-overlapping proxy/server/KV ports at runtime. Recommended for simple local setups.",
        ),
    ] = True


class RLConfig(BaseSettings):
    """Configures an RL training run."""

    ### Submodule configurations

    trainer: TrainerConfig
    orchestrator: OrchestratorConfig
    inference: Annotated[
        InferenceConfig | None,
        Field(
            description="The inference config. If None, will not start an inference process. Only viable, if an inference server was started manually."
        ),
    ] = None

    ### Top-level configurations

    log: Annotated[
        SharedLogConfig,
        Field(
            description="Shared log configs. If None, will fallback to the log configs specified on submodule configs."
        ),
    ] = SharedLogConfig()

    clean: Annotated[
        bool,
        Field(
            description="Whether to clean the rollouts, checkpoint, checkpoint weights and logs directories at the beginning of the run. If True, will forceably, and irreversibly, delete all directories.",
        ),
    ] = True

    inference_gpu_ids: Annotated[list[int], Field(description="The GPU IDs to use for inference.")] = [0]
    trainer_gpu_ids: Annotated[list[int], Field(description="The GPU IDs to use for trainer.")] = [1]
    teacher_gpu_ids: Annotated[
        list[int] | None,
        Field(
            description="The GPU IDs to use for teacher inference. If None, teacher inference server will not be started."
        ),
    ] = None

    teacher_inference: Annotated[
        InferenceConfig | None,
        Field(
            description="The teacher inference config. If None, will use the same config as inference (if available) or a default config. Only used when teacher_gpu_ids is set."
        ),
    ] = None

    pd_disagg: Annotated[
        PDDisaggConfig,
        Field(
            description="Prefill/decode disaggregation setup for vLLM-based inference. Uses a local proxy that routes requests through prefill then decode.",
        ),
    ] = PDDisaggConfig()

    ### Shared configurations

    output_dir: Annotated[
        Path,
        Field(description="The directory to store the outputs. Should typically be set to an experiment identifier."),
    ] = Path("outputs")  # NOTE: Must match `OUTPUT_DIR` in `tmux.sh` to see logs

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
            description="The sequence length to use. If set, will configure both trainer.model.seq_len and orchestrator.seq_len to this value. If None, each can be set independently."
        ),
    ] = None

    max_async_level: Annotated[
        int | None,
        Field(
            description="The async level to use. If None, will fallback to the async level specified on submodule configs."
        ),
    ] = None

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the trainer and orchestrator to benchmark mode and, if present, configure the W&B project by suffixing the project with `-bench`.",
        ),
    ] = False

    weight_broadcast: Annotated[
        SharedWeightBroadcastConfig | None, Field(description="The weight broadcast config.")
    ] = None

    dump_config: Annotated[
        Path | None,
        Field(
            description="If set, dump resolved subconfigs (trainer, orchestrator, inference) to this directory and exit without starting any processes."
        ),
    ] = None

    @model_validator(mode="after")
    def auto_setup_dp(self):
        if (
            self.inference
            and not self.pd_disagg.enabled
            and len(self.inference_gpu_ids) != self.inference.parallel.dp * self.inference.parallel.tp
        ):
            assert len(self.inference_gpu_ids) % self.inference.parallel.tp == 0, (
                "Number of inference GPUs must be divisible by the tensor parallel size"
            )
            self.inference.parallel.dp = len(self.inference_gpu_ids) // self.inference.parallel.tp
        return self

    @model_validator(mode="after")
    def auto_setup_pd_disagg(self):
        if not self.pd_disagg.enabled:
            return self

        if self.inference is None:
            raise ValueError("pd_disagg.enabled requires an [inference] config.")

        if self.orchestrator.client.is_elastic:
            raise ValueError("pd_disagg is not supported with orchestrator.client.elastic.")

        default_tp = self.inference.parallel.tp
        prefill_tp = self.pd_disagg.prefill_tp or default_tp
        decode_tp = self.pd_disagg.decode_tp or default_tp
        if len(self.pd_disagg.prefill_gpu_ids) == 0:
            raise ValueError("pd_disagg.prefill_gpu_ids cannot be empty when pd_disagg.enabled is true.")
        if len(self.pd_disagg.decode_gpu_ids) == 0:
            raise ValueError("pd_disagg.decode_gpu_ids cannot be empty when pd_disagg.enabled is true.")

        if len(self.pd_disagg.prefill_gpu_ids) % prefill_tp != 0:
            raise ValueError(
                f"pd_disagg.prefill_gpu_ids must be a multiple of prefill_tp ({prefill_tp}); "
                "each prefill worker uses one TP group."
            )
        if len(self.pd_disagg.decode_gpu_ids) % decode_tp != 0:
            raise ValueError(
                f"pd_disagg.decode_gpu_ids must be a multiple of decode_tp ({decode_tp}); "
                "each decode worker uses one TP group."
            )

        prefill_workers = len(self.pd_disagg.prefill_gpu_ids) // prefill_tp
        decode_workers = len(self.pd_disagg.decode_gpu_ids) // decode_tp

        if len(set(self.pd_disagg.prefill_gpu_ids + self.pd_disagg.decode_gpu_ids)) != len(
            self.pd_disagg.prefill_gpu_ids + self.pd_disagg.decode_gpu_ids
        ):
            raise ValueError("pd_disagg.prefill_gpu_ids and pd_disagg.decode_gpu_ids must not overlap.")

        if not self.pd_disagg.auto_ports:
            ports = [self.pd_disagg.proxy_port]
            ports.extend(get_contiguous_ports(self.pd_disagg.prefill_port, prefill_workers))
            ports.extend(get_contiguous_ports(self.pd_disagg.decode_port, decode_workers))
            ports.extend(get_contiguous_ports(self.pd_disagg.prefill_kv_port, prefill_workers))
            ports.extend(get_contiguous_ports(self.pd_disagg.decode_kv_port, decode_workers))
            if len(set(ports)) != len(ports):
                raise ValueError("pd_disagg proxy/server/KV port ranges must all be distinct.")

        host = self.pd_disagg.host or self.inference.server.host or "127.0.0.1"
        self.pd_disagg.host = host
        self.orchestrator.client.base_url = [f"http://{host}:{self.pd_disagg.proxy_port}/v1"]
        return self

    @model_validator(mode="after")
    def auto_setup_num_train_workers(self):
        non_data_parallel_size = self.trainer.model.cp * self.trainer.model.tp

        if len(self.trainer_gpu_ids) > 1:
            self.orchestrator.num_train_workers = len(self.trainer_gpu_ids) // non_data_parallel_size

        return self

    @model_validator(mode="after")
    def auto_setup_logs(self):
        # Copy log level
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

    ### Setup and validate shared configs

    @model_validator(mode="after")
    def auto_setup_ckpt(self):
        # If specified, automatically setup checkpoint configs for trainer and orchestrator
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
        # If specified, automatically use shared W&B project for orchestrator and trainer
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
    def auto_setup_bench(self):
        if self.bench:
            # Set trainer and orchestrator to benchmark mode
            self.trainer.bench = True
            self.orchestrator.bench = True

            # Configure the trainer fake data to match the orchestrator config
            self.trainer.data.fake = FakeDataLoaderConfig(
                batch_size=self.orchestrator.batch_size,
            )

        trainer_bench_enabled = self.trainer.bench is not None
        if trainer_bench_enabled != self.orchestrator.bench:
            raise ValueError(
                f"Trainer benchmark mode ({self.trainer.bench}) and orchestrator benchmark mode ({self.orchestrator.bench}) are not the same. Please specify the same benchmark mode for both."
            )

        return self

    @model_validator(mode="after")
    def auto_setup_model(self):
        # Use the same model for trainer, orchestrator and inference
        if self.model is not None:
            self.trainer.model.name = self.model.name
            self.orchestrator.model.name = self.model.name
            if self.inference is not None:
                self.inference.model.name = self.model.name

        validate_shared_model_name(self.trainer, self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def auto_setup_max_steps(self):
        # If specified, use the same max steps for trainer and orchestrator
        if self.max_steps is not None:
            self.trainer.max_steps = self.max_steps
            self.orchestrator.max_steps = self.max_steps

        validate_shared_max_steps(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_async_level(self):
        # If specified, use the same async level for trainer and orchestrator
        if self.max_async_level is not None:
            self.trainer.max_async_level = self.max_async_level
            self.orchestrator.max_async_level = self.max_async_level

        validate_shared_max_async_level(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_output_dir(self):
        # If specified, use the same outputs directory for trainer and orchestrator
        if self.output_dir is not None:
            self.trainer.output_dir = self.output_dir
            self.orchestrator.output_dir = self.output_dir / "run_default"

        validate_shared_output_dir(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_seq_len(self):
        # If specified, use the same seq_len for trainer and orchestrator
        if self.seq_len is not None:
            self.trainer.model.seq_len = self.seq_len
            self.orchestrator.seq_len = self.seq_len

        if self.trainer.model.seq_len < self.orchestrator.seq_len:
            raise ValueError(
                f"Trainer model seq_len ({self.trainer.model.seq_len}) must be >= orchestrator seq_len ({self.orchestrator.seq_len}). "
                f"The trainer needs to be able to handle sequences at least as long as those produced by the orchestrator."
            )

        return self

    @model_validator(mode="after")
    def auto_setup_weight_broadcast(self):
        if self.weight_broadcast is not None:
            if self.weight_broadcast.type == "nccl":
                inference_world_size = self.inference.parallel.dp * self.inference.parallel.tp if self.inference else 1
                if self.pd_disagg.enabled:
                    default_tp = self.inference.parallel.tp
                    prefill_tp = self.pd_disagg.prefill_tp or default_tp
                    decode_tp = self.pd_disagg.decode_tp or default_tp
                    if prefill_tp != decode_tp:
                        raise ValueError(
                            "pd_disagg with NCCL weight broadcast currently requires matching prefill_tp and "
                            "decode_tp so inference ranks can be assigned consistently."
                        )
                    inference_world_size = len(self.pd_disagg.prefill_gpu_ids) + len(self.pd_disagg.decode_gpu_ids)
                self.trainer.weight_broadcast = TrainerNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type, inference_world_size=inference_world_size
                )
                self.orchestrator.weight_broadcast = OrchestratorNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type
                )
            elif self.weight_broadcast.type == "filesystem":
                self.trainer.weight_broadcast = TrainerFileSystemWeightBroadcastConfig()
                self.orchestrator.weight_broadcast = OrchestratorFileSystemWeightBroadcastConfig()
            if self.inference is not None:
                self.inference.weight_broadcast = InferenceWeightBroadcastConfig(type=self.weight_broadcast.type)

        validate_shared_weight_broadcast(self.trainer, self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def auto_setup_lora(self):
        if self.trainer.model.lora is not None:
            if self.trainer.weight_broadcast.type == "nccl":
                raise ValueError("NCCL weight broadcast does not support LoRA yet.")

            # Ensure orchestrator has LoRA config
            if self.orchestrator.model.lora is None:
                from prime_rl.orchestrator.config import LoRAConfig

                self.orchestrator.model.lora = LoRAConfig()

            # Validate orchestrator LoRA rank/alpha don't conflict with trainer
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

            # Propagate rank/alpha from trainer when not explicitly set
            if self.orchestrator.model.lora.rank is None:
                self.orchestrator.model.lora.rank = self.trainer.model.lora.rank

            if self.orchestrator.model.lora.alpha is None:
                self.orchestrator.model.lora.alpha = self.trainer.model.lora.alpha

            # Auto-generate name if not provided
            if self.orchestrator.model.lora.name is None:
                self.orchestrator.model.lora.name = (
                    f"r{self.orchestrator.model.lora.rank}-a{self.orchestrator.model.lora.alpha}"
                )

            if self.inference is not None:
                self.inference.enable_lora = True
                self.inference.max_lora_rank = self.trainer.model.lora.rank
            else:
                warnings.warn(
                    "LoRA is enabled, but inference is not configured. When manually starting the inference server, make sure to set `--enable_lora` and `--max-lora-rank`."
                )

        return self

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

    @model_validator(mode="after")
    def validate_enough_devices_for_nccl(self):
        if self.trainer.weight_broadcast.type == "nccl":
            inference_gpu_ids = (
                self.pd_disagg.prefill_gpu_ids + self.pd_disagg.decode_gpu_ids
                if self.pd_disagg.enabled
                else self.inference_gpu_ids
            )
            num_gpus = len(set(self.trainer_gpu_ids + inference_gpu_ids))
            if num_gpus < 2:
                raise ValueError("NCCL weight broadcast requires at least 2 GPUs to build the broadcast process group.")
        return self

    @model_validator(mode="after")
    def auto_setup_teacher_inference(self):
        """Auto-configure teacher inference server and orchestrator teacher_model client."""
        if self.teacher_gpu_ids is None or len(self.teacher_gpu_ids) == 0:
            return self

        import copy

        from prime_rl.orchestrator.config import TeacherModelConfig

        # Create or complete teacher_inference config
        if self.teacher_inference is None:
            self.teacher_inference = copy.deepcopy(self.inference) if self.inference else InferenceConfig()
            # Avoid port conflict with main inference by using next port
            if self.inference is not None:
                self.teacher_inference.server.port = self.inference.server.port + 1
        elif self.inference is not None and self.teacher_inference.server.port == self.inference.server.port:
            raise ValueError(
                f"teacher_inference.server.port ({self.teacher_inference.server.port}) conflicts with "
                f"inference.server.port ({self.inference.server.port}). "
                "Either use different ports or let teacher_inference be auto-configured."
            )

        # Auto-configure DP based on GPU count
        tp = self.teacher_inference.parallel.tp
        if len(self.teacher_gpu_ids) != self.teacher_inference.parallel.dp * tp:
            assert len(self.teacher_gpu_ids) % tp == 0, (
                "Number of teacher GPUs must be divisible by tensor parallel size"
            )
            assert len(self.teacher_gpu_ids) > 0, "teacher_gpu_ids cannot be empty"
            self.teacher_inference.parallel.dp = len(self.teacher_gpu_ids) // tp

        # Auto-configure orchestrator's teacher_model client
        if self.orchestrator.teacher_model is None:
            self.orchestrator.teacher_model = TeacherModelConfig()
        host = self.teacher_inference.server.host or "localhost"
        port = self.teacher_inference.server.port
        self.orchestrator.teacher_model.client.base_url = [f"http://{host}:{port}/v1"]
        self.orchestrator.teacher_model.model.name = self.teacher_inference.model.name

        return self

    @model_validator(mode="after")
    def validate_teacher_model(self):
        if self.trainer.loss.teacher_tau > 0 and not self.orchestrator.teacher_model:
            raise ValueError(
                "teacher_model must be configured when teacher_tau > 0. "
                "Either set teacher_tau = 0, set teacher_gpu_ids, or configure teacher_model manually."
            )
        return self


def cleanup_threads(threads: list[Thread]):
    for thread in threads:
        thread.join(timeout=5)


def cleanup_processes(processes: list[Popen]):
    for process in processes:
        if process.poll() is None:  # Process is still running
            process.terminate()
            try:
                process.wait(timeout=60)  # 60 seconds to terminate gracefully
            except subprocess.TimeoutExpired:
                process.kill()


def monitor_process(process: Popen, stop_event: Event, error_queue: list, process_name: str):
    """Monitor a subprocess and signal errors via shared queue"""
    try:
        # Wait for process to complete
        process.wait()

        if process.returncode != 0:
            err_msg = f"{process_name.capitalize()} failed with exit code {process.returncode}"
            if process.stderr:
                err_msg += f"\n{process.stderr.read().decode('utf-8')}"
            error_queue.append(RuntimeError(err_msg))
        stop_event.set()
    except Exception as e:
        error_queue.append(RuntimeError(f"Error monitoring {process_name}: {e}"))
        stop_event.set()


def check_gpus_available(gpu_ids: list[int]) -> None:
    """Raise error if there are existing processes on the specified GPUs."""
    pynvml.nvmlInit()

    occupied = []
    for gpu_id in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if processes:
            pids = [p.pid for p in processes]
            occupied.append((gpu_id, pids))

    if occupied:
        msg = "Existing processes found on GPUs:\n"
        for gpu_id, pids in occupied:
            msg += f"  GPU {gpu_id}: PIDs {pids}\n"
        msg += "Kill these processes or use different GPUs."
        raise RuntimeError(msg)


def write_subconfigs(config: RLConfig, output_dir: Path) -> None:
    """Write resolved subconfigs to disk as TOML files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "trainer.toml", "wb") as f:
        tomli_w.dump(config.trainer.model_dump(exclude_none=True, mode="json"), f)

    with open(output_dir / "orchestrator.toml", "wb") as f:
        tomli_w.dump(config.orchestrator.model_dump(exclude_none=True, mode="json"), f)

    if config.inference is not None:
        with open(output_dir / "inference.toml", "wb") as f:
            tomli_w.dump(config.inference.model_dump(exclude_none=True, mode="json"), f)

    if config.teacher_inference is not None:
        with open(output_dir / "teacher_inference.toml", "wb") as f:
            tomli_w.dump(config.teacher_inference.model_dump(exclude_none=True, mode="json"), f)


def write_toml_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def get_contiguous_ports(base_port: int, count: int) -> list[int]:
    return [base_port + i for i in range(count)]


def chunk_gpu_ids(gpu_ids: list[int], chunk_size: int) -> list[list[int]]:
    return [gpu_ids[i : i + chunk_size] for i in range(0, len(gpu_ids), chunk_size)]


def is_port_available(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.listen(1)
            return True
        except OSError:
            return False


def reserve_contiguous_ports(count: int, start_port: int, used_ports: set[int]) -> list[int]:
    if count <= 0:
        return []

    port = max(start_port, 1024)
    while True:
        candidates = [port + i for i in range(count)]
        if any(candidate in used_ports for candidate in candidates):
            port += 1
            continue
        if all(is_port_available(candidate) for candidate in candidates):
            used_ports.update(candidates)
            return candidates
        port += 1


def resolve_pd_runtime_endpoints(config: RLConfig, logger) -> None:
    if config.inference is None or not config.pd_disagg.enabled:
        return

    pd = config.pd_disagg
    default_tp = config.inference.parallel.tp
    prefill_tp = pd.prefill_tp or default_tp
    decode_tp = pd.decode_tp or default_tp
    prefill_workers = len(pd.prefill_gpu_ids) // prefill_tp
    decode_workers = len(pd.decode_gpu_ids) // decode_tp

    if pd.host is None:
        pd.host = config.inference.server.host or "127.0.0.1"

    if pd.auto_ports:
        used_ports: set[int] = set()
        pd.proxy_port = reserve_contiguous_ports(1, pd.proxy_port, used_ports)[0]
        prefill_ports = reserve_contiguous_ports(prefill_workers, pd.prefill_port, used_ports)
        decode_ports = reserve_contiguous_ports(decode_workers, pd.decode_port, used_ports)
        prefill_kv_ports = reserve_contiguous_ports(prefill_workers, pd.prefill_kv_port, used_ports)
        decode_kv_ports = reserve_contiguous_ports(decode_workers, pd.decode_kv_port, used_ports)
        pd.prefill_port = prefill_ports[0]
        pd.decode_port = decode_ports[0]
        pd.prefill_kv_port = prefill_kv_ports[0]
        pd.decode_kv_port = decode_kv_ports[0]

        ports = [pd.proxy_port]
        ports.extend(get_contiguous_ports(pd.prefill_port, prefill_workers))
        ports.extend(get_contiguous_ports(pd.decode_port, decode_workers))
        ports.extend(get_contiguous_ports(pd.prefill_kv_port, prefill_workers))
        ports.extend(get_contiguous_ports(pd.decode_kv_port, decode_workers))
        if len(set(ports)) != len(ports):
            raise ValueError("pd_disagg proxy/server/KV port ranges must all be distinct.")

    config.orchestrator.client.base_url = [f"http://{pd.host}:{pd.proxy_port}/v1"]
    logger.info(f"Resolved PD proxy endpoint: {config.orchestrator.client.base_url[0]}")


def build_pd_worker_inference_config(
    base_inference: InferenceConfig,
    *,
    tp: int,
    http_port: int,
    kv_port: int,
    kv_role: Literal["kv_producer", "kv_consumer"],
    kv_rank: int,
    kv_parallel_size: int,
    kv_connector: str,
    kv_send_type: Literal["PUT_ASYNC", "PUT", "GET"],
    all2all_backend_override: All2AllBackend | None,
) -> InferenceConfig:
    worker_inference = copy.deepcopy(base_inference)
    worker_inference.parallel.tp = tp
    worker_inference.parallel.dp = 1
    if all2all_backend_override is not None:
        worker_inference.all2all_backend = all2all_backend_override
    worker_inference.server.port = http_port
    worker_inference.kv_transfer_config = {
        "kv_connector": kv_connector,
        "kv_role": kv_role,
        "kv_rank": kv_rank,
        "kv_parallel_size": kv_parallel_size,
        "kv_port": kv_port,
        "kv_connector_extra_config": {
            "http_port": http_port,
            "send_type": kv_send_type,
        },
    }
    return worker_inference


def start_monitored_process(
    *,
    command: list[str],
    log_path: Path,
    process_key: str,
    process_name: str,
    processes: list[Popen],
    stop_events: dict[str, Event],
    monitor_threads: list[Thread],
    error_queue: list[Exception],
    env: dict[str, str] | None = None,
) -> Popen:
    with open(log_path, "w") as log_file:
        process = Popen(
            command,
            env=env,
            stdout=log_file,
            stderr=log_file,
        )
    processes.append(process)

    stop_event = Event()
    stop_events[process_key] = stop_event
    monitor_thread = Thread(
        target=monitor_process,
        args=(process, stop_event, error_queue, process_name),
        daemon=True,
    )
    monitor_thread.start()
    monitor_threads.append(monitor_thread)
    return process


def start_pd_role_workers(
    *,
    role_name: Literal["prefill", "decode"],
    base_inference: InferenceConfig,
    gpu_groups: list[list[int]],
    tp: int,
    http_ports: list[int],
    kv_ports: list[int],
    kv_role: Literal["kv_producer", "kv_consumer"],
    rank_offset: int,
    kv_parallel_size: int,
    kv_connector: str,
    kv_send_type: Literal["PUT_ASYNC", "PUT", "GET"],
    all2all_backend_override: All2AllBackend | None,
    host: str,
    config_dir: Path,
    log_dir: Path,
    processes: list[Popen],
    stop_events: dict[str, Event],
    monitor_threads: list[Thread],
    error_queue: list[Exception],
    logger,
) -> tuple[list[str], list[str]]:
    urls: list[str] = []
    kv_addrs: list[str] = []

    for worker_idx, (gpu_group, http_port, kv_port) in enumerate(zip(gpu_groups, http_ports, kv_ports, strict=True)):
        worker_inference = build_pd_worker_inference_config(
            base_inference,
            tp=tp,
            http_port=http_port,
            kv_port=kv_port,
            kv_role=kv_role,
            kv_rank=rank_offset + worker_idx,
            kv_parallel_size=kv_parallel_size,
            kv_connector=kv_connector,
            kv_send_type=kv_send_type,
            all2all_backend_override=all2all_backend_override,
        )

        worker_inference_path = config_dir / f"{role_name}_inference_{worker_idx}.toml"
        write_toml_file(worker_inference_path, worker_inference.model_dump(exclude_none=True, mode="json"))
        worker_cmd = ["uv", "run", "inference", "@", worker_inference_path.as_posix()]

        logger.info(f"Starting PD {role_name} inference worker {worker_idx} on GPU(s) {' '.join(map(str, gpu_group))}")
        logger.debug(f"PD {role_name} worker {worker_idx} start command: {' '.join(worker_cmd)}")
        process_key = f"{role_name}_inference_{worker_idx}"
        start_monitored_process(
            command=worker_cmd,
            log_path=log_dir / f"{role_name}_inference_{worker_idx}.stdout",
            process_key=process_key,
            process_name=f"{role_name} inference {worker_idx}",
            processes=processes,
            stop_events=stop_events,
            monitor_threads=monitor_threads,
            error_queue=error_queue,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_group))},
        )

        urls.append(f"http://{host}:{http_port}")
        kv_addrs.append(f"{host}:{kv_port}")

    return urls, kv_addrs


def rl(config: RLConfig):
    # Setup logger
    logger = setup_logger(
        config.log.level or "info",
        log_file=config.output_dir / "logs" / "rl.log" if config.log.file else None,
        json_logging=config.log.json_logging,
    )

    # If dump_config is set, write resolved subconfigs and exit early
    if config.dump_config is not None:
        logger.warning(
            "--dump-config is set. No RL training will be started. Only writing resolved subconfigs to disk."
        )
        write_subconfigs(config, config.dump_config)
        logger.info(f"Dumping resolved subconfigs to {config.dump_config}")
        logger.info(f"  Wrote trainer config to {config.dump_config / 'trainer.toml'}")
        logger.info(f"  Wrote orchestrator config to {config.dump_config / 'orchestrator.toml'}")
        if config.inference is not None:
            logger.info(f"  Wrote inference config to {config.dump_config / 'inference.toml'}")
        if config.teacher_inference is not None:
            logger.info(f"  Wrote teacher inference config to {config.dump_config / 'teacher_inference.toml'}")
        logger.success(f"Config dump complete. Files written to {config.dump_config}")
        logger.warning("To start an RL run, remove --dump-config from your command.")
        return

    start_command = sys.argv
    logger.info("Starting RL run")
    logger.debug(f"RL start command: {' '.join(start_command)}")

    # Check for existing processes on GPUs
    inference_gpu_ids = (
        config.pd_disagg.prefill_gpu_ids + config.pd_disagg.decode_gpu_ids
        if config.pd_disagg.enabled
        else config.inference_gpu_ids
    )
    all_gpu_ids = list(set(inference_gpu_ids + config.trainer_gpu_ids + (config.teacher_gpu_ids or [])))
    check_gpus_available(all_gpu_ids)

    if config.pd_disagg.enabled:
        resolve_pd_runtime_endpoints(config, logger)

    # Validate client port matches inference server port
    if config.inference is not None and not config.orchestrator.client.is_elastic:
        from urllib.parse import urlparse

        base_url = config.orchestrator.client.base_url[0]
        parsed = urlparse(base_url)
        client_port = parsed.port
        expected_port = config.pd_disagg.proxy_port if config.pd_disagg.enabled else config.inference.server.port
        if client_port != expected_port:
            raise ValueError(
                f"orchestrator.client.base_url port ({client_port}) does not match "
                f"inference endpoint port ({expected_port}). "
                f"Update the base_url to use port {expected_port} to match the inference server."
            )

    # Prepare paths to communicate with the trainer
    log_dir = get_log_dir(config.output_dir)
    orch_log_dir = get_log_dir(config.orchestrator.output_dir)
    rollout_dir = get_rollout_dir(config.orchestrator.output_dir)
    broadcast_dir = get_broadcast_dir(config.orchestrator.output_dir)

    # Clean up directories if specified
    if config.clean:
        logger.info("Cleaning checkpoint, logs, weights, broadcast and rollout directories")

        # Cleaning logs (so that streaming logs to terminal works)
        logger.info(f"Cleaning log dir ({log_dir})")
        shutil.rmtree(log_dir, ignore_errors=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Cleaning orchestrator log dir ({orch_log_dir})")
        shutil.rmtree(orch_log_dir, ignore_errors=True)
        orch_log_dir.mkdir(parents=True, exist_ok=True)

        # Cleaning broadcast dir (so that orchestrator does not pre-maturely update weights)
        if not (
            config.ckpt
            and config.ckpt.resume_step
            and config.trainer.weight_broadcast
            and config.trainer.weight_broadcast.type == "filesystem"
        ):
            logger.info(f"Cleaning broadcast directory ({broadcast_dir})")
            shutil.rmtree(broadcast_dir, ignore_errors=True)

        # Cleaning rollouts (so that trainer does not train on old rollouts)
        logger.info(f"Cleaning rollout dir ({rollout_dir})")
        shutil.rmtree(rollout_dir, ignore_errors=True)

    # Write all resolved subconfigs to disk
    config_dir = Path(".pydantic_config") / uuid.uuid4().hex
    write_subconfigs(config, config_dir)

    # Start processes
    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    try:
        # Optionally, start inference process(es)
        if config.inference:
            if config.pd_disagg.enabled:
                pd = config.pd_disagg
                default_tp = config.inference.parallel.tp
                prefill_tp = pd.prefill_tp or default_tp
                decode_tp = pd.decode_tp or default_tp
                prefill_gpu_groups = chunk_gpu_ids(pd.prefill_gpu_ids, prefill_tp)
                decode_gpu_groups = chunk_gpu_ids(pd.decode_gpu_ids, decode_tp)
                prefill_ports = get_contiguous_ports(pd.prefill_port, len(prefill_gpu_groups))
                decode_ports = get_contiguous_ports(pd.decode_port, len(decode_gpu_groups))
                prefill_kv_ports = get_contiguous_ports(pd.prefill_kv_port, len(prefill_gpu_groups))
                decode_kv_ports = get_contiguous_ports(pd.decode_kv_port, len(decode_gpu_groups))
                kv_parallel_size = len(prefill_gpu_groups) + len(decode_gpu_groups)

                prefill_urls, prefill_kv_addrs = start_pd_role_workers(
                    role_name="prefill",
                    base_inference=config.inference,
                    gpu_groups=prefill_gpu_groups,
                    tp=prefill_tp,
                    http_ports=prefill_ports,
                    kv_ports=prefill_kv_ports,
                    kv_role="kv_producer",
                    rank_offset=0,
                    kv_parallel_size=kv_parallel_size,
                    kv_connector=pd.kv_connector,
                    kv_send_type=pd.kv_send_type,
                    all2all_backend_override=pd.prefill_all2all_backend,
                    host=pd.host,
                    config_dir=config_dir,
                    log_dir=log_dir,
                    processes=processes,
                    stop_events=stop_events,
                    monitor_threads=monitor_threads,
                    error_queue=error_queue,
                    logger=logger,
                )
                decode_urls, decode_kv_addrs = start_pd_role_workers(
                    role_name="decode",
                    base_inference=config.inference,
                    gpu_groups=decode_gpu_groups,
                    tp=decode_tp,
                    http_ports=decode_ports,
                    kv_ports=decode_kv_ports,
                    kv_role="kv_consumer",
                    rank_offset=len(prefill_gpu_groups),
                    kv_parallel_size=kv_parallel_size,
                    kv_connector=pd.kv_connector,
                    kv_send_type=pd.kv_send_type,
                    all2all_backend_override=pd.decode_all2all_backend,
                    host=pd.host,
                    config_dir=config_dir,
                    log_dir=log_dir,
                    processes=processes,
                    stop_events=stop_events,
                    monitor_threads=monitor_threads,
                    error_queue=error_queue,
                    logger=logger,
                )

                pd_proxy_cmd = [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "prime_rl.inference.vllm.pd_proxy",
                    "--host",
                    pd.host,
                    "--port",
                    str(pd.proxy_port),
                    "--prefill-urls",
                    ",".join(prefill_urls),
                    "--decode-urls",
                    ",".join(decode_urls),
                    "--prefill-kv-addrs",
                    ",".join(prefill_kv_addrs),
                    "--decode-kv-addrs",
                    ",".join(decode_kv_addrs),
                    "--timeout",
                    str(config.orchestrator.client.timeout),
                ]
                logger.info(
                    f"Starting PD proxy process on {pd.host}:{pd.proxy_port} "
                    f"for {len(prefill_urls)} prefill and {len(decode_urls)} decode workers"
                )
                logger.debug(f"PD proxy start command: {' '.join(pd_proxy_cmd)}")
                start_monitored_process(
                    command=pd_proxy_cmd,
                    log_path=log_dir / "pd_proxy.stdout",
                    process_key="pd_proxy",
                    process_name="pd proxy",
                    processes=processes,
                    stop_events=stop_events,
                    monitor_threads=monitor_threads,
                    error_queue=error_queue,
                )
            else:
                inference_cmd = ["uv", "run", "inference", "@", (config_dir / "inference.toml").as_posix()]
                logger.info(f"Starting inference process on GPU(s) {' '.join(map(str, config.inference_gpu_ids))}")
                logger.debug(f"Inference start command: {' '.join(inference_cmd)}")
                # If we don't log stdout, the server hangs
                with open(log_dir / "inference.stdout", "w") as log_file:
                    inference_process = Popen(
                        inference_cmd,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.inference_gpu_ids))},
                        stdout=log_file,
                        stderr=log_file,
                    )
                processes.append(inference_process)

                # Start monitoring thread
                stop_event = Event()
                stop_events["inference"] = stop_event
                monitor_thread = Thread(
                    target=monitor_process,
                    args=(inference_process, stop_event, error_queue, "inference"),
                    daemon=True,
                )
                monitor_thread.start()
                monitor_threads.append(monitor_thread)
        else:
            logger.warning(
                "No inference config specified, skipping starting inference server. Is your inference server running?"
            )

        # Optionally, start teacher inference process
        if config.teacher_inference:
            if config.teacher_gpu_ids is None or len(config.teacher_gpu_ids) == 0:
                raise ValueError(
                    "teacher_inference is configured but teacher_gpu_ids is not set or is empty. "
                    "Either set teacher_gpu_ids to start a teacher inference server, "
                    "or omit teacher_inference and configure orchestrator.teacher_model to use an existing server."
                )

            teacher_inference_cmd = ["uv", "run", "inference", "@", (config_dir / "teacher_inference.toml").as_posix()]
            logger.info(f"Starting teacher inference process on GPU(s) {' '.join(map(str, config.teacher_gpu_ids))}")
            logger.debug(f"Teacher inference start command: {' '.join(teacher_inference_cmd)}")
            with open(log_dir / "teacher_inference.stdout", "w") as log_file:
                teacher_inference_process = Popen(
                    teacher_inference_cmd,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.teacher_gpu_ids))},
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(teacher_inference_process)

            # Start monitoring thread
            stop_event = Event()
            stop_events["teacher_inference"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(teacher_inference_process, stop_event, error_queue, "teacher_inference"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        elif config.trainer.loss.teacher_tau > 0 or config.orchestrator.teacher_model:
            logger.warning(
                "No teacher_inference config specified, skipping starting teacher inference server. "
                "Is your teacher inference server running? Make sure orchestrator.teacher_model is configured."
            )

        # Start orchestrator process
        orchestrator_cmd = [
            "uv",
            "run",
            "orchestrator",
            "@",
            (config_dir / "orchestrator.toml").as_posix(),
        ]
        logger.info("Starting orchestrator process")
        logger.debug(f"Orchestrator start command: {' '.join(orchestrator_cmd)}")
        with open(log_dir / "orchestrator.stdout", "w") as log_file:
            orchestrator_process = Popen(
                orchestrator_cmd,
                stdout=log_file,
                stderr=log_file,
                env={
                    **os.environ,
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
            )
        processes.append(orchestrator_process)

        # Start monitoring thread
        stop_event = Event()
        stop_events["orchestrator"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(orchestrator_process, stop_event, error_queue, "orchestrator"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # Start training process
        trainer_cmd = [
            "uv",
            "run",
            "env",
            "PYTHONUNBUFFERED=1",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "torchrun",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            f"--rdzv-id={uuid.uuid4().hex}",
            # Pipe all logs to file, and only master rank logs to stdout
            f"--log-dir={config.output_dir / 'torchrun'}",
            "--local-ranks-filter=0",
            "--redirect=3",
            "--tee=3",
            f"--nproc-per-node={len(config.trainer_gpu_ids)}",
            "-m",
            "prime_rl.trainer.rl.train",
            "@",
            (config_dir / "trainer.toml").as_posix(),
        ]
        logger.info(f"Starting trainer process on GPU(s) {' '.join(map(str, config.trainer_gpu_ids))}")
        logger.debug(f"Training start command: {' '.join(trainer_cmd)}")
        with open(log_dir / "trainer.stdout", "w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.trainer_gpu_ids)),
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
                stdout=log_file,
                stderr=log_file,
            )
        processes.append(trainer_process)

        # Start monitoring thread
        stop_event = Event()
        stop_events["trainer"] = stop_event
        monitor_thread = Thread(
            target=monitor_process, args=(trainer_process, stop_event, error_queue, "trainer"), daemon=True
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # Monitor all processes for failures
        logger.success("Startup complete. Showing trainer logs...")

        tail_process = Popen(["tail", "-F", log_dir / "trainer.stdout"])
        processes.append(tail_process)

        # Check for errors from monitor threads
        while not (stop_events["orchestrator"].is_set() and stop_events["trainer"].is_set()):
            if error_queue:
                error = error_queue[0]
                logger.error(f"Error: {error}")
                logger.error("Terminating all processes...")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                sys.exit(1)

            # Small delay to avoid busy waiting
            time.sleep(1)

        # Check if any critical process failed
        if orchestrator_process.returncode != 0:
            logger.error(f"Orchestrator failed with exit code {orchestrator_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        if trainer_process.returncode != 0:
            logger.error(f"Trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        logger.success("RL training finished!")

        # Cleanup threads and processes
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise


def main():
    rl(parse_argv(RLConfig))


if __name__ == "__main__":
    main()

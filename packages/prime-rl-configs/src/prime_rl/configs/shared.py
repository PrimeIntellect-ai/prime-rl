import os
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.utils.config import BaseConfig


class SlurmConfig(BaseConfig):
    job_name: str = "prime-rl"
    """SLURM job name."""

    project_dir: Path = Path(".")
    """Path to the project root, used to source .env, activate .venv, and run uv sync."""

    template_path: Path | None = None
    """SLURM template file. If None, uses the bundled single-node or multi-node template."""

    partition: str = "cluster"
    """SLURM partition (#SBATCH --partition)."""

    nodelist: str | None = None
    """Comma-separated list of specific nodes to run on (#SBATCH --nodelist)."""

    exclude: str | None = None
    """Comma-separated list of nodes to exclude (#SBATCH --exclude)."""

    account: str | None = None
    """SLURM account to charge (#SBATCH --account)."""

    time: str | None = None
    """Maximum wall time, e.g. '24:00:00' or '7-00:00:00' (#SBATCH --time)."""

    pre_run_command: str | None = None
    """Shell command to run on the head node after cd, .env sourcing, and venv activation. Useful for cleanup like ``sudo pkill -f vllm``; wrap with ``srun bash -c '...'`` to fan out to all nodes."""

    @property
    def template_vars(self) -> dict:
        """Common template variables for all SLURM templates."""
        return {
            "job_name": self.job_name,
            "project_dir": self.project_dir,
            "partition": self.partition,
            "nodelist": self.nodelist,
            "exclude": self.exclude,
            "account": self.account,
            "time": self.time,
            "pre_run_command": self.pre_run_command,
        }

    @model_validator(mode="after")
    def resolve_project_dir(self):
        self.project_dir = self.project_dir.resolve()
        return self


ServerType = Literal["vllm", "openai"]


class VLMConfig(BaseConfig):
    vision_encoder_attr: str
    """Dotted attribute path to the vision encoder module (e.g. ``model.visual``)."""

    language_model_attr: str
    """Dotted attribute path to the language model module (e.g. ``model.language_model``)."""

    freeze_vision_encoder: bool = True
    """Freeze the vision encoder. When False, it is trainable and FSDP-sharded per-block. No effect with LoRA (LoRA freezes all non-adapter parameters)."""


class BaseModelConfig(BaseConfig):
    name: str = "Qwen/Qwen3-0.6B"
    """HF model name or local path."""

    trust_remote_code: bool = False
    """Trust remote code when initializing the tokenizer."""

    vlm: "VLMConfig | None" = None
    """VLM configuration. Setting this enables vision-language model support."""

    @property
    def is_vlm(self) -> bool:
        return self.vlm is not None


class RendererConfig(BaseConfig):
    name: str = "auto"
    """Renderer used for chat-template tokenization. One of: ``auto`` (detect from tokenizer), ``qwen3``, ``qwen3_vl``, ``qwen3.5``, ``glm5``, ``glm4.5``, ``minimax-m2``, ``deepseek_v3``, ``kimi_k2``, ``kimi_k25``, ``nemotron3``, ``gpt_oss``, ``default``."""

    tool_parser: str | None = None
    """Tool parser from ``renderers.parsers``. Only consumed by DefaultRenderer; model-specific renderers bake their own parsing in. Options: ``qwen3``, ``qwen3.5``, ``glm``, ``deepseek_v3``."""

    reasoning_parser: str | None = None
    """Reasoning parser from ``renderers.parsers``. Only consumed by DefaultRenderer. Options: ``think``."""

    pool_size: int | None = Field(None, ge=1)
    """Number of renderer slots shared across concurrent rollouts. Bump for long multi-turn prompts where client-side jinja tokenization serializes."""

    preserve_all_thinking: bool = False
    """Re-emit every past-assistant turn's ``reasoning_content`` between ``<think>``/``</think>`` (or the model's equivalent), even when the chat template would drop it. Strict superset of preserve_thinking_between_tool_calls."""

    preserve_thinking_between_tool_calls: bool = False
    """Preserve past-assistant ``reasoning_content`` only inside the current tool cycle — the contiguous assistant→tool→…→assistant block after the most recent user message, when that block contains at least one tool response. A new user turn closes the block."""


class ElasticConfig(BaseConfig):
    hostname: str
    """DNS hostname that resolves to inference server IPs."""

    port: int = 8000
    """Port that inference servers listen on."""

    sync_interval: float = 5.0
    """Seconds between server discovery checks."""


class ClientConfig(BaseConfig):
    timeout: int = 1200
    """Request timeout in seconds."""

    connect_timeout: float = 30.0
    """TCP connect timeout in seconds for inference API requests."""

    wait_for_ready_timeout: int = 1800
    """Seconds to wait at startup for the inference pool to become ready. Applies to both the static health check and elastic DNS-based discovery."""

    base_url: list[str] = ["http://localhost:8000/v1"]
    """Base URLs for the OpenAI API. With more than one URL, the client round-robins (chat) completion requests across all servers. Ignored when ``elastic`` is set."""

    api_key_var: str = "VLLM_API_KEY"
    """Environment variable name containing the API key, resolved via ``os.getenv``. Can be any string when the server is not protected by an API key; the same key is used for every URL."""

    headers: dict[str, str] = {}
    """Static headers sent with every request."""

    headers_from_env: dict[str, str] = {}
    """Maps HTTP header names to environment variable names; each entry is resolved via ``os.getenv`` and merged into request headers. e.g. ``{"X-Prime-Team-ID": "PRIME_TEAM_ID"}``."""

    extra_headers_from_state: dict[str, str] = {}
    """Maps HTTP header names to rollout-state field names. The header value is read from the rollout state dict on every request. e.g. ``{"X-Session-ID": "example_id"}`` enables sticky routing at the inference router."""

    skip_model_check: bool = False
    """Skip checking that the model is available in the inference pool. Useful for external APIs or keys that do not expose ``/models``."""

    dp_rank_count: int = Field(1, ge=1)
    """Number of data-parallel ranks behind each base URL. When > 1, each URL is expanded into ``dp_rank_count`` logical clients pinned via the ``X-data-parallel-rank`` header, so every request within a rollout hits the same DP engine and reuses KV cache. Auto-set from the inference config when using the RL entrypoint."""

    admin_base_url: list[str] | None = None
    """Separate base URLs for admin operations (weight updates, health checks). When set, admin clients bypass routers and hit each server directly — used in disaggregated P/D deployments where the router must not handle admin traffic."""

    elastic: ElasticConfig | None = None
    """Elastic inference pool config for DNS-based service discovery. When set, ``base_url`` is ignored and inference servers are discovered dynamically via DNS."""

    router_url: str | None = None
    """vllm-router URL for load-aware inference routing. With elastic mode, inference requests go through the router while admin ops still hit discovered pods directly."""

    @property
    def is_elastic(self) -> bool:
        """Check if elastic mode is enabled."""
        return self.elastic is not None


class LogConfig(BaseConfig):
    level: str = Field(default_factory=lambda: os.environ.get("PRIME_LOG_LEVEL", "info"))
    """Log level for the process. Defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``."""

    vf_level: str = Field(default_factory=lambda: os.environ.get("PRIME_VF_LOG_LEVEL", "info"))
    """Log level for the verifiers package. Defaults to ``$PRIME_VF_LOG_LEVEL`` if set, else ``info``."""

    json_logging: bool = False
    """Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.)."""

    log_data: bool = False
    """Log the first data sample at startup."""


class TrainerLogConfig(LogConfig):
    ranks_filter: list[int] = [0]
    """Trainer ranks to show in console output. Passed to ``torchrun --local-ranks-filter``."""


class LogExtrasConfig(BaseConfig):
    samples: bool = True
    """Log prompt/response samples."""

    distributions: bool = True
    """Log distributions (rewards, advantages, etc.)."""

    interval: int = Field(10, ge=1)
    """Step interval between extras logs."""

    sample_ratio: float | None = Field(None, ge=0.0, le=1.0)
    """Fraction of rollouts to log per step. The effective cap is ``len(rollouts) * sample_ratio``; 1.0 = all, 0.5 = half, 0.0 = none."""


class WandbConfig(BaseConfig):
    # Shared configs (May be overwritten by WandbConfig from `rl.py`)
    project: str = "prime-rl"
    """W&B project to log to."""

    entity: str | None = None
    """W&B entity to log to."""

    name: str | None = None
    """W&B run name."""

    group: str | None = None
    """W&B group."""

    tags: list[str] | None = None
    """W&B tags attached to the run."""

    offline: bool = False
    """Run W&B in offline mode."""


class WandbWithExtrasConfig(WandbConfig):
    log_extras: LogExtrasConfig | None = LogExtrasConfig()
    """Extras logging configuration. If None, no extras are logged."""


class PrimeMonitorConfig(BaseConfig):
    base_url: str = "https://api.primeintellect.ai/api/v1/rft"
    """Base URL for the Prime Intellect monitoring API."""

    api_key_var: str = "PRIME_API_KEY"
    """Environment variable name containing the Prime Intellect API key, resolved via ``os.getenv``."""

    log_extras: LogExtrasConfig | None = LogExtrasConfig()
    """Extras logging configuration. If None, no extras are logged."""

    run_name: str | None = None
    """Run name shown on the platform. Defaults to the W&B run name when set, otherwise the platform auto-generates one."""

    team_id: str | None = None
    """Team ID to associate the run with."""

    frontend_url: str | None = None
    """Frontend base URL used for the dashboard link printed after registration. Defaults to the Prime CLI frontend URL when unset."""


class HeartbeatConfig(BaseConfig):
    url: str
    """URL to send the heartbeat to."""


class MetricsServerConfig(BaseConfig):
    port: int = Field(8000, ge=1, le=65535)
    """Port to expose metrics and health endpoints on."""

    host: str = "0.0.0.0"
    """Host to bind the server to."""


class BaseTransportConfig(BaseConfig):
    pass


class FileSystemTransportConfig(BaseTransportConfig):
    type: Literal["filesystem"] = "filesystem"


class ZMQTransportConfig(BaseTransportConfig):
    type: Literal["zmq"] = "zmq"

    host: str = "localhost"
    """Host address for ZMQ transport."""

    port: int = 5555
    """Base port for ZMQ transport."""

    hwm: int = 10
    """High-water mark (max in-flight messages per ZMQ socket)."""


TransportConfig: TypeAlias = Annotated[FileSystemTransportConfig | ZMQTransportConfig, Field(discriminator="type")]

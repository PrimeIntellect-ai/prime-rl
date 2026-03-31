from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from prime_rl.utils.config import BaseConfig


class SlurmConfig(BaseConfig):
    """Configures SLURM scheduling."""

    job_name: Annotated[str, Field(description="The SLURM job name.")] = "prime-rl"

    project_dir: Annotated[
        Path,
        Field(description="Path to the project root. Used to source .env, activate .venv, and run uv sync."),
    ] = Path(".")

    template_path: Annotated[
        Path | None,
        Field(
            description="The path to the SLURM template file. If None, will use the default single-node/multi-node template."
        ),
    ] = None

    partition: Annotated[
        str, Field(description="The SLURM partition to use. Will be passed as #SBATCH --partition.")
    ] = "cluster"

    nodelist: Annotated[
        str | None,
        Field(description="Comma-separated list of specific nodes to run on. Passed as #SBATCH --nodelist."),
    ] = None

    exclude: Annotated[
        str | None,
        Field(description="Comma-separated list of nodes to exclude. Passed as #SBATCH --exclude."),
    ] = None

    account: Annotated[
        str | None,
        Field(description="SLURM account to charge. Passed as #SBATCH --account."),
    ] = None

    time: Annotated[
        str | None,
        Field(description="Maximum wall time (e.g. '24:00:00', '7-00:00:00'). Passed as #SBATCH --time."),
    ] = None

    pre_run_command: Annotated[
        str | None,
        Field(
            description="Shell command to run on the head node before starting the job. "
            "Runs after cd into project dir, .env sourcing, and venv activation. "
            "Useful for cleanup routines like 'sudo pkill -f vllm'. "
            "To run on all nodes, wrap with srun: 'srun bash -c \"pkill -f vllm || true\"'.",
        ),
    ] = None

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
    """Configures vision-language model support.

    Presence of this config enables VLM mode. You must specify where the
    vision encoder and language model live on the model object.

    Usage:
        [model.vlm]
        vision_encoder_attr = "model.visual"
        language_model_attr = "model.language_model"
    """

    vision_encoder_attr: Annotated[
        str,
        Field(description="Dotted attribute path to the vision encoder module (e.g. 'model.visual')."),
    ]

    language_model_attr: Annotated[
        str,
        Field(description="Dotted attribute path to the language model module (e.g. 'model.language_model')."),
    ]

    freeze_vision_encoder: Annotated[
        bool,
        Field(
            description="Whether to freeze the vision encoder. When False, the vision encoder is trainable "
            "and FSDP-sharded per-block. Has no effect with LoRA (LoRA freezes all non-adapter parameters).",
        ),
    ] = True


class BaseModelConfig(BaseConfig):
    """Configures the model."""

    name: Annotated[str, Field(description="Name or path of the HF model to use.")] = "Qwen/Qwen3-0.6B"

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code for tokenizer initialization.",
        ),
    ] = False

    vlm: Annotated[
        "VLMConfig | None",
        Field(
            description="VLM configuration. Set this to enable vision-language model support.",
        ),
    ] = None


class ElasticConfig(BaseConfig):
    """Configures elastic inference pool with DNS-based service discovery.

    Works with any DNS hostname that resolves to multiple IP addresses.
    """

    hostname: Annotated[
        str,
        Field(
            description="DNS hostname that resolves to inference server IPs.",
        ),
    ]

    port: Annotated[
        int,
        Field(
            description="Port that inference servers listen on.",
        ),
    ] = 8000

    sync_interval: Annotated[
        float,
        Field(
            description="Interval in seconds between server discovery checks.",
        ),
    ] = 5.0


class ClientConfig(BaseConfig):
    """Configures the OAI client.

    Supports two modes:
    - Static mode (default): Uses fixed base_url list
    - Elastic mode: Uses DNS-based service discovery via hostname

    If elastic config is provided, base_url is ignored and servers are discovered dynamically.
    """

    timeout: Annotated[
        int,
        Field(
            description="Timeout in seconds. By default, it is set to 1200 seconds.",
        ),
    ] = 1200

    connect_timeout: Annotated[
        float,
        Field(
            description="TCP connect timeout in seconds for inference API requests.",
        ),
    ] = 30.0

    base_url: Annotated[
        list[str],
        Field(
            description="Base URLs to use for the OpenAI API. By default, it is set to a single server on localhost at port 8000 which matches the default local vLLM server configuration. If you specify more than one URL, the client will round-robin (chat) completion requests across all servers. Ignored if elastic config is provided.",
        ),
    ] = ["http://localhost:8000/v1"]

    api_key_var: Annotated[
        str,
        Field(
            description="Name of environment variable containing the API key to use for the inference API. Will parse using `os.getenv(client_config.api_key_var)`. Can be set to an arbitrary string if the inference server is not protected by an API key. If multiple URLs are specified, the same API key will be used for all servers.",
        ),
    ] = "VLLM_API_KEY"

    headers: Annotated[
        dict[str, str],
        Field(
            description="Headers to use for the OpenAI API. By default, it is set to an empty dictionary.",
        ),
    ] = {}

    extra_headers_from_state: Annotated[
        dict[str, str],
        Field(
            description="Maps HTTP header names to state field names. For each inference request, "
            "the header value is dynamically read from the rollout state dict. "
            'e.g. {"X-Session-ID": "example_id"} enables sticky routing at the inference router.',
        ),
    ] = {}

    skip_model_check: Annotated[
        bool,
        Field(
            description="Whether to skip checking if the model is available in the inference pool. Useful for external APIs or API Keys that don't support the /models endpoint.",
        ),
    ] = False

    dp_rank_count: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Number of data-parallel ranks behind each base URL. When > 1, "
                "each URL is expanded into dp_rank_count logical clients, each "
                "pinned to a specific DP rank via the X-data-parallel-rank header. "
                "This ensures all requests within a multi-turn rollout hit the same "
                "DP engine, maximizing KV cache reuse. Auto-set from "
                "inference.data_parallel_size_local (or inference.parallel.dp) "
                "when using the RL entrypoint."
            ),
        ),
    ] = 1

    admin_base_url: Annotated[
        list[str] | None,
        Field(
            description="Separate base URLs for admin operations (weight updates, health checks). "
            "When set, admin clients use these URLs instead of base_url, allowing weight "
            "updates to bypass routers and hit each server directly. Used in disaggregated "
            "P/D deployments where the inference router should not handle admin traffic.",
        ),
    ] = None

    elastic: Annotated[
        ElasticConfig | None,
        Field(
            description="Elastic inference pool configuration for DNS-based service discovery. If provided, base_url is ignored and inference servers are discovered dynamically via DNS.",
        ),
    ] = None

    router_url: Annotated[
        str | None,
        Field(
            description="URL of a vllm-router for load-aware inference routing. When set with elastic mode, inference requests go through the router while admin operations (weight updates, LoRA loading) still go directly to discovered pods.",
        ),
    ] = None

    @property
    def is_elastic(self) -> bool:
        """Check if elastic mode is enabled."""
        return self.elastic is not None


class LogConfig(BaseConfig):
    """Configures the logger."""

    level: Annotated[
        str,
        Field(description="Logging level for the process. Will determine the logging verbosity and format."),
    ] = "info"

    vf_level: Annotated[
        str,
        Field(description="Logging level for the verifiers package. Will determine the logging verbosity and format."),
    ] = "info"

    json_logging: Annotated[
        bool,
        Field(
            description="Emit JSON logs (newline-delimited) for log aggregation (Loki, Grafana, etc.).",
        ),
    ] = False

    log_data: Annotated[
        bool,
        Field(
            description="Whether to log the first data sample to the logger.",
        ),
    ] = False


class TrainerLogConfig(LogConfig):
    """Trainer-specific log config."""

    ranks_filter: Annotated[
        list[int],
        Field(description="Which trainer ranks to show in console output. Passed to torchrun's --local-ranks-filter."),
    ] = [0]


class LogExtrasConfig(BaseConfig):
    """Configures extra logging for monitoring platforms."""

    samples: Annotated[
        bool,
        Field(
            description="Whether to log prompt/response samples.",
        ),
    ] = True

    distributions: Annotated[
        bool,
        Field(
            description="Whether to log distributions (like rewards, advantages, etc.).",
        ),
    ] = True

    interval: Annotated[
        int,
        Field(
            ge=1,
            description="Step interval at which to log extras.",
        ),
    ] = 10

    sample_ratio: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=1.0,
            description="Fraction of rollouts to log per step (0.0–1.0). "
            "When set, the effective sample cap is len(rollouts) * sample_ratio. "
            "1.0 = all rollouts, 0.5 = half, 0.0 = none. "
            "None (default)",
        ),
    ] = None


class WandbConfig(BaseConfig):
    """Configures logging to Weights and Biases."""

    # Shared configs (May be overwritten by WandbConfig from `rl.py`)
    project: Annotated[str, Field(description="The W&B project to log to.")] = "prime-rl"

    name: Annotated[
        str | None,
        Field(
            description="The W&B name to to use for logging.",
        ),
    ] = None

    offline: Annotated[bool, Field(description="Whether to run W&B in offline mode.")] = False


class WandbWithExtrasConfig(WandbConfig):
    """Configures logging to Weights and Biases with extras."""

    log_extras: Annotated[
        LogExtrasConfig | None,
        Field(
            description="Configuration for logging extras. If None, no extras are logged.",
        ),
    ] = LogExtrasConfig()


class PrimeMonitorConfig(BaseConfig):
    """Configures logging to Prime Intellect API."""

    base_url: Annotated[
        str,
        Field(
            description="The base URL for Prime Intellect monitoring API.",
        ),
    ] = "https://api.primeintellect.ai/api/v1/rft"

    api_key_var: Annotated[
        str,
        Field(
            description="Name of environment variable containing the API key for Prime Intellect API. Will parse using `os.getenv(config.api_key_var)`.",
        ),
    ] = "PRIME_API_KEY"

    log_extras: Annotated[
        LogExtrasConfig | None,
        Field(
            description="Configuration for logging extras. If None, no extras are logged.",
        ),
    ] = LogExtrasConfig()

    run_name: Annotated[
        str | None,
        Field(
            description="Name for the run shown on the platform. Defaults to the W&B run name if set, otherwise auto-generated by the platform.",
        ),
    ] = None

    team_id: Annotated[
        str | None,
        Field(
            description="Team ID to associate the run with.",
        ),
    ] = None

    frontend_url: Annotated[
        str | None,
        Field(
            description="Frontend base URL used for the dashboard link shown after registration. Defaults to the Prime CLI frontend URL when unset.",
        ),
    ] = None


class HeartbeatConfig(BaseConfig):
    """Configures the heartbeat for BetterStack."""

    url: Annotated[str, Field(description="The URL to send the heartbeat to.")]


class MetricsServerConfig(BaseConfig):
    """Configures the Prometheus metrics server for trainer observability."""

    port: Annotated[
        int,
        Field(
            ge=1,
            le=65535,
            description="Port to expose metrics and health endpoints. Defaults to 8000.",
        ),
    ] = 8000

    host: Annotated[
        str,
        Field(
            description="Host to bind the server to. Defaults to 0.0.0.0.",
        ),
    ] = "0.0.0.0"


class BaseTransportConfig(BaseModel):
    """Base configuration for transport."""

    pass


class FileSystemTransportConfig(BaseTransportConfig):
    """Configures filesystem-based transport for training examples."""

    type: Literal["filesystem"] = "filesystem"


class ZMQTransportConfig(BaseTransportConfig):
    """Configures ZMQ-based transport for training examples."""

    type: Literal["zmq"] = "zmq"
    host: Annotated[str, Field(description="The host address for ZMQ transport.")] = "localhost"
    port: Annotated[int, Field(description="The base port for ZMQ transport.")] = 5555
    hwm: Annotated[int, Field(description="High water mark (max messages in queue) for ZMQ sockets.")] = 10


TransportConfig: TypeAlias = Annotated[FileSystemTransportConfig | ZMQTransportConfig, Field(discriminator="type")]

from typing import Annotated, Literal

from pydantic import Field

from prime_rl.utils.pydantic_config import BaseConfig


class ModelConfig(BaseConfig):
    """Configures the model."""

    name: Annotated[str, Field(description="Name or path of the HF model to use.")] = "Qwen/Qwen3-0.6B"

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code for tokenizer initialization.",
        ),
    ] = False


ServerType = Literal["vllm", "openai"]


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

    base_url: Annotated[
        list[str],
        Field(
            description="Base URLs to use for the OpenAI API. By default, it is set to a single server on localhost at port 8000 which matches the default local vLLM server configuration. If you specify more than one URL, the client will round-robin (chat) completion requests across all servers. Ignored if elastic config is provided.",
        ),
    ] = ["http://localhost:8000/v1"]

    api_key_var: Annotated[
        str,
        Field(
            description="Name of environment variable containing the API key to use for the OpenAI API. Will parse using `os.getenv(client_config.api_key_var)`. Can be set to an arbitrary string if the inference server is not protected by an API key. If multiple URLs are specified, the same API key will be used for all servers.",
        ),
    ] = "OPENAI_API_KEY"

    headers: Annotated[
        dict[str, str],
        Field(
            description="Headers to use for the OpenAI API. By default, it is set to an empty dictionary.",
        ),
    ] = {}

    skip_model_check: Annotated[
        bool,
        Field(
            description="Whether to skip checking if the model is available in the inference pool. Useful for external APIs or API Keys that don't support the /models endpoint.",
        ),
    ] = False

    elastic: Annotated[
        ElasticConfig | None,
        Field(
            description="Elastic inference pool configuration for DNS-based service discovery. If provided, base_url is ignored and inference servers are discovered dynamically via DNS.",
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
    ] = "warn"

    file: Annotated[
        bool,
        Field(
            description="Whether to log to a file. If True, will log to a file in the output directory.",
        ),
    ] = True

    env_worker_logs: Annotated[
        bool,
        Field(
            description="Whether env workers log to files. If True, workers write to logs/env_workers/{env_name}.log.",
        ),
    ] = False

    log_data: Annotated[
        bool,
        Field(
            description="Whether to log the first data sample to the logger.",
        ),
    ] = False

    json_logging: Annotated[
        bool,
        Field(
            description="Emit JSON logs (newline-delimited) for log aggregation (Loki, Grafana, etc.).",
        ),
    ] = False


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

    # Individual configs (can only be specified on trainer or orchestrator)
    id: Annotated[
        str | None,
        Field(
            description="The W&B run ID to log to. If None, a random ID will be generated. If you want to resume a run, you can set the ID to the run ID you want to resume.",
        ),
    ] = None


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
    ] = "https://api.primeintellect.ai/api/internal/rft"

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

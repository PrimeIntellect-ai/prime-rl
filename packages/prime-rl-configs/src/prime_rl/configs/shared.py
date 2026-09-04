import ipaddress
import os
import re
from pathlib import Path
from typing import Annotated, Literal, TypeAlias
from urllib.parse import urlsplit

from pydantic import AfterValidator, Field, field_validator, model_validator

from prime_rl.utils.config import BaseConfig

# Launcher-managed env vars that a component's `env_vars` must not set: GPU partitioning
# and the single shared W&B run. The launcher always sets these last, so allowing them in
# `env_vars` would be a silent no-op (or, on multi-node, a footgun) — reject them instead.
PROTECTED_ENV_VARS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "PRL_RUN_ID",
        "PRL_RUN_NAME",
        "WANDB_RUN_ID",
        "WANDB_SHARED_MODE",
        "WANDB_SHARED_LABEL",
        "WANDB_SHARED_PRIMARY",
        "WANDB_SHARED_FINISHER",
    }
)


def reject_protected_env_vars(env_vars: dict[str, str]) -> dict[str, str]:
    clobbered = sorted(PROTECTED_ENV_VARS & env_vars.keys())
    if clobbered:
        raise ValueError(
            f"env_vars cannot set launcher-managed vars {clobbered} — set by the launcher, not overridable"
        )
    return env_vars


EnvVars: TypeAlias = Annotated[dict[str, str], AfterValidator(reject_protected_env_vars)]
"""A per-component `env_vars` mapping, validated to not clobber `PROTECTED_ENV_VARS`."""


def validate_http_url(value: str, *, field_name: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} must be an http(s) URL with a host")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError(f"{field_name} must contain a valid port") from error
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not contain a query or fragment")
    return value.rstrip("/")


def normalize_admin_host_allowlist_entry(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("admin host allowlist entries must not be empty")
    try:
        if "/" in value:
            return str(ipaddress.ip_network(value, strict=False))
        return str(ipaddress.ip_address(value))
    except ValueError as error:
        if "/" in value:
            raise ValueError(f"invalid admin host CIDR {value!r}") from error

    hostname = value.rstrip(".").lower()
    if len(hostname) > 253 or not all(
        label and len(label) <= 63 and re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
        for label in hostname.split(".")
    ):
        raise ValueError(f"invalid exact admin DNS host {value!r}")
    return hostname


def canonical_http_origin(value: str, *, field_name: str) -> str:
    normalized_url = validate_http_url(value, field_name=field_name)
    parsed = urlsplit(normalized_url)
    assert parsed.hostname is not None
    hostname = normalize_admin_host_allowlist_entry(parsed.hostname)
    formatted_hostname = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 80 if parsed.scheme == "http" else 443
    return f"{parsed.scheme}://{formatted_hostname}:{parsed.port or default_port}"


def normalize_admin_origin_allowlist_entry(value: str) -> str:
    normalized_url = validate_http_url(value, field_name="dynamo.admin_origin_allowlist")
    if urlsplit(normalized_url).path not in ("", "/"):
        raise ValueError("dynamo.admin_origin_allowlist entries must be origins without paths")
    return canonical_http_origin(normalized_url, field_name="dynamo.admin_origin_allowlist")


def is_secure_or_loopback_url(value: str) -> bool:
    parsed = urlsplit(value)
    if parsed.scheme == "https":
        return True
    hostname = parsed.hostname
    if hostname is None:
        return False
    if hostname.rstrip(".").lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_header_environment_mapping(values: dict[str, str]) -> dict[str, str]:
    if len(values) > 128:
        raise ValueError("header environment mappings must not contain more than 128 entries")
    for header_name, environment_name in values.items():
        if not header_name or len(header_name) > 256 or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", header_name):
            raise ValueError(f"invalid HTTP header name {header_name!r}")
        if (
            not environment_name
            or len(environment_name) > 256
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", environment_name)
        ):
            raise ValueError(f"invalid environment variable name {environment_name!r}")
    normalized_names = [name.lower() for name in values]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("header environment mappings must not contain case-insensitive duplicate names")
    return dict(values)


class BaseWeightBroadcastConfig(BaseConfig):
    timeout: int = 1200
    """Timeout in seconds for the broadcast handshake and transfer. The trainer
    fails the run when no consumer acknowledges an offered version in time."""


class RunConfig(BaseConfig):
    name: str | None = None
    """Run name. Auto-generated as ``<envs>--<model>--<short-id>`` when unset, so every launch gets a fresh, readable run directory; set an explicit name (e.g. an experiment name) to get a predictable run directory, which is also required to resume a previous run. Unless set explicitly, the W&B run name and the Prime platform run name inherit it."""

    dir: str | None = None
    """Run directory name — the run writes all its artifacts to ``output_dir / dir``. Defaults to ``run.name``; set it only when the directory should differ from the display name."""


class ResumeConfig(BaseConfig):
    """Resume the run from a checkpoint. A bare ``--resume`` (or empty ``[resume]`` block)
    resumes from the latest checkpoint."""

    step: int | None = Field(None, ge=1)
    """Checkpoint step to resume from. None resumes from the latest checkpoint."""

    dir: Path | None = None
    """External checkpoint step directory to resume from (e.g. ``other/run/checkpoints/step_50``) — forks another run's checkpoint into this run. Mutually exclusive with ``step``."""

    @model_validator(mode="after")
    def validate_step_xor_dir(self):
        if self.step is not None and self.dir is not None:
            raise ValueError(
                "resume.step and resume.dir are mutually exclusive — the step is taken from the directory name"
            )
        if self.dir is not None and not re.fullmatch(r"step_\d+", self.dir.name):
            raise ValueError(f"resume.dir must point at a checkpoint step directory (`.../step_<N>`), got '{self.dir}'")
        return self

    @property
    def dir_step(self) -> int:
        """The step encoded in ``dir``'s name (validated to exist)."""
        assert self.dir is not None
        return int(self.dir.name.removeprefix("step_"))


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

    launch_modelexpress: bool = True
    """Start a job-scoped ModelExpress service for NIXL weight transfer."""

    cleanup_grace_period: int = Field(3600, ge=0)
    """Seconds to wait before tearing down a multi-node RL job that hit a non-zero exit, letting in-flight checkpoints flush. Set to 0 to tear down immediately."""

    shared_fs: bool = True
    """Whether the project filesystem (including the venv) is shared across nodes (e.g. NFS). When True, a single ``uv sync`` on the batch node suffices. Set to False when the venv is node-local (e.g. ``UV_PROJECT_ENVIRONMENT`` on ``/tmp``) so ``uv sync`` runs on every node via srun."""

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
            "cleanup_grace_period": self.cleanup_grace_period,
            "shared_fs": self.shared_fs,
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
    """Freeze the vision encoder parameters during training."""


class BaseModelConfig(BaseConfig):
    name: str = "Qwen/Qwen3-0.6B"
    """HF model name or local path."""

    trust_remote_code: bool = False
    """Trust remote code when initializing the tokenizer."""

    vlm: "VLMConfig | None" = None
    """VLM configuration. Setting this enables vision-language model support."""


class DynamoConfig(BaseConfig):
    discovery_url: str = Field(min_length=1, max_length=2048)
    """Trusted Dynamo frontend URL used to discover the inference workers pinned for RL control."""

    expected_namespace: str = Field(min_length=1, max_length=256)
    """Exact Dynamo namespace expected in both the discovery snapshot and selected workers."""

    admin_host_allowlist: tuple[str, ...] = Field(min_length=1, max_length=128)
    """Exact DNS/IP hosts and IP CIDRs permitted for discovered direct admin endpoints."""

    admin_origin_allowlist: tuple[str, ...] | None = Field(None, min_length=1, max_length=128)
    """Optional exact scheme, host, and effective port origins permitted for direct admin endpoints."""

    api_key_var: str | None = Field(None, min_length=1, max_length=256)
    """Environment variable containing a discovery-only bearer token."""

    headers_from_env: dict[str, str] = Field(default_factory=dict, max_length=128)
    """Discovery header names mapped to environment variables."""

    admin_api_key_var: str | None = Field(None, min_length=1, max_length=256)
    """Environment variable containing a bearer token used only for discovered admin endpoints."""

    admin_headers_from_env: dict[str, str] = Field(default_factory=dict, max_length=128)
    """Direct admin header names mapped to environment variables."""

    @field_validator("discovery_url")
    @classmethod
    def validate_discovery_url(cls, value: str) -> str:
        return validate_http_url(value, field_name="dynamo.discovery_url")

    @field_validator("admin_host_allowlist")
    @classmethod
    def validate_admin_host_allowlist(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(normalize_admin_host_allowlist_entry(value) for value in values))

    @field_validator("admin_origin_allowlist")
    @classmethod
    def validate_admin_origin_allowlist(cls, values: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if values is None:
            return None
        return tuple(dict.fromkeys(normalize_admin_origin_allowlist_entry(value) for value in values))

    @field_validator("headers_from_env", "admin_headers_from_env")
    @classmethod
    def validate_header_environment_mapping(cls, values: dict[str, str]) -> dict[str, str]:
        return _validate_header_environment_mapping(values)

class ClientConfig(BaseConfig):
    wait_for_ready_timeout: int = 1800
    """Seconds to wait at startup for the inference pool to become ready."""

    base_url: str = "http://localhost:8000/v1"
    """Base URL for the OpenAI API. For multi-replica deployments, point this at a router in front of the replicas."""

    api_key_var: str = "VLLM_API_KEY"
    """Environment variable name containing the API key, resolved via ``os.getenv``. Can be any string when the server is not protected by an API key; the same key is used for every URL."""

    headers: dict[str, str] = {}
    """Static headers sent with every request."""

    headers_from_env: dict[str, str] = {}
    """Maps HTTP header names to environment variable names; each entry is resolved via ``os.getenv`` and merged into request headers. e.g. ``{"X-Prime-Team-ID": "PRIME_TEAM_ID"}``."""

    skip_model_check: bool = False
    """Skip checking that the model is available in the inference pool. Useful for external APIs or keys that do not expose ``/models``."""

    admin_base_url: list[str] | None = None
    """Separate base URLs for admin operations (weight updates, health checks). When set, admin clients bypass routers and hit each server directly — used in multi-replica or disaggregated P/D deployments where the router must not handle admin traffic."""

    dynamo: DynamoConfig | None = None
    """Dynamo RL worker-discovery configuration."""


class LogConfig(BaseConfig):
    level: str = Field(default_factory=lambda: os.environ.get("PRIME_LOG_LEVEL", "info"))
    """Log level for the process. Defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``."""

    vf_level: str = Field(default_factory=lambda: os.environ.get("PRIME_VF_LOG_LEVEL", "info"))
    """Log level for the verifiers package. Defaults to ``$PRIME_VF_LOG_LEVEL`` if set, else ``info``."""

    json_logging: bool = False
    """Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.)."""

    log_data: bool = False
    """Log the first data sample at startup."""

    interval: float = Field(10.0, gt=0)
    """Interval (seconds) for periodic logs across components."""


class TrainerLogConfig(LogConfig):
    ranks_filter: list[int] = [0]
    """Trainer ranks to show in console output. Passed to ``torchrun --local-ranks-filter``."""


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

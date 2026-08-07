import json
from argparse import Namespace
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import ConfigDict, Field, model_validator
from pydantic_config import BaseConfig

from prime_rl.configs.shared import EnvVars, LogConfig, SlurmConfig
from prime_rl.utils.config import find_package_resource
from prime_rl.utils.parsers import resolve_reasoning_parser, resolve_tool_call_parser

# TODO: Set thinking/ solution budget


class ServerConfig(BaseConfig):
    host: str | None = None
    """Host to bind to."""

    port: int = 8000
    """Port to bind to."""

    liveness_timeout_seconds: float = Field(30.0, gt=0)
    """Timeout in seconds for the ``/liveness`` endpoint's internal vLLM worker RPC. With Kubernetes liveness probes, keep the probe ``timeoutSeconds`` at least this high."""


# Valid vLLM max_lora_rank values (`vllm.config.lora.MaxLoRARanks`), excluding 1 so
# tiny adapters round up to 8. Hardcoded rather than imported: prime-rl-configs does
# not depend on vLLM, and importing it costs seconds in every config-parsing process.
VALID_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)

# vLLM all2all backend options for expert-parallel deployments.
All2AllBackend = Literal[
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
    "flashinfer_nvlink_one_sided",
    "flashinfer_nvlink_two_sided",
]


class VllmConfig(BaseConfig):
    """Arguments forwarded to the vLLM server, under vLLM's own argument names
    (https://docs.vllm.ai/en/latest/configuration/engine_args.html).

    The fields below are the arguments prime-rl types, defaults, or reads back;
    any other vLLM argument can be set as well (``[inference.vllm] max_num_seqs = 256``
    or ``--vllm.max-num-seqs 256``) and passes through to the server verbatim.

    Parser fields (``tool_call_parser``, ``reasoning_parser``) default to ``"auto"``,
    which resolves to a concrete parser name at validation time from the model name.
    Set to ``None`` to disable.
    """

    model_config = ConfigDict(extra="allow")

    model: str = "Qwen/Qwen3-0.6B"
    """HF model name or local path."""

    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    """dtype for model weights and activations. ``auto`` uses FP16 for FP32/FP16 models and BF16 for BF16 models."""

    max_model_len: int | None = None
    """Maximum model context length. If None, uses the model config's value."""

    enforce_eager: bool = False
    """Enforce eager mode. When False, PyTorch eager and cuda graphs run hybrid for maximum performance."""

    trust_remote_code: bool = False
    """Trust remote code when loading the model."""

    chat_template: str | None = None
    """Chat template — a Jinja2 template string or path to a template file. If None, uses the model's default."""

    tool_call_parser: str | None = "auto"
    """Tool-call parser. Set to ``"auto"`` (default) to detect from the model name, or ``None`` to disable."""

    reasoning_parser: str | None = "auto"
    """Parser for extracting reasoning content from model outputs. Set to ``"auto"`` (default) to detect from the model name, or ``None`` to disable."""

    rope_scaling: dict[str, Any] | str | None = None
    """RoPE scaling configuration as a dict (e.g. ``{rope_type="yarn", factor=4.0, original_max_position_embeddings=32768}``)."""

    tensor_parallel_size: int = 1
    """Tensor parallel size."""

    data_parallel_size: int = Field(1, ge=1)
    """Data parallel size."""

    data_parallel_size_local: int | None = Field(None, ge=1)
    """Data parallel replicas to run on this node."""

    data_parallel_rpc_port: int = Field(13345, ge=1, le=65535)
    """RPC port for data parallel communication."""

    api_server_count: int = Field(1, ge=0)
    """API servers to run. Set to 0 for headless mode."""

    seed: int = 0
    """Seed the inference components."""

    gpu_memory_utilization: float = 0.9
    """GPU memory utilization."""

    enable_prefix_caching: bool | None = None
    """Enable prefix caching."""

    quantization: str | None = None
    """Online inference quantization method. If None, vLLM infers it from the checkpoint."""

    enable_lora: bool = False
    """Enable LoRA."""

    max_loras: int = 8
    """Maximum number of LoRAs."""

    # TODO: The default value is very high because our areal impl for lora isn't ideal
    # We add a lora with the same name instead of changing weights inplace
    # Because we dont cancel requests that are past max_async, these requests could be using a LoRA that gets unloaded which will crash the inference server
    max_cpu_loras: int = 100
    """Maximum number of LoRAs on CPU."""

    max_lora_rank: int | None = None
    """Maximum LoRA rank. Rounded up to the nearest value vLLM accepts."""

    lora_target_modules: list[str] | None = None
    """LoRA target modules."""

    enable_expert_parallel: bool = False
    """Enable expert parallelism for MoE models."""

    all2all_backend: All2AllBackend = "allgather_reducescatter"
    """All-to-all backend for expert-parallel communication."""

    enable_eplb: bool = False
    """Enable expert parallel load balancer (EPLB)."""

    enable_dbo: bool = False
    """Enable dual batch overlap (DBO)."""

    enable_return_routed_experts: bool = False
    """Return routed experts in responses."""

    @model_validator(mode="before")
    @classmethod
    def parse_extra_values(cls, data: dict) -> dict:
        """Normalize pass-through (untyped) entries: kebab-case keys become snake_case,
        and string values — which is how CLI overrides arrive — are JSON-parsed so
        ``--vllm.max-num-seqs 256`` lands as an int and ``--vllm.compilation-config
        '{"cudagraph_mode": "NONE"}'`` as a dict. Non-JSON strings stay strings."""
        if not isinstance(data, dict):
            return data
        for key in [k for k in data if k not in cls.model_fields]:
            value = data.pop(key)
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            data[key.replace("-", "_")] = value
        return data

    @model_validator(mode="after")
    def auto_resolve_parsers(self):
        """Resolve ``"auto"`` parser values to concrete parser names from the model name.

        Runs after ``RLConfig.auto_setup_shared_configs`` (mode=before) has
        propagated the shared ``[model] name`` into ``inference.vllm``, so the
        name is set even when only the shared block specifies it.
        """
        if self.tool_call_parser == "auto":
            self.tool_call_parser = resolve_tool_call_parser(self.model)
        if self.reasoning_parser == "auto":
            self.reasoning_parser = resolve_reasoning_parser(self.model)
        return self

    @model_validator(mode="after")
    def auto_setup_max_lora_rank(self):
        """Auto-setup max_lora_rank by rounding up to the nearest valid vLLM value.

        vLLM only accepts specific values for max_lora_rank: (1, 8, 16, 32, 64, 128, 256, 320, 512).
        This validator ensures that any configured rank is rounded up to the minimum valid value
        that can serve adapters of the requested rank.
        """
        if self.max_lora_rank is not None:
            original_rank = self.max_lora_rank
            for valid_rank in VALID_VLLM_LORA_RANKS:
                if valid_rank >= self.max_lora_rank:
                    self.max_lora_rank = valid_rank
                    break
            else:
                raise ValueError(f"max_lora_rank={original_rank} exceeds vLLM maximum of {VALID_VLLM_LORA_RANKS[-1]}")
        return self

    @model_validator(mode="after")
    def auto_setup_api_server_count(self):
        """
        Ensures that we have at least as many API servers as data parallel
        size. Unless LoRA is enabled, in which case only one API server is
        supported (vLLM limitation).
        """
        if self.model_extra and self.model_extra.get("headless", False):
            self.api_server_count = 0
            return self

        if "api_server_count" not in self.model_fields_set:
            min_api_server_count = self.data_parallel_size_local or self.data_parallel_size
            if self.api_server_count < min_api_server_count:
                self.api_server_count = min_api_server_count

        if self.enable_lora:
            self.api_server_count = 1  # LoRA requires only one API server
        return self


class WeightBroadcastConfig(BaseConfig):
    type: Literal["nccl", "filesystem", "nixl"] = "filesystem"
    """Weight broadcast transport."""


class CPUOffloadTier(BaseConfig):
    num_bytes: int = Field(..., gt=0)
    """CPU/DRAM offload capacity. For the ``native`` backend this is vLLM's aggregate ``cpu_bytes_to_use`` (scaled across workers internally). For the ``mooncake`` backend this is the per-node store client's DRAM segment (``-global_segment_size``)."""


class DiskOffloadTier(BaseConfig):
    path: Path
    """Filesystem root for the disk tier. For ``native`` this is the ``fs_python`` secondary tier's ``root_dir``; for ``mooncake`` it is the store client's ``MOONCAKE_OFFLOAD_FILE_STORAGE_PATH``. Capacity is bounded by the filesystem at ``path`` (neither backend enforces a byte quota)."""


class BaseKVCacheOffloadConfig(BaseConfig):
    cpu: CPUOffloadTier | None = None
    """CPU/DRAM offload tier. Always required — disk-only offload is not supported."""

    disk: DiskOffloadTier | None = None
    """Optional disk tier, layered behind the CPU tier (GPU → DRAM → disk)."""

    @model_validator(mode="after")
    def valid_tiers(self):
        # Both backends support only two shapes: cpu-only or cpu+disk. Native disk
        # tiering needs a CPU primary tier; Mooncake standalone-store needs a DRAM
        # staging tier. Disk-only is rejected for both.
        if self.cpu is None:
            raise ValueError("inference.kv_cache_offload requires a cpu tier (disk-only offload is not supported).")
        return self


class NativeKVCacheOffloadConfig(BaseKVCacheOffloadConfig):
    type: Literal["native"] = "native"
    """vLLM-native offloading. cpu-only uses ``OffloadingConnector`` + ``CPUOffloadingSpec``; cpu+disk uses ``TieringOffloadingSpec`` (CPU primary tier + ``fs_python`` disk secondary). Fully self-contained — no external processes."""

    def to_connector_dict(self) -> dict[str, Any]:
        assert self.cpu is not None
        extra: dict[str, Any] = {"cpu_bytes_to_use": int(self.cpu.num_bytes)}
        if self.disk is not None:
            extra["spec_name"] = "TieringOffloadingSpec"
            extra["secondary_tiers"] = [{"type": "fs_python", "root_dir": str(self.disk.path)}]
        return {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
        }


class MooncakeKVCacheOffloadConfig(BaseKVCacheOffloadConfig):
    type: Literal["mooncake"] = "mooncake"
    """Mooncake distributed store offloading (SLURM only). One ``mooncake_master`` + metadata server runs on the head inference node; every node runs a ``mooncake_client`` contributing its segment to the single shared pool, so prefixes cached on any node are reusable by all. The cpu tier sizes each node's DRAM segment; the optional disk tier adds an SSD tier."""

    device_name: str = ""
    """RDMA device name(s) for the store (empty = auto-detect)."""

    def to_connector_dict(self) -> dict[str, Any]:
        # Addresses/sizes/tiers are realized by the per-node store launch in the sbatch
        # template (MOONCAKE_CONFIG_PATH JSON); blocks are keyed by model + parallel rank +
        # content hash (no instance id), so the shared pool is reused across nodes/replicas.
        return {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {},
        }


KVCacheOffloadConfig: TypeAlias = Annotated[
    NativeKVCacheOffloadConfig | MooncakeKVCacheOffloadConfig, Field(discriminator="type")
]


# Known llm-d EPP scorer plugins (used to guard the ``scorers`` map against typos).
KNOWN_SCORERS = frozenset(
    {
        "prefix-cache-scorer",
        "precise-prefix-cache-scorer",
        "queue-scorer",
        "kv-cache-utilization-scorer",
        "active-request-scorer",
        "load-aware-scorer",
        "running-requests-size-scorer",
        "token-load-scorer",
        "latency-scorer",
        "session-affinity-scorer",
        "lora-affinity-scorer",
    }
)


class VllmRouterConfig(BaseConfig):
    """PrimeIntellect vllm-router."""

    type: Literal["vllm-router"] = "vllm-router"

    policy: str = "consistent_hash"
    """Routing policy, e.g. ``consistent_hash`` or ``round_robin``."""


class LlmdRouterConfig(BaseConfig):
    """llm-d router backend (EPP + Envoy)."""

    type: Literal["llm-d"] = "llm-d"

    scorers: dict[str, float] = {
        "prefix-cache-scorer": 3.0,
        "active-request-scorer": 2.0,
    }
    """EPP scorer name → weight, applied to every routing profile (before the per-profile P/D overrides). Defaults to prefix-cache affinity plus in-flight (active-request) load balancing. Unknown scorer names are rejected."""

    prefill_scorer_overrides: dict[str, float] = {
        "queue-scorer": 2.0,
        "kv-cache-utilization-scorer": 2.0,
    }
    """P/D only: scorer → weight merged onto ``scorers`` for the prefill profile (a per-profile weight overrides the base)."""

    decode_scorer_overrides: dict[str, float] = {}
    """P/D only: scorer → weight merged onto ``scorers`` for the decode profile (a per-profile weight overrides the base); empty by default."""

    non_cached_tokens: int = 16
    """P/D only: requests with fewer than this many non-cached prompt tokens skip remote prefill and run decode-only."""

    decode_sidecar_port: int = 8300
    """P/D only: port the decode-side llm-d sidecar listens on."""

    @property
    def prefill_scorers(self) -> dict[str, float]:
        """Effective prefill-profile scorers: ``scorers`` merged with ``prefill_scorer_overrides``."""
        return {**self.scorers, **self.prefill_scorer_overrides}

    @property
    def decode_scorers(self) -> dict[str, float]:
        """Effective decode-profile scorers: ``scorers`` merged with ``decode_scorer_overrides``."""
        return {**self.scorers, **self.decode_scorer_overrides}

    @model_validator(mode="after")
    def validate_scorers(self):
        unknown = (
            set(self.scorers) | set(self.prefill_scorer_overrides) | set(self.decode_scorer_overrides)
        ) - KNOWN_SCORERS
        if unknown:
            raise ValueError(f"Unknown llm-d scorer(s): {sorted(unknown)}. Known scorers: {sorted(KNOWN_SCORERS)}.")
        return self


# Discriminated on ``type`` so the launch path can pick the router backend.
RouterConfig: TypeAlias = Annotated[VllmRouterConfig | LlmdRouterConfig, Field(discriminator="type")]


class BaseInferenceDeploymentConfig(BaseConfig):
    gpus_per_node: int = 8
    """GPUs per node."""


class SingleNodeInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    type: Literal["single_node"] = "single_node"


# Multi-node inference: each node runs an independent vLLM replica.
class MultiNodeInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    type: Literal["multi_node"] = "multi_node"

    num_nodes: int = Field(2, ge=1)
    """Inference nodes."""


# Disaggregated prefill/decode inference. Each replica is split into separate
# prefill and decode node groups. Requires NIXL for KV transfer and a router for
# request routing.
class DisaggregatedInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    type: Literal["disaggregated"] = "disaggregated"

    prefill_nodes_per_replica: int = Field(1, ge=1)
    """Nodes in each prefill vLLM instance."""

    decode_nodes_per_replica: int = Field(1, ge=1)
    """Nodes in each decode vLLM instance."""

    num_prefill_replicas: int = Field(1, ge=1)
    """Independent prefill vLLM instances."""

    num_decode_replicas: int = Field(1, ge=1)
    """Independent decode vLLM instances."""

    prefill_port: int = 8100
    """Port for prefill vLLM instances."""

    decode_port: int = 8200
    """Port for decode vLLM instances."""

    prefill_env_vars: EnvVars = {}
    """Extra environment variables exported only on prefill nodes."""

    decode_env_vars: EnvVars = {}
    """Extra environment variables exported only on decode nodes."""

    prefill_vllm_overrides: dict[str, Any] = {}
    """Extra vLLM config options merged into --vllm-extra only for prefill ranks (SLURM only)."""

    decode_vllm_overrides: dict[str, Any] = {}
    """Extra vLLM config options merged into --vllm-extra only for decode ranks (SLURM only)."""

    @property
    def num_prefill_nodes(self) -> int:
        return self.prefill_nodes_per_replica * self.num_prefill_replicas

    @property
    def num_decode_nodes(self) -> int:
        return self.decode_nodes_per_replica * self.num_decode_replicas

    @property
    def num_nodes(self) -> int:
        return self.num_prefill_nodes + self.num_decode_nodes


InferenceDeploymentConfig: TypeAlias = Annotated[
    SingleNodeInferenceDeploymentConfig | MultiNodeInferenceDeploymentConfig | DisaggregatedInferenceDeploymentConfig,
    Field(discriminator="type"),
]


class InferenceConfig(BaseConfig):
    server: ServerConfig = ServerConfig()

    router: RouterConfig | None = Field(default_factory=VllmRouterConfig)
    """Router fronting the engine(s). Every deployment runs a single global router as the client-facing endpoint on ``server.port``, with the engines listening on ``backend_port``. Set to ``None`` to run a bare engine on ``server.port`` without a router (the SLURM launchers do this for per-rank engine processes; the sbatch script starts the router)."""

    backend_port: int = 8100
    """Port for the vLLM engine(s) behind the router. Defaults to ``server.port + 100``. Multi-node deployments start one engine per local DP rank at ``backend_port + rank``."""

    vllm: VllmConfig = Field(default_factory=VllmConfig)
    """vLLM server arguments, under vLLM's own argument names. Unknown fields pass through verbatim."""

    log: LogConfig = LogConfig()
    """Logging configuration."""

    env_vars: EnvVars = {}
    """Extra environment variables for the inference server process(es). Merged on top of the launcher defaults."""

    use_deep_gemm: bool = False
    """Enable vLLM DeepGEMM FP8 kernels ``VLLM_USE_DEEP_GEMM=1``. Only works with block-wise FP8 quantization (e.g. GLM-5-FP8)."""

    weight_broadcast: WeightBroadcastConfig = WeightBroadcastConfig()

    kv_cache_offload: KVCacheOffloadConfig | None = None
    """KV cache offload for inference workers, as composable CPU/disk tiers. Discriminated on ``type``: ``native`` (vLLM ``OffloadingConnector``/``TieringOffloadingSpec``, self-contained) or ``mooncake`` (per-node Mooncake distributed store). Disaggregated P/D combines the chosen connector with NIXL through ``MultiConnector``."""

    use_pd_kv_transfer: bool = False
    """Auto-set for disaggregated P/D: emit the NIXL transfer connector. Persisted into the per-node config (which drops ``deployment``) so the connector is still built per worker. Not meant to be set by hand."""

    enable_fp32_lm_head: bool = True
    """Run the lm_head projection in fp32 via a native bf16×bf16 → fp32 GEMM (``torch.mm`` with ``out_dtype=torch.float32``). Stabilizes logprob precision under FP8/bf16 inference, matching SGLang's ``--enable-fp32-lm-head``. Implemented as a monkey-patch over vLLM's LogitsProcessor, activated by setting ``additional_config["fp32_lm_head"] = True`` on the vLLM config."""

    enable_fp32_router_logits: bool = True
    """Emit fp32 MoE router logits for DeepSeek-family models (incl. GLM-5.x) by setting ``out_dtype=float32`` on the gate: the bf16×bf16 gate GEMM writes its fp32 accumulator out unrounded instead of truncating logits to bf16 before expert scoring. Matches fp32-routed checkpoints (e.g. GLM-5.x, trained with Megatron ``--moe-router-dtype fp32``); pairs with ``trainer.model.moe_router_dtype = "float32"``. Implemented as a monkey-patch over vLLM's DeepseekV2MoE, activated by setting ``additional_config["fp32_router_logits"] = True`` on the vLLM config."""

    # Launcher-only fields

    deployment: InferenceDeploymentConfig = SingleNodeInferenceDeploymentConfig()

    slurm: SlurmConfig | None = None
    """SLURM configuration. When set, the run is submitted as a SLURM job instead of running locally."""

    output_dir: Path = Path("outputs")
    """Directory for SLURM logs and generated scripts."""

    dry_run: bool = False
    """Only validate and dump resolved configs, then exit early."""

    @model_validator(mode="after")
    def validate_multi_node_requires_slurm(self):
        if self.deployment.type in ("multi_node", "disaggregated") and self.slurm is None:
            raise ValueError("Must use SLURM for multi-node / disaggregated deployment.")
        return self

    @model_validator(mode="after")
    def validate_llmd_no_routed_experts(self):
        """Reject routed-expert return with the llm-d router (breaks P/D, unverified for multi-node)."""
        if self.router is not None and self.router.type == "llm-d" and self.vllm.enable_return_routed_experts:
            raise ValueError(
                "The llm-d router backend does not support routed-expert return "
                "(enable_return_routed_experts): it breaks P/D and is unverified for multi-node. "
                "Use router type 'vllm-router' for routed-expert runs."
            )
        return self

    @model_validator(mode="after")
    def validate_router_deployment(self):
        """The llm-d router (EPP + Envoy) is launched by the SLURM templates only; multi-node deployments need a router to front the per-rank engines."""
        if self.router is not None and self.router.type == "llm-d" and self.deployment.type == "single_node":
            raise ValueError("The llm-d router backend requires a multi-node or disaggregated SLURM deployment.")
        if self.router is None and self.deployment.type in ("multi_node", "disaggregated"):
            raise ValueError("Multi-node / disaggregated deployments require a router fronting the per-rank engines.")
        return self

    @model_validator(mode="after")
    def auto_setup_backend_port(self):
        if self.router is not None and "backend_port" not in self.model_fields_set:
            self.backend_port = self.server.port + 100
        return self

    @model_validator(mode="after")
    def auto_setup_kv_cache_offload(self):
        if self.kv_cache_offload is not None:
            if self.vllm.enable_prefix_caching is False:
                raise ValueError("KV cache offloading requires inference.vllm.enable_prefix_caching to be true.")
            if "enable_prefix_caching" not in self.vllm.model_fields_set:
                self.vllm.enable_prefix_caching = True

        return self

    @model_validator(mode="after")
    def auto_setup_disaggregated(self):
        """Auto-configure inference for disaggregated P/D: enable EP and compute DP."""
        if self.deployment.type == "disaggregated":
            self.use_pd_kv_transfer = True
            if "enable_expert_parallel" not in self.vllm.model_fields_set:
                self.vllm.enable_expert_parallel = True
            if "enable_eplb" not in self.vllm.model_fields_set:
                self.vllm.enable_eplb = False
            gpus_per_node = self.deployment.gpus_per_node
            tp = self.vllm.tensor_parallel_size
            dp_per_node = gpus_per_node // tp
            if self.vllm.data_parallel_size_local is None:
                self.vllm.data_parallel_size_local = dp_per_node
            if self.vllm.data_parallel_size == 1:
                self.vllm.data_parallel_size = dp_per_node
            if self.vllm.api_server_count == 1:
                self.vllm.api_server_count = dp_per_node
        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        if self.slurm is not None and self.slurm.template_path is None:
            templates_dir = find_package_resource("templates")
            if templates_dir is not None:
                self.slurm.template_path = templates_dir / "inference.sbatch.j2"
        return self

    def build_kv_transfer_config(self) -> dict[str, Any] | None:
        """Build the single vLLM ``kv_transfer_config`` from the transfer + offload connectors.

        Disaggregated P/D always uses NIXL for prefill→decode transfer. KV cache offload (if
        configured) contributes its own connector. When both are present they are composed via
        ``MultiConnector``. Returns None when neither applies.
        """
        connectors: list[dict[str, Any]] = []
        if self.use_pd_kv_transfer:
            connectors.append(
                {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"num_threads": 1},
                }
            )
        if self.kv_cache_offload is not None:
            connectors.append(self.kv_cache_offload.to_connector_dict())

        if not connectors:
            return None
        if len(connectors) == 1:
            return connectors[0]
        return {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"connectors": connectors},
        }

    # Fields vLLM rejects as None — omitted from the namespace so vLLM applies its
    # own default (e.g. quantization is inferred from the checkpoint).
    _OMIT_IF_NONE = frozenset(
        {"chat_template", "tool_call_parser", "reasoning_parser", "lora_target_modules", "quantization", "rope_scaling"}
    )

    def to_namespace(self) -> Namespace:
        """Dump the server + vllm sections into a vLLM-compatible Namespace."""
        namespace = Namespace(
            host=self.server.host,
            port=self.server.port,
            liveness_timeout_seconds=self.server.liveness_timeout_seconds,
        )

        extra_fields = self.vllm.model_extra or {}
        for key in (*type(self.vllm).model_fields, *extra_fields):
            value = getattr(self.vllm, key)
            # None extras are dropped too: `--vllm.foo None` means "use vLLM's default".
            if value is None and (key in self._OMIT_IF_NONE or key in extra_fields):
                continue
            setattr(namespace, key, value)

        # Auto tool choice is gated on a tool-call parser being resolved, unless
        # explicitly overridden.
        if "enable_auto_tool_choice" not in extra_fields:
            namespace.enable_auto_tool_choice = hasattr(namespace, "tool_call_parser")

        # Default `logprobs_mode` to `processed_logprobs`
        if not hasattr(namespace, "logprobs_mode"):
            namespace.logprobs_mode = "processed_logprobs"

        kv_transfer_config = self.build_kv_transfer_config()
        if kv_transfer_config is not None:
            namespace.kv_transfer_config = kv_transfer_config

        # Pass prime-rl-specific flags through vLLM's additional_config dict;
        # workers read these via get_current_vllm_config().additional_config.
        additional_config = getattr(namespace, "additional_config", None) or {}
        if self.enable_fp32_lm_head:
            additional_config["fp32_lm_head"] = True
        if self.enable_fp32_router_logits:
            additional_config["fp32_router_logits"] = True
        if additional_config:
            namespace.additional_config = additional_config

        return namespace

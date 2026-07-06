from argparse import Namespace
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

from prime_rl.configs.shared import BaseModelConfig, EnvVars, LogConfig, SlurmConfig
from prime_rl.utils.config import find_package_resource, rgetattr, rsetattr
from prime_rl.utils.parsers import resolve_reasoning_parser, resolve_tool_call_parser

# TODO: Set thinking/ solution budget


class ServerConfig(BaseConfig):
    host: str | None = None
    """Host to bind to."""

    port: int = 8000
    """Port to bind to."""

    liveness_timeout_seconds: float = Field(30.0, gt=0)
    """Timeout in seconds for the ``/liveness`` endpoint's internal vLLM worker RPC. With Kubernetes liveness probes, keep the probe ``timeoutSeconds`` at least this high."""


class ParallelConfig(BaseConfig):
    tp: int = 1
    """Tensor parallel size. Forwarded to vLLM as ``--tensor-parallel-size``."""

    dp: int = Field(1, ge=1)
    """Data parallel size. Forwarded to vLLM as ``--data-parallel-size``."""

    def __str__(self) -> str:
        return f"tp={self.tp} dp={self.dp}"


class ModelConfig(BaseModelConfig):
    """Configures the inference model. Most arguments are passed directly to the vLLM LLM class (https://docs.vllm.ai/en/latest/api/vllm.LLM.html).

    Parser fields (``tool_call_parser``, ``reasoning_parser``) default to ``"auto"``,
    which resolves to a concrete parser name at validation time from the model name.
    Set to ``None`` to disable.
    """

    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    """dtype for model weights and activations. ``auto`` uses FP16 for FP32/FP16 models and BF16 for BF16 models. Forwarded as ``--dtype``."""

    max_model_len: int | None = None
    """Maximum model context length. If None, uses the model config's value. Forwarded as ``--max-model-len``."""

    enforce_eager: bool = False
    """Enforce eager mode. When False, PyTorch eager and cuda graphs run hybrid for maximum performance. Forwarded as ``--enforce-eager``."""

    trust_remote_code: bool = False
    """Trust remote code. Forwarded to vLLM engine init."""

    chat_template: str | None = None
    """Chat template — a Jinja2 template string or path to a template file. Forwarded as ``--chat-template``. If None, uses the model's default."""

    tool_call_parser: str | None = "auto"
    """Tool-call parser. Forwarded as ``--tool-call-parser``. Set to ``"auto"`` (default) to detect from the model name, or ``None`` to disable."""

    reasoning_parser: str | None = "auto"
    """Parser for extracting reasoning content from model outputs. Forwarded as ``--reasoning-parser``. Set to ``"auto"`` (default) to detect from the model name, or ``None`` to disable."""

    rope_scaling: dict[str, Any] | str | None = None
    """RoPE scaling configuration as a dict (e.g. ``{rope_type="yarn", factor=4.0, original_max_position_embeddings=32768}``). Forwarded as ``--rope-scaling``."""

    @model_validator(mode="after")
    def auto_resolve_parsers(self):
        """Resolve ``"auto"`` parser values to concrete parser names from the model name.

        Runs after ``RLConfig.auto_setup_shared_configs`` (mode=before) has
        propagated the shared ``[model] name`` into ``inference.model``, so the
        name is set even when only the shared block specifies it.
        """
        if self.tool_call_parser == "auto":
            self.tool_call_parser = resolve_tool_call_parser(self.name)
        if self.reasoning_parser == "auto":
            self.reasoning_parser = resolve_reasoning_parser(self.name)
        return self


class WeightBroadcastConfig(BaseConfig):
    type: Literal["nccl", "filesystem"] = "filesystem"
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


# Valid vLLM max_lora_rank values (from vllm/config/lora.py)
# TODO: on newer vLLM, can import via `get_args(vllm.config.lora.MaxLoRARanks)`
VALID_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)

# vLLM all2all backend options for expert-parallel deployments.
All2AllBackend = Literal[
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
    "flashinfer_nvlink_one_sided",
    "flashinfer_nvlink_two_sided",
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

    port: int = 8000
    """Port the router listens on — becomes the client-facing router URL."""

    policy: str = "consistent_hash"
    """Routing policy, e.g. ``consistent_hash`` or ``round_robin``."""


class LlmdRouterConfig(BaseConfig):
    """llm-d router backend (EPP + Envoy)."""

    type: Literal["llm-d"] = "llm-d"

    port: int = 8000
    """Port the Envoy gateway listens on — becomes the client-facing router URL."""

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

    backend_port: int = 8100
    """Port for the per-rank vLLM backend instances."""

    router: RouterConfig = VllmRouterConfig()
    """Router fronting the per-rank endpoints."""


class SingleNodeInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    type: Literal["single_node"] = "single_node"


# Multi-node inference: each node runs an independent vLLM replica.
class MultiNodeInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    type: Literal["multi_node"] = "multi_node"

    num_nodes: int = Field(2, ge=1)
    """Inference nodes."""


# Disaggregated prefill/decode inference. Each replica is split into separate
# prefill and decode node groups. Requires NIXL for KV transfer and a vllm-router
# for request routing. Multi-replica: set ``num_prefill_replicas`` /
# ``num_decode_replicas`` to run multiple independent vLLM instances within the
# prefill / decode node groups. E.g. ``num_prefill_nodes=4, num_prefill_replicas=2``
# creates two prefill vLLM instances each spanning 2 nodes (EP16 with 8 GPUs/node).
class DisaggregatedInferenceDeploymentConfig(BaseInferenceDeploymentConfig):
    type: Literal["disaggregated"] = "disaggregated"

    num_prefill_nodes: int = Field(1, ge=1)
    """Total prefill nodes."""

    num_decode_nodes: int = Field(1, ge=1)
    """Total decode nodes."""

    num_prefill_replicas: int = Field(1, ge=1)
    """Independent prefill vLLM instances. Must evenly divide ``num_prefill_nodes``."""

    num_decode_replicas: int = Field(1, ge=1)
    """Independent decode vLLM instances. Must evenly divide ``num_decode_nodes``."""

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
    def num_nodes(self) -> int:
        return self.num_prefill_nodes + self.num_decode_nodes

    @model_validator(mode="after")
    def validate_replicas_divide_nodes(self):
        if self.num_prefill_nodes % self.num_prefill_replicas != 0:
            raise ValueError(
                f"num_prefill_replicas ({self.num_prefill_replicas}) must evenly divide "
                f"num_prefill_nodes ({self.num_prefill_nodes})"
            )
        if self.num_decode_nodes % self.num_decode_replicas != 0:
            raise ValueError(
                f"num_decode_replicas ({self.num_decode_replicas}) must evenly divide "
                f"num_decode_nodes ({self.num_decode_nodes})"
            )
        return self


InferenceDeploymentConfig: TypeAlias = Annotated[
    SingleNodeInferenceDeploymentConfig | MultiNodeInferenceDeploymentConfig | DisaggregatedInferenceDeploymentConfig,
    Field(discriminator="type"),
]


class InferenceConfig(BaseConfig):
    server: ServerConfig = ServerConfig()

    model: ModelConfig = Field(default_factory=ModelConfig)

    parallel: ParallelConfig = ParallelConfig()
    """Multi-node and multi-GPU parallelism (TP, DP, PP)."""

    log: LogConfig = LogConfig()
    """Logging configuration."""

    env_vars: EnvVars = {}
    """Extra environment variables for the inference server process(es). Merged on top of the launcher defaults."""

    enable_lora: bool = False
    """Enable LoRA. Forwarded as ``--enable-lora``."""

    max_loras: int = 8
    """Maximum number of LoRAs. Forwarded as ``--max-loras``."""

    # TODO: The default value is very high because our areal impl for lora isn't ideal
    # We add a lora with the same name instead of changing weights inplace
    # Because we dont cancel requests that are past max_async, these requests could be using a LoRA that gets unloaded which will crash the inference server
    max_cpu_loras: int = 100
    """Maximum number of LoRAs on CPU. Forwarded as ``--max-cpu-loras``."""

    max_lora_rank: int | None = None
    """Maximum LoRA rank. Forwarded as ``--max-lora-rank``."""

    lora_target_modules: list[str] | None = None
    """LoRA target modules. Forwarded as ``--lora-target-modules``."""

    enable_prefix_caching: bool | None = None
    """Enable prefix caching. Forwarded as ``--enable-prefix-caching``."""

    gpu_memory_utilization: float = 0.9
    """GPU memory utilization. Forwarded as ``--gpu-memory-utilization``."""

    api_server_count: int = Field(1, ge=0)
    """API servers to run. Forwarded as ``--api-server-count``. Set to 0 for headless mode."""

    data_parallel_size_local: int | None = Field(None, ge=1)
    """Data parallel replicas to run on this node. Forwarded as ``--data-parallel-size-local``."""

    data_parallel_rpc_port: int = Field(13345, ge=1, le=65535)
    """RPC port for data parallel communication. Forwarded as ``--data-parallel-rpc-port``."""

    seed: int = 0
    """Seed the inference components. Forwarded as ``--seed``."""

    enable_expert_parallel: bool = False
    """Enable expert parallelism for MoE models. Forwarded as ``--enable-expert-parallel``."""

    all2all_backend: All2AllBackend = "allgather_reducescatter"
    """All-to-all backend for expert-parallel communication. Forwarded as ``--all2all-backend``."""

    enable_eplb: bool = False
    """Enable expert parallel load balancer (EPLB). Forwarded as ``--enable-eplb``."""

    enable_dbo: bool = False
    """Enable dual batch overlap (DBO). Forwarded as ``--enable-dbo``."""

    use_deep_gemm: bool = False
    """Enable vLLM DeepGEMM FP8 kernels ``VLLM_USE_DEEP_GEMM=1``. Only works with block-wise FP8 quantization (e.g. GLM-5-FP8)."""

    weight_broadcast: WeightBroadcastConfig = WeightBroadcastConfig()

    kv_cache_offload: KVCacheOffloadConfig | None = None
    """KV cache offload for inference workers, as composable CPU/disk tiers. Discriminated on ``type``: ``native`` (vLLM ``OffloadingConnector``/``TieringOffloadingSpec``, self-contained) or ``mooncake`` (per-node Mooncake distributed store). Disaggregated P/D combines the chosen connector with NIXL through ``MultiConnector``."""

    use_pd_kv_transfer: bool = False
    """Auto-set for disaggregated P/D: emit the NIXL transfer connector. Persisted into the per-node config (which drops ``deployment``) so the connector is still built per worker. Not meant to be set by hand."""

    enable_return_routed_experts: bool = False
    """Return routed experts in responses. Forwarded as ``--enable-return-routed-experts``."""

    enable_fp32_lm_head: bool = True
    """Run the lm_head projection in fp32 via a native bf16×bf16 → fp32 GEMM (``torch.mm`` with ``out_dtype=torch.float32``). Stabilizes logprob precision under FP8/bf16 inference, matching SGLang's ``--enable-fp32-lm-head``. Implemented as a monkey-patch over vLLM's LogitsProcessor, activated by setting ``additional_config["fp32_lm_head"] = True`` on the vLLM config."""

    enable_fp32_router_logits: bool = True
    """Emit fp32 MoE router logits for DeepSeek-family models (incl. GLM-5.x) by setting ``out_dtype=float32`` on the gate: the bf16×bf16 gate GEMM writes its fp32 accumulator out unrounded instead of truncating logits to bf16 before expert scoring. Matches fp32-routed checkpoints (e.g. GLM-5.x, trained with Megatron ``--moe-router-dtype fp32``); pairs with ``trainer.model.moe_router_dtype = "float32"``. Implemented as a monkey-patch over vLLM's DeepseekV2MoE, activated by setting ``additional_config["fp32_router_logits"] = True`` on the vLLM config."""

    vllm_extra: dict[str, Any] = {}
    """Extra arguments forwarded to vLLM. Applied as attributes on the vLLM namespace after config translation."""

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
        router = getattr(self.deployment, "router", None)
        if router is not None and router.type == "llm-d" and self.enable_return_routed_experts:
            raise ValueError(
                "The llm-d router backend does not support routed-expert return "
                "(enable_return_routed_experts): it breaks P/D and is unverified for multi-node. "
                "Use router type 'vllm-router' for routed-expert runs."
            )
        return self

    @model_validator(mode="after")
    def auto_setup_kv_cache_offload(self):
        if self.kv_cache_offload is not None:
            if self.enable_prefix_caching is False:
                raise ValueError("KV cache offloading requires inference.enable_prefix_caching to be true.")
            if "enable_prefix_caching" not in self.model_fields_set:
                self.enable_prefix_caching = True

        return self

    @model_validator(mode="after")
    def auto_setup_disaggregated(self):
        """Auto-configure inference for disaggregated P/D: enable EP and compute DP."""
        if self.deployment.type == "disaggregated":
            self.use_pd_kv_transfer = True
            if "enable_expert_parallel" not in self.model_fields_set:
                self.enable_expert_parallel = True
            if "enable_eplb" not in self.model_fields_set:
                self.enable_eplb = False
            gpus_per_node = self.deployment.gpus_per_node
            tp = self.parallel.tp
            dp_per_node = gpus_per_node // tp
            if self.data_parallel_size_local is None:
                self.data_parallel_size_local = dp_per_node
            if self.parallel.dp == 1:
                self.parallel.dp = dp_per_node
            if self.api_server_count == 1:
                self.api_server_count = dp_per_node
        return self

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        if self.slurm is not None and self.slurm.template_path is None:
            templates_dir = find_package_resource("templates")
            if templates_dir is not None:
                self.slurm.template_path = templates_dir / "inference.sbatch.j2"
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
        if self.vllm_extra.get("headless", False):
            self.api_server_count = 0
            return self

        if "api_server_count" not in self.model_fields_set:
            min_api_server_count = self.data_parallel_size_local or self.parallel.dp
            if self.api_server_count < min_api_server_count:
                self.api_server_count = min_api_server_count

        if self.enable_lora:
            self.api_server_count = 1  # LoRA requires only one API server
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

    def to_vllm(self) -> Namespace:
        """Convert InferenceConfig to vLLM-compatible Namespace."""
        namespace = Namespace()
        to_vllm = {
            "server.host": "host",
            "server.port": "port",
            "server.liveness_timeout_seconds": "liveness_timeout_seconds",
            "model.name": "model",
            "model.dtype": "dtype",
            "model.max_model_len": "max_model_len",
            "model.enforce_eager": "enforce_eager",
            "model.trust_remote_code": "trust_remote_code",
            "model.chat_template": "chat_template",
            "model.tool_call_parser": "tool_call_parser",
            "model.reasoning_parser": "reasoning_parser",
            "model.rope_scaling": "rope_scaling",
            "parallel.tp": "tensor_parallel_size",
            "parallel.dp": "data_parallel_size",
            "data_parallel_size_local": "data_parallel_size_local",
            "data_parallel_rpc_port": "data_parallel_rpc_port",
            "enable_lora": "enable_lora",
            "enable_prefix_caching": "enable_prefix_caching",
            "max_loras": "max_loras",
            "max_cpu_loras": "max_cpu_loras",
            "max_lora_rank": "max_lora_rank",
            "lora_target_modules": "lora_target_modules",
            "gpu_memory_utilization": "gpu_memory_utilization",
            "api_server_count": "api_server_count",
            "enable_return_routed_experts": "enable_return_routed_experts",
            "enable_expert_parallel": "enable_expert_parallel",
            "all2all_backend": "all2all_backend",
            "enable_eplb": "enable_eplb",
            "enable_dbo": "enable_dbo",
            "seed": "seed",
        }

        for config_key, vllm_key in to_vllm.items():
            value = rgetattr(self, config_key.replace("-", "_"))
            rsetattr(namespace, vllm_key, value)

        # Set `logprobs_mode` to `processed_logprobs` by default
        rsetattr(namespace, "logprobs_mode", "processed_logprobs")

        kv_transfer_config = self.build_kv_transfer_config()
        if kv_transfer_config is not None:
            rsetattr(namespace, "kv_transfer_config", kv_transfer_config)

        # Pass prime-rl-specific flags through vLLM's additional_config dict;
        # workers read these via get_current_vllm_config().additional_config.
        if self.enable_fp32_lm_head:
            existing = getattr(namespace, "additional_config", None) or {}
            existing["fp32_lm_head"] = True
            rsetattr(namespace, "additional_config", existing)
        if self.enable_fp32_router_logits:
            existing = getattr(namespace, "additional_config", None) or {}
            existing["fp32_router_logits"] = True
            rsetattr(namespace, "additional_config", existing)

        # Remove chat_template if not set (vLLM doesn't accept None)
        if namespace.chat_template is None:
            delattr(namespace, "chat_template")

        # Remove tool_call_parser if not set (vLLM doesn't accept None) and gate
        # `enable_auto_tool_choice` on its presence.
        if namespace.tool_call_parser is None:
            delattr(namespace, "tool_call_parser")
        namespace.enable_auto_tool_choice = hasattr(namespace, "tool_call_parser")

        # Remove reasoning_parser if not set (vLLM doesn't accept None)
        if namespace.reasoning_parser is None:
            delattr(namespace, "reasoning_parser")

        # Remove lora_target_modules if not set (vLLM doesn't accept None)
        if hasattr(namespace, "lora_target_modules") and namespace.lora_target_modules is None:
            delattr(namespace, "lora_target_modules")

        # Remove rope_scaling if not set (vLLM doesn't accept None)
        if hasattr(namespace, "rope_scaling"):
            if namespace.rope_scaling is None:
                delattr(namespace, "rope_scaling")

        return namespace

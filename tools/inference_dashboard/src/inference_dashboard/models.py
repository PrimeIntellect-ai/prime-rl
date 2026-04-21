from __future__ import annotations

from pydantic import BaseModel, Field


class FromSlurmRequest(BaseModel):
    job_id: int = Field(ge=1)
    num_cold_prefill_nodes_per_pod: int = Field(default=0, ge=0)
    num_prefill_nodes_per_pod: int = Field(ge=1)
    num_decode_nodes_per_pod: int = Field(ge=1)
    num_prefill_replicas_per_pod: int = Field(default=1, ge=1)
    num_decode_replicas_per_pod: int = Field(default=1, ge=1)
    num_replicas: int = Field(ge=1)

    @property
    def nodes_per_pod(self) -> int:
        return self.num_cold_prefill_nodes_per_pod + self.num_prefill_nodes_per_pod + self.num_decode_nodes_per_pod

    @property
    def inference_nodes(self) -> int:
        return self.num_replicas * self.nodes_per_pod

    @property
    def prefill_nodes_per_replica(self) -> int:
        return self.num_prefill_nodes_per_pod // self.num_prefill_replicas_per_pod

    @property
    def decode_nodes_per_replica(self) -> int:
        return self.num_decode_nodes_per_pod // self.num_decode_replicas_per_pod


class NodeEndpoint(BaseModel):
    id: str
    hostname: str
    role: str
    pod_index: int
    role_replica_index: int
    global_index: int
    role_index: int
    role_replica_rank: int
    metrics_url: str
    health_url: str
    display_name: str


class PodEndpoint(BaseModel):
    id: str
    pod_index: int
    router_hostname: str
    router_url: str
    router_health_url: str
    cold_prefill_nodes: list[NodeEndpoint]
    prefill_nodes: list[NodeEndpoint]
    decode_nodes: list[NodeEndpoint]


class Topology(BaseModel):
    source: str = "from_slurm"
    job_id: int
    total_job_nodes: int
    total_inference_nodes: int
    num_replicas: int
    num_cold_prefill_nodes_per_pod: int
    num_prefill_nodes_per_pod: int
    num_decode_nodes_per_pod: int
    num_prefill_replicas_per_pod: int
    num_decode_replicas_per_pod: int
    hostnames: list[str]
    pods: list[PodEndpoint]


class Alert(BaseModel):
    severity: str
    scope: str
    target: str
    message: str


class TimeSeriesPoint(BaseModel):
    timestamp: float
    value: float


class EngineSnapshot(BaseModel):
    id: str
    engine_index: str
    running_requests: float = 0.0
    waiting_requests: float = 0.0
    kv_cache_usage: float = 0.0
    cpu_kv_cache_usage: float | None = None
    prefix_cache_hit_rate: float | None = None
    cpu_prefix_cache_hit_rate: float | None = None
    prompt_tokens_per_second: float = 0.0
    generation_tokens_per_second: float = 0.0
    requests_finished_per_second: float = 0.0
    avg_prefill_time_seconds: float | None = None
    avg_decode_time_seconds: float | None = None
    avg_queue_time_seconds: float | None = None
    avg_ttft_seconds: float | None = None
    avg_tpot_seconds: float | None = None
    avg_e2e_latency_seconds: float | None = None
    nixl_avg_transfer_time_seconds: float | None = None
    nixl_transfers_per_second: float = 0.0
    nixl_bytes_per_second: float = 0.0
    nixl_avg_bytes_per_transfer: float | None = None
    nixl_failed_transfers_total: float = 0.0
    nixl_failed_notifications_total: float = 0.0
    nixl_kv_expired_requests_total: float = 0.0


class NodeSnapshot(BaseModel):
    id: str
    hostname: str
    role: str
    pod_index: int
    role_replica_index: int
    scrape_ok: bool
    scrape_age_seconds: float
    running_requests: float = 0.0
    waiting_requests: float = 0.0
    kv_cache_usage_mean: float = 0.0
    kv_cache_usage_max: float = 0.0
    cpu_kv_cache_usage_mean: float | None = None
    cpu_kv_cache_usage_max: float | None = None
    prefix_cache_hit_rate: float | None = None
    cpu_prefix_cache_hit_rate: float | None = None
    prompt_tokens_per_second: float = 0.0
    generation_tokens_per_second: float = 0.0
    requests_finished_per_second: float = 0.0
    avg_prefill_time_seconds: float | None = None
    avg_decode_time_seconds: float | None = None
    avg_queue_time_seconds: float | None = None
    avg_ttft_seconds: float | None = None
    avg_tpot_seconds: float | None = None
    avg_e2e_latency_seconds: float | None = None
    nixl_avg_transfer_time_seconds: float | None = None
    nixl_transfers_per_second: float = 0.0
    nixl_bytes_per_second: float = 0.0
    nixl_avg_bytes_per_transfer: float | None = None
    nixl_failed_transfers_total: float = 0.0
    nixl_failed_notifications_total: float = 0.0
    nixl_kv_expired_requests_total: float = 0.0
    engine_count: int = 0
    engines: list[EngineSnapshot] = Field(default_factory=list)


class RoleReplicaSnapshot(BaseModel):
    id: str
    role: str
    pod_index: int
    replica_index: int
    total_tokens_per_second: float
    total_requests_per_second: float
    total_running_requests: float
    total_waiting_requests: float
    avg_kv_cache_usage: float
    max_kv_cache_usage: float
    prefix_cache_hit_rate: float | None = None
    avg_nixl_transfer_time_seconds: float | None = None
    healthy_nodes: int
    total_nodes: int
    nodes: list[NodeSnapshot]


class PodSnapshot(BaseModel):
    id: str
    pod_index: int
    router_healthy: bool
    last_router_status: str
    router_url: str
    total_prefill_tokens_per_second: float
    total_decode_tokens_per_second: float
    total_requests_per_second: float
    max_kv_cache_usage: float
    prefill_prefix_cache_hit_rate: float | None = None
    prefill_cpu_prefix_cache_hit_rate: float | None = None
    cold_prefill_running_requests: float
    cold_prefill_waiting_requests: float
    prefill_waiting_requests: float
    decode_waiting_requests: float
    healthy_cold_prefill_nodes: int
    healthy_prefill_nodes: int
    healthy_decode_nodes: int
    cold_prefill_nodes: list[NodeSnapshot]
    prefill_replica: RoleReplicaSnapshot
    decode_replica: RoleReplicaSnapshot
    prefill_replicas: list[RoleReplicaSnapshot]
    decode_replicas: list[RoleReplicaSnapshot]
    prefill_nodes: list[NodeSnapshot]
    decode_nodes: list[NodeSnapshot]
    alerts: list[Alert]


class ClusterSummary(BaseModel):
    healthy_routers: int
    total_routers: int
    healthy_cold_prefill_nodes: int
    total_cold_prefill_nodes: int
    healthy_prefill_nodes: int
    total_prefill_nodes: int
    healthy_decode_nodes: int
    total_decode_nodes: int
    total_prefill_tokens_per_second: float
    total_decode_tokens_per_second: float
    total_requests_per_second: float
    total_running_requests: float
    total_waiting_requests: float
    max_kv_cache_usage: float


class Snapshot(BaseModel):
    job_id: int
    timestamp: float
    topology: Topology
    summary: ClusterSummary
    pods: list[PodSnapshot]
    alerts: list[Alert]
    history: dict[str, list[TimeSeriesPoint]]
    pod_history: dict[str, dict[str, list[TimeSeriesPoint]]]

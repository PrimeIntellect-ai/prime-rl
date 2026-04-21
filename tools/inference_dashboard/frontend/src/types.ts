export type Topology = {
  source: string;
  job_id: number;
  total_job_nodes: number;
  total_inference_nodes: number;
  num_replicas: number;
  num_cold_prefill_nodes_per_pod: number;
  num_prefill_nodes_per_pod: number;
  num_decode_nodes_per_pod: number;
  num_prefill_replicas_per_pod: number;
  num_decode_replicas_per_pod: number;
  hostnames: string[];
};

export type Alert = {
  severity: "warning" | "critical";
  scope: string;
  target: string;
  message: string;
};

export type TimeSeriesPoint = {
  timestamp: number;
  value: number;
};

export type EngineSnapshot = {
  id: string;
  engine_index: string;
  running_requests: number;
  waiting_requests: number;
  kv_cache_usage: number;
  cpu_kv_cache_usage: number | null;
  prefix_cache_hit_rate: number | null;
  cpu_prefix_cache_hit_rate: number | null;
  prompt_tokens_per_second: number;
  generation_tokens_per_second: number;
  requests_finished_per_second: number;
  avg_prefill_time_seconds: number | null;
  avg_decode_time_seconds: number | null;
  avg_queue_time_seconds: number | null;
  avg_ttft_seconds: number | null;
  avg_tpot_seconds: number | null;
  avg_e2e_latency_seconds: number | null;
  nixl_avg_transfer_time_seconds: number | null;
  nixl_transfers_per_second: number;
  nixl_bytes_per_second: number;
  nixl_avg_bytes_per_transfer: number | null;
  nixl_failed_transfers_total: number;
  nixl_failed_notifications_total: number;
  nixl_kv_expired_requests_total: number;
};

export type NodeSnapshot = {
  id: string;
  hostname: string;
  role: "cold_prefill" | "prefill" | "decode";
  pod_index: number;
  role_replica_index: number;
  scrape_ok: boolean;
  scrape_age_seconds: number;
  running_requests: number;
  waiting_requests: number;
  kv_cache_usage_mean: number;
  kv_cache_usage_max: number;
  cpu_kv_cache_usage_mean: number | null;
  cpu_kv_cache_usage_max: number | null;
  prefix_cache_hit_rate: number | null;
  cpu_prefix_cache_hit_rate: number | null;
  prompt_tokens_per_second: number;
  generation_tokens_per_second: number;
  requests_finished_per_second: number;
  avg_prefill_time_seconds: number | null;
  avg_decode_time_seconds: number | null;
  avg_queue_time_seconds: number | null;
  avg_ttft_seconds: number | null;
  avg_tpot_seconds: number | null;
  avg_e2e_latency_seconds: number | null;
  nixl_avg_transfer_time_seconds: number | null;
  nixl_transfers_per_second: number;
  nixl_bytes_per_second: number;
  nixl_avg_bytes_per_transfer: number | null;
  nixl_failed_transfers_total: number;
  nixl_failed_notifications_total: number;
  nixl_kv_expired_requests_total: number;
  engine_count: number;
  engines: EngineSnapshot[];
};

export type RoleReplicaSnapshot = {
  id: string;
  role: "prefill" | "decode";
  pod_index: number;
  replica_index: number;
  total_tokens_per_second: number;
  total_requests_per_second: number;
  total_running_requests: number;
  total_waiting_requests: number;
  avg_kv_cache_usage: number;
  max_kv_cache_usage: number;
  prefix_cache_hit_rate: number | null;
  avg_nixl_transfer_time_seconds: number | null;
  healthy_nodes: number;
  total_nodes: number;
  nodes: NodeSnapshot[];
};

export type PodSnapshot = {
  id: string;
  pod_index: number;
  router_healthy: boolean;
  last_router_status: string;
  router_url: string;
  total_prefill_tokens_per_second: number;
  total_decode_tokens_per_second: number;
  total_requests_per_second: number;
  max_kv_cache_usage: number;
  prefill_prefix_cache_hit_rate: number | null;
  prefill_cpu_prefix_cache_hit_rate: number | null;
  cold_prefill_running_requests: number;
  cold_prefill_waiting_requests: number;
  prefill_waiting_requests: number;
  decode_waiting_requests: number;
  healthy_cold_prefill_nodes: number;
  healthy_prefill_nodes: number;
  healthy_decode_nodes: number;
  cold_prefill_nodes: NodeSnapshot[];
  prefill_replica: RoleReplicaSnapshot;
  decode_replica: RoleReplicaSnapshot;
  prefill_replicas: RoleReplicaSnapshot[];
  decode_replicas: RoleReplicaSnapshot[];
  prefill_nodes: NodeSnapshot[];
  decode_nodes: NodeSnapshot[];
  alerts: Alert[];
};

export type Snapshot = {
  job_id: number;
  timestamp: number;
  topology: Topology;
  summary: {
    healthy_routers: number;
    total_routers: number;
    healthy_cold_prefill_nodes: number;
    total_cold_prefill_nodes: number;
    healthy_prefill_nodes: number;
    total_prefill_nodes: number;
    healthy_decode_nodes: number;
    total_decode_nodes: number;
    total_prefill_tokens_per_second: number;
    total_decode_tokens_per_second: number;
    total_requests_per_second: number;
    total_running_requests: number;
    total_waiting_requests: number;
    max_kv_cache_usage: number;
  };
  pods: PodSnapshot[];
  alerts: Alert[];
  history: Record<string, TimeSeriesPoint[]>;
  pod_history: Record<string, Record<string, TimeSeriesPoint[]>>;
};

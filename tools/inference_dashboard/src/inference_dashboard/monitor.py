from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass

import httpx

from inference_dashboard.metrics import NodeRollup, compute_rates, parse_prometheus_text
from inference_dashboard.models import (
    Alert,
    ClusterSummary,
    EngineSnapshot,
    NodeSnapshot,
    PodSnapshot,
    RoleReplicaSnapshot,
    Snapshot,
    TimeSeriesPoint,
    Topology,
)


HISTORY_KEYS = (
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "requests_per_second",
    "waiting_requests",
    "max_kv_cache_usage",
)

POD_HISTORY_KEYS = (
    "running_requests",
    "waiting_requests",
    "cold_prefill_running_requests",
    "cold_prefill_waiting_requests",
    "prefill_running_requests",
    "decode_running_requests",
    "prefill_waiting_requests",
    "decode_waiting_requests",
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "prefix_cache_hit_rate",
    "cpu_prefix_cache_hit_rate",
    "requests_per_second",
    "kv_cache_usage",
    "nixl_transfer_time_seconds",
)


@dataclass
class NodeState:
    current_timestamp: float | None = None
    current_rollup: NodeRollup | None = None
    previous_timestamp: float | None = None
    previous_rollup: NodeRollup | None = None
    scrape_error: str | None = None


class JobMonitor:
    def __init__(self, topology: Topology, history_window_seconds: int = 1800):
        self.topology = topology
        self.history_window_seconds = history_window_seconds
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=1.0))
        self._task: asyncio.Task[None] | None = None
        self._latest_snapshot: Snapshot | None = None
        self._node_state: dict[str, NodeState] = {}
        self._last_scrape_at: dict[str, float] = {}
        self._router_status: dict[str, tuple[bool, str]] = {}
        self._history: dict[str, deque[TimeSeriesPoint]] = {key: deque() for key in HISTORY_KEYS}
        self._pod_history: dict[str, dict[str, deque[TimeSeriesPoint]]] = {
            pod.id: {key: deque() for key in POD_HISTORY_KEYS} for pod in topology.pods
        }

    @property
    def latest_snapshot(self) -> Snapshot | None:
        return self._latest_snapshot

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def refresh_once(self) -> Snapshot:
        snapshot = await self._poll()
        self._latest_snapshot = snapshot
        return snapshot

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.http_client.aclose()

    async def _run(self) -> None:
        while True:
            snapshot = await self._poll()
            self._latest_snapshot = snapshot
            await asyncio.sleep(2.0)

    async def _poll(self) -> Snapshot:
        router_tasks = [self._poll_router_health(pod.id, pod.router_health_url) for pod in self.topology.pods]
        node_tasks = []
        for pod in self.topology.pods:
            for node in (*pod.cold_prefill_nodes, *pod.prefill_nodes, *pod.decode_nodes):
                node_tasks.append(self._poll_node_metrics(node.id, node.metrics_url))

        await asyncio.gather(*router_tasks, return_exceptions=True)
        await asyncio.gather(*node_tasks, return_exceptions=True)
        timestamp = time.time()

        pod_snapshots: list[PodSnapshot] = []
        cluster_alerts: list[Alert] = []
        all_nodes: list[NodeSnapshot] = []
        for pod in self.topology.pods:
            node_snapshots = [
                self._build_node_snapshot(node.id, timestamp)
                for node in pod.cold_prefill_nodes + pod.prefill_nodes + pod.decode_nodes
            ]
            cold_prefill_nodes = [node for node in node_snapshots if node.role == "cold_prefill"]
            prefill_nodes = [node for node in node_snapshots if node.role == "prefill"]
            decode_nodes = [node for node in node_snapshots if node.role == "decode"]
            prefill_replicas = self._build_role_replica_snapshots(pod.id, "prefill", pod.pod_index, prefill_nodes)
            decode_replicas = self._build_role_replica_snapshots(pod.id, "decode", pod.pod_index, decode_nodes)
            router_healthy, router_status = self._router_status.get(pod.id, (False, "unknown"))
            alerts = self._build_pod_alerts(pod.id, router_healthy, cold_prefill_nodes, prefill_nodes, decode_nodes)
            cluster_alerts.extend(alerts)
            all_nodes.extend(node_snapshots)
            pod_snapshots.append(
                PodSnapshot(
                    id=pod.id,
                    pod_index=pod.pod_index,
                    router_healthy=router_healthy,
                    last_router_status=router_status,
                    router_url=pod.router_url,
                    total_prefill_tokens_per_second=sum(node.prompt_tokens_per_second for node in prefill_nodes),
                    total_decode_tokens_per_second=sum(node.generation_tokens_per_second for node in decode_nodes),
                    total_requests_per_second=sum(node.requests_finished_per_second for node in node_snapshots),
                    max_kv_cache_usage=max((node.kv_cache_usage_max for node in node_snapshots), default=0.0),
                    prefill_prefix_cache_hit_rate=self._aggregate_prefix_cache_hit_rate(
                        [node.id for node in prefill_nodes]
                    ),
                    prefill_cpu_prefix_cache_hit_rate=self._aggregate_cpu_prefix_cache_hit_rate(
                        [node.id for node in prefill_nodes]
                    ),
                    cold_prefill_running_requests=sum(node.running_requests for node in cold_prefill_nodes),
                    cold_prefill_waiting_requests=sum(node.waiting_requests for node in cold_prefill_nodes),
                    prefill_waiting_requests=sum(node.waiting_requests for node in prefill_nodes),
                    decode_waiting_requests=sum(node.waiting_requests for node in decode_nodes),
                    healthy_cold_prefill_nodes=sum(1 for node in cold_prefill_nodes if node.scrape_ok),
                    healthy_prefill_nodes=sum(1 for node in prefill_nodes if node.scrape_ok),
                    healthy_decode_nodes=sum(1 for node in decode_nodes if node.scrape_ok),
                    cold_prefill_nodes=cold_prefill_nodes,
                    prefill_replica=self._build_role_replica_snapshot(
                        pod.id, "prefill", pod.pod_index, 0, prefill_nodes
                    ),
                    decode_replica=self._build_role_replica_snapshot(pod.id, "decode", pod.pod_index, 0, decode_nodes),
                    prefill_replicas=prefill_replicas,
                    decode_replicas=decode_replicas,
                    prefill_nodes=prefill_nodes,
                    decode_nodes=decode_nodes,
                    alerts=alerts,
                )
            )

        summary = ClusterSummary(
            healthy_routers=sum(1 for pod in pod_snapshots if pod.router_healthy),
            total_routers=len(pod_snapshots),
            healthy_cold_prefill_nodes=sum(1 for node in all_nodes if node.role == "cold_prefill" and node.scrape_ok),
            total_cold_prefill_nodes=sum(1 for node in all_nodes if node.role == "cold_prefill"),
            healthy_prefill_nodes=sum(1 for node in all_nodes if node.role == "prefill" and node.scrape_ok),
            total_prefill_nodes=sum(1 for node in all_nodes if node.role == "prefill"),
            healthy_decode_nodes=sum(1 for node in all_nodes if node.role == "decode" and node.scrape_ok),
            total_decode_nodes=sum(1 for node in all_nodes if node.role == "decode"),
            total_prefill_tokens_per_second=sum(
                node.prompt_tokens_per_second for node in all_nodes if node.role == "prefill"
            ),
            total_decode_tokens_per_second=sum(
                node.generation_tokens_per_second for node in all_nodes if node.role == "decode"
            ),
            total_requests_per_second=sum(node.requests_finished_per_second for node in all_nodes),
            total_running_requests=sum(node.running_requests for node in all_nodes),
            total_waiting_requests=sum(node.waiting_requests for node in all_nodes),
            max_kv_cache_usage=max((node.kv_cache_usage_max for node in all_nodes), default=0.0),
        )

        self._append_history(timestamp, summary)
        self._append_pod_history(timestamp, pod_snapshots)
        history = {key: list(points) for key, points in self._history.items()}
        pod_history = {
            pod_id: {key: list(points) for key, points in series.items()}
            for pod_id, series in self._pod_history.items()
        }
        return Snapshot(
            job_id=self.topology.job_id,
            timestamp=timestamp,
            topology=self.topology,
            summary=summary,
            pods=pod_snapshots,
            alerts=cluster_alerts,
            history=history,
            pod_history=pod_history,
        )

    async def _poll_router_health(self, pod_id: str, url: str) -> None:
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            self._router_status[pod_id] = (True, response.text.strip() or "healthy")
        except Exception as exc:
            self._router_status[pod_id] = (False, str(exc))

    async def _poll_node_metrics(self, node_id: str, url: str) -> None:
        state = self._node_state.setdefault(node_id, NodeState())
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            rollup = parse_prometheus_text(response.text)
            state.previous_timestamp = state.current_timestamp
            state.previous_rollup = state.current_rollup
            state.current_timestamp = time.time()
            state.current_rollup = rollup
            state.scrape_error = None
            self._last_scrape_at[node_id] = state.current_timestamp
        except Exception as exc:
            state.scrape_error = str(exc)

    def _build_node_snapshot(self, node_id: str, now: float) -> NodeSnapshot:
        endpoint = next(
            node
            for pod in self.topology.pods
            for node in (*pod.cold_prefill_nodes, *pod.prefill_nodes, *pod.decode_nodes)
            if node.id == node_id
        )
        state = self._node_state.get(node_id)
        if state is None or state.current_rollup is None or state.current_timestamp is None:
            return NodeSnapshot(
                id=endpoint.id,
                hostname=endpoint.hostname,
                role=endpoint.role,
                pod_index=endpoint.pod_index,
                role_replica_index=endpoint.role_replica_index,
                scrape_ok=False,
                scrape_age_seconds=9999.0,
            )

        dt = 0.0
        if state.previous_timestamp is not None:
            dt = state.current_timestamp - state.previous_timestamp
        rates = compute_rates(state.current_rollup, state.previous_rollup, dt)
        scrape_age = max(now - self._last_scrape_at.get(node_id, state.current_timestamp), 0.0)
        engines = self._build_engine_snapshots(
            node_id=node_id,
            current_rollup=state.current_rollup,
            previous_rollup=state.previous_rollup,
            dt=dt,
        )
        kv_values = [engine.kv_cache_usage for engine in engines]
        cpu_kv_values = [engine.cpu_kv_cache_usage for engine in engines if engine.cpu_kv_cache_usage is not None]

        return NodeSnapshot(
            id=endpoint.id,
            hostname=endpoint.hostname,
            role=endpoint.role,
            pod_index=endpoint.pod_index,
            role_replica_index=endpoint.role_replica_index,
            scrape_ok=state.scrape_error is None,
            scrape_age_seconds=scrape_age,
            running_requests=state.current_rollup.summed("running_requests"),
            waiting_requests=state.current_rollup.summed("waiting_requests"),
            kv_cache_usage_mean=sum(kv_values) / len(kv_values) if kv_values else 0.0,
            kv_cache_usage_max=max(kv_values, default=0.0),
            cpu_kv_cache_usage_mean=sum(cpu_kv_values) / len(cpu_kv_values) if cpu_kv_values else None,
            cpu_kv_cache_usage_max=max(cpu_kv_values, default=None),
            prefix_cache_hit_rate=rates.prefix_cache_hit_rate,
            cpu_prefix_cache_hit_rate=rates.cpu_prefix_cache_hit_rate,
            prompt_tokens_per_second=rates.prompt_tokens_per_second,
            generation_tokens_per_second=rates.generation_tokens_per_second,
            requests_finished_per_second=rates.requests_finished_per_second,
            avg_prefill_time_seconds=rates.avg_prefill_time_seconds,
            avg_decode_time_seconds=rates.avg_decode_time_seconds,
            avg_queue_time_seconds=rates.avg_queue_time_seconds,
            avg_ttft_seconds=rates.avg_ttft_seconds,
            avg_tpot_seconds=rates.avg_tpot_seconds,
            avg_e2e_latency_seconds=rates.avg_e2e_latency_seconds,
            nixl_avg_transfer_time_seconds=rates.nixl_avg_transfer_time_seconds,
            nixl_transfers_per_second=rates.nixl_transfers_per_second,
            nixl_bytes_per_second=rates.nixl_bytes_per_second,
            nixl_avg_bytes_per_transfer=rates.nixl_avg_bytes_per_transfer,
            nixl_failed_transfers_total=state.current_rollup.summed("nixl_failed_transfers_total"),
            nixl_failed_notifications_total=state.current_rollup.summed("nixl_failed_notifications_total"),
            nixl_kv_expired_requests_total=state.current_rollup.summed("nixl_kv_expired_requests_total"),
            engine_count=len(engines),
            engines=engines,
        )

    def _build_engine_snapshots(
        self,
        node_id: str,
        current_rollup: NodeRollup,
        previous_rollup: NodeRollup | None,
        dt: float,
    ) -> list[EngineSnapshot]:
        snapshots: list[EngineSnapshot] = []
        previous_engines = previous_rollup.engines if previous_rollup is not None else {}
        for engine_index, engine in sorted(current_rollup.engines.items()):
            current_engine_rollup = NodeRollup(engines={engine_index: engine})
            previous_engine = previous_engines.get(engine_index)
            previous_engine_rollup = None
            if previous_engine is not None:
                previous_engine_rollup = NodeRollup(engines={engine_index: previous_engine})
            rates = compute_rates(current_engine_rollup, previous_engine_rollup, dt)
            snapshots.append(
                EngineSnapshot(
                    id=f"{node_id}-engine-{engine_index}",
                    engine_index=engine_index,
                    running_requests=engine.running_requests,
                    waiting_requests=engine.waiting_requests,
                    kv_cache_usage=engine.kv_cache_usage_perc,
                    cpu_kv_cache_usage=engine.cpu_cache_usage_perc,
                    prefix_cache_hit_rate=rates.prefix_cache_hit_rate,
                    cpu_prefix_cache_hit_rate=rates.cpu_prefix_cache_hit_rate,
                    prompt_tokens_per_second=rates.prompt_tokens_per_second,
                    generation_tokens_per_second=rates.generation_tokens_per_second,
                    requests_finished_per_second=rates.requests_finished_per_second,
                    avg_prefill_time_seconds=rates.avg_prefill_time_seconds,
                    avg_decode_time_seconds=rates.avg_decode_time_seconds,
                    avg_queue_time_seconds=rates.avg_queue_time_seconds,
                    avg_ttft_seconds=rates.avg_ttft_seconds,
                    avg_tpot_seconds=rates.avg_tpot_seconds,
                    avg_e2e_latency_seconds=rates.avg_e2e_latency_seconds,
                    nixl_avg_transfer_time_seconds=rates.nixl_avg_transfer_time_seconds,
                    nixl_transfers_per_second=rates.nixl_transfers_per_second,
                    nixl_bytes_per_second=rates.nixl_bytes_per_second,
                    nixl_avg_bytes_per_transfer=rates.nixl_avg_bytes_per_transfer,
                    nixl_failed_transfers_total=engine.nixl_failed_transfers_total,
                    nixl_failed_notifications_total=engine.nixl_failed_notifications_total,
                    nixl_kv_expired_requests_total=engine.nixl_kv_expired_requests_total,
                )
            )
        return snapshots

    def _build_role_replica_snapshot(
        self,
        pod_id: str,
        role: str,
        pod_index: int,
        replica_index: int,
        nodes: list[NodeSnapshot],
    ) -> RoleReplicaSnapshot:
        total_tokens_per_second = sum(
            node.prompt_tokens_per_second if role == "prefill" else node.generation_tokens_per_second for node in nodes
        )
        kv_means = [node.kv_cache_usage_mean for node in nodes if node.scrape_ok]
        nixl_values = [
            node.nixl_avg_transfer_time_seconds for node in nodes if node.nixl_avg_transfer_time_seconds is not None
        ]
        return RoleReplicaSnapshot(
            id=f"{pod_id}-{role}-replica-{replica_index}",
            role=role,
            pod_index=pod_index,
            replica_index=replica_index,
            total_tokens_per_second=total_tokens_per_second,
            total_requests_per_second=sum(node.requests_finished_per_second for node in nodes),
            total_running_requests=sum(node.running_requests for node in nodes),
            total_waiting_requests=sum(node.waiting_requests for node in nodes),
            avg_kv_cache_usage=sum(kv_means) / len(kv_means) if kv_means else 0.0,
            max_kv_cache_usage=max((node.kv_cache_usage_max for node in nodes), default=0.0),
            prefix_cache_hit_rate=self._aggregate_prefix_cache_hit_rate([node.id for node in nodes])
            if role == "prefill"
            else None,
            avg_nixl_transfer_time_seconds=sum(nixl_values) / len(nixl_values) if nixl_values else None,
            healthy_nodes=sum(1 for node in nodes if node.scrape_ok),
            total_nodes=len(nodes),
            nodes=nodes,
        )

    def _aggregate_prefix_cache_hit_rate(self, node_ids: list[str]) -> float | None:
        current_hits = 0.0
        current_queries = 0.0
        previous_hits = 0.0
        previous_queries = 0.0
        has_previous = False
        for node_id in node_ids:
            state = self._node_state.get(node_id)
            if state is None or state.current_rollup is None or state.previous_rollup is None:
                continue
            has_previous = True
            current_hits += state.current_rollup.summed("prefix_cache_hits")
            current_queries += state.current_rollup.summed("prefix_cache_queries")
            previous_hits += state.previous_rollup.summed("prefix_cache_hits")
            previous_queries += state.previous_rollup.summed("prefix_cache_queries")
        if not has_previous:
            return None
        query_delta = current_queries - previous_queries
        hit_delta = current_hits - previous_hits
        if query_delta <= 0 or hit_delta < 0:
            return None
        return hit_delta / query_delta

    def _aggregate_cpu_prefix_cache_hit_rate(self, node_ids: list[str]) -> float | None:
        values: list[float] = []
        for node_id in node_ids:
            state = self._node_state.get(node_id)
            if state is None or state.current_rollup is None:
                continue
            values.extend(state.current_rollup.cpu_prefix_cache_hit_rate_values())
        if not values:
            return None
        return sum(values) / len(values)

    def _build_role_replica_snapshots(
        self,
        pod_id: str,
        role: str,
        pod_index: int,
        nodes: list[NodeSnapshot],
    ) -> list[RoleReplicaSnapshot]:
        grouped_nodes: dict[int, list[NodeSnapshot]] = {}
        for node in nodes:
            grouped_nodes.setdefault(node.role_replica_index, []).append(node)
        return [
            self._build_role_replica_snapshot(pod_id, role, pod_index, replica_index, replica_nodes)
            for replica_index, replica_nodes in sorted(grouped_nodes.items())
        ]

    def _build_pod_alerts(
        self,
        pod_id: str,
        router_healthy: bool,
        cold_prefill_nodes: list[NodeSnapshot],
        prefill_nodes: list[NodeSnapshot],
        decode_nodes: list[NodeSnapshot],
    ) -> list[Alert]:
        alerts: list[Alert] = []
        if not router_healthy:
            alerts.append(Alert(severity="critical", scope="pod", target=pod_id, message="router health check failed"))

        for node in cold_prefill_nodes + prefill_nodes + decode_nodes:
            if not node.scrape_ok or node.scrape_age_seconds > 6.0:
                alerts.append(
                    Alert(
                        severity="critical", scope="node", target=node.id, message="metrics scrape is stale or failing"
                    )
                )
            if node.kv_cache_usage_max >= 0.95:
                alerts.append(Alert(severity="critical", scope="node", target=node.id, message="KV cache above 95%"))
            elif node.kv_cache_usage_max >= 0.85:
                alerts.append(Alert(severity="warning", scope="node", target=node.id, message="KV cache above 85%"))
            if node.nixl_failed_transfers_total > 0:
                alerts.append(
                    Alert(severity="warning", scope="node", target=node.id, message="NIXL transfer failures detected")
                )

        prefill_waiting = sum(node.waiting_requests for node in prefill_nodes)
        decode_waiting = sum(node.waiting_requests for node in decode_nodes)
        decode_running = sum(node.running_requests for node in decode_nodes)
        prefill_running = sum(node.running_requests for node in prefill_nodes)
        if prefill_waiting > 0 and decode_running == 0:
            alerts.append(Alert(severity="warning", scope="pod", target=pod_id, message="prefill queue is backing up"))
        if decode_waiting > 0 and prefill_running == 0:
            alerts.append(Alert(severity="warning", scope="pod", target=pod_id, message="decode queue is backing up"))
        return alerts

    def _append_history(self, timestamp: float, summary: ClusterSummary) -> None:
        values = {
            "prefill_tokens_per_second": summary.total_prefill_tokens_per_second,
            "decode_tokens_per_second": summary.total_decode_tokens_per_second,
            "requests_per_second": summary.total_requests_per_second,
            "waiting_requests": summary.total_waiting_requests,
            "max_kv_cache_usage": summary.max_kv_cache_usage,
        }
        cutoff = timestamp - self.history_window_seconds
        for key, value in values.items():
            series = self._history[key]
            series.append(TimeSeriesPoint(timestamp=timestamp, value=value))
            while series and series[0].timestamp < cutoff:
                series.popleft()

    def _append_pod_history(self, timestamp: float, pods: list[PodSnapshot]) -> None:
        cutoff = timestamp - self.history_window_seconds
        for pod in pods:
            series = self._pod_history.setdefault(pod.id, {key: deque() for key in POD_HISTORY_KEYS})
            values = {
                "running_requests": pod.cold_prefill_running_requests
                + pod.prefill_replica.total_running_requests
                + pod.decode_replica.total_running_requests,
                "waiting_requests": pod.cold_prefill_waiting_requests + pod.prefill_waiting_requests + pod.decode_waiting_requests,
                "cold_prefill_running_requests": pod.cold_prefill_running_requests,
                "cold_prefill_waiting_requests": pod.cold_prefill_waiting_requests,
                "prefill_running_requests": pod.prefill_replica.total_running_requests,
                "decode_running_requests": pod.decode_replica.total_running_requests,
                "prefill_waiting_requests": pod.prefill_waiting_requests,
                "decode_waiting_requests": pod.decode_waiting_requests,
                "prefill_tokens_per_second": pod.total_prefill_tokens_per_second,
                "decode_tokens_per_second": pod.total_decode_tokens_per_second,
                "requests_per_second": pod.total_requests_per_second,
                "kv_cache_usage": pod.max_kv_cache_usage,
                "nixl_transfer_time_seconds": pod.decode_replica.avg_nixl_transfer_time_seconds or 0.0,
            }
            if pod.prefill_prefix_cache_hit_rate is not None:
                values["prefix_cache_hit_rate"] = pod.prefill_prefix_cache_hit_rate
            if pod.prefill_cpu_prefix_cache_hit_rate is not None:
                values["cpu_prefix_cache_hit_rate"] = pod.prefill_cpu_prefix_cache_hit_rate
            for node in pod.cold_prefill_nodes + pod.prefill_nodes + pod.decode_nodes:
                values[f"node_kv_cache_usage:{node.id}"] = node.kv_cache_usage_max
                if node.cpu_kv_cache_usage_max is not None:
                    values[f"node_cpu_kv_cache_usage:{node.id}"] = node.cpu_kv_cache_usage_max
            for key, value in values.items():
                bucket = series.setdefault(key, deque())
                bucket.append(TimeSeriesPoint(timestamp=timestamp, value=value))
                while bucket and bucket[0].timestamp < cutoff:
                    bucket.popleft()

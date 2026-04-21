from __future__ import annotations

import asyncio
from collections.abc import Sequence

from inference_dashboard.models import FromSlurmRequest, NodeEndpoint, PodEndpoint, Topology


async def run_command(*args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode().strip() or f"command failed: {' '.join(args)}")
    return stdout.decode().strip()


def build_topology(hostnames: Sequence[str], request: FromSlurmRequest) -> Topology:
    if request.num_prefill_nodes_per_pod % request.num_prefill_replicas_per_pod != 0:
        raise ValueError("prefill replicas per pod must evenly divide prefill nodes per pod")
    if request.num_decode_nodes_per_pod % request.num_decode_replicas_per_pod != 0:
        raise ValueError("decode replicas per pod must evenly divide decode nodes per pod")

    ordered_hostnames = list(hostnames)
    if len(ordered_hostnames) < request.inference_nodes:
        raise ValueError(
            f"job only has {len(ordered_hostnames)} nodes but {request.inference_nodes} inference nodes were requested"
        )

    inference_hosts = ordered_hostnames[: request.inference_nodes]
    pods: list[PodEndpoint] = []

    for pod_index in range(request.num_replicas):
        pod_base = pod_index * request.nodes_per_pod
        pod_hosts = inference_hosts[pod_base : pod_base + request.nodes_per_pod]
        router_host = pod_hosts[0]
        cold_prefill_nodes: list[NodeEndpoint] = []
        prefill_nodes: list[NodeEndpoint] = []
        decode_nodes: list[NodeEndpoint] = []

        for role_index in range(request.num_cold_prefill_nodes_per_pod):
            global_index = pod_base + role_index
            hostname = pod_hosts[role_index]
            cold_prefill_nodes.append(
                NodeEndpoint(
                    id=f"pod-{pod_index}-cold-prefill-n{role_index}",
                    hostname=hostname,
                    role="cold_prefill",
                    pod_index=pod_index,
                    role_replica_index=0,
                    global_index=global_index,
                    role_index=role_index,
                    role_replica_rank=role_index,
                    metrics_url=f"http://{hostname}:8100/metrics",
                    health_url=f"http://{hostname}:8100/health",
                    display_name=f"C{pod_index}N{role_index}",
                )
            )

        for role_index in range(request.num_prefill_nodes_per_pod):
            pod_role_index = request.num_cold_prefill_nodes_per_pod + role_index
            global_index = pod_base + pod_role_index
            hostname = pod_hosts[pod_role_index]
            role_replica_index = role_index // request.prefill_nodes_per_replica
            role_replica_rank = role_index % request.prefill_nodes_per_replica
            prefill_nodes.append(
                NodeEndpoint(
                    id=f"pod-{pod_index}-prefill-r{role_replica_index}-n{role_replica_rank}",
                    hostname=hostname,
                    role="prefill",
                    pod_index=pod_index,
                    role_replica_index=role_replica_index,
                    global_index=global_index,
                    role_index=role_index,
                    role_replica_rank=role_replica_rank,
                    metrics_url=f"http://{hostname}:8100/metrics",
                    health_url=f"http://{hostname}:8100/health",
                    display_name=f"P{pod_index}R{role_replica_index}N{role_replica_rank}",
                )
            )

        for role_index in range(request.num_decode_nodes_per_pod):
            pod_role_index = request.num_cold_prefill_nodes_per_pod + request.num_prefill_nodes_per_pod + role_index
            global_index = pod_base + pod_role_index
            hostname = pod_hosts[pod_role_index]
            role_replica_index = role_index // request.decode_nodes_per_replica
            role_replica_rank = role_index % request.decode_nodes_per_replica
            decode_nodes.append(
                NodeEndpoint(
                    id=f"pod-{pod_index}-decode-r{role_replica_index}-n{role_replica_rank}",
                    hostname=hostname,
                    role="decode",
                    pod_index=pod_index,
                    role_replica_index=role_replica_index,
                    global_index=global_index,
                    role_index=role_index,
                    role_replica_rank=role_replica_rank,
                    metrics_url=f"http://{hostname}:8200/metrics",
                    health_url=f"http://{hostname}:8200/health",
                    display_name=f"D{pod_index}R{role_replica_index}N{role_replica_rank}",
                )
            )

        pods.append(
            PodEndpoint(
                id=f"pod-{pod_index}",
                pod_index=pod_index,
                router_hostname=router_host,
                router_url=f"http://{router_host}:8000",
                router_health_url=f"http://{router_host}:8000/health",
                cold_prefill_nodes=cold_prefill_nodes,
                prefill_nodes=prefill_nodes,
                decode_nodes=decode_nodes,
            )
        )

    return Topology(
        job_id=request.job_id,
        total_job_nodes=len(ordered_hostnames),
        total_inference_nodes=request.inference_nodes,
        num_replicas=request.num_replicas,
        num_cold_prefill_nodes_per_pod=request.num_cold_prefill_nodes_per_pod,
        num_prefill_nodes_per_pod=request.num_prefill_nodes_per_pod,
        num_decode_nodes_per_pod=request.num_decode_nodes_per_pod,
        num_prefill_replicas_per_pod=request.num_prefill_replicas_per_pod,
        num_decode_replicas_per_pod=request.num_decode_replicas_per_pod,
        hostnames=ordered_hostnames,
        pods=pods,
    )


async def resolve_from_slurm(request: FromSlurmRequest) -> Topology:
    node_expression = await run_command("squeue", "-h", "-j", str(request.job_id), "-o", "%N")
    if not node_expression:
        raise RuntimeError(f"job {request.job_id} was not found in squeue")

    hostnames_output = await run_command("scontrol", "show", "hostnames", node_expression)
    hostnames = [line.strip() for line in hostnames_output.splitlines() if line.strip()]
    return build_topology(hostnames, request)

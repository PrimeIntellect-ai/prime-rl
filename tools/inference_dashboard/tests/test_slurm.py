from inference_dashboard.models import FromSlurmRequest
from inference_dashboard.slurm import build_topology


def test_build_topology_uses_first_inference_nodes_in_order():
    hostnames = [
        "node-1",
        "node-3",
        "node-4",
        "node-6",
        "node-7",
        "node-8",
        "node-9",
        "node-10",
        "node-11",
        "node-12",
    ]
    request = FromSlurmRequest(
        job_id=3781,
        num_prefill_nodes_per_pod=2,
        num_decode_nodes_per_pod=2,
        num_replicas=2,
    )

    topology = build_topology(hostnames, request)

    assert topology.total_inference_nodes == 8
    assert topology.hostnames[:8] == hostnames[:8]
    assert topology.pods[0].router_hostname == "node-1"
    assert [node.hostname for node in topology.pods[0].prefill_nodes] == ["node-1", "node-3"]
    assert [node.hostname for node in topology.pods[0].decode_nodes] == ["node-4", "node-6"]
    assert topology.pods[1].router_hostname == "node-7"
    assert [node.hostname for node in topology.pods[1].prefill_nodes] == ["node-7", "node-8"]
    assert [node.hostname for node in topology.pods[1].decode_nodes] == ["node-9", "node-10"]


def test_build_topology_assigns_role_replica_indexes_for_multi_replica_pods():
    request = FromSlurmRequest(
        job_id=3781,
        num_prefill_nodes_per_pod=4,
        num_decode_nodes_per_pod=4,
        num_prefill_replicas_per_pod=2,
        num_decode_replicas_per_pod=2,
        num_replicas=1,
    )

    topology = build_topology(
        ["node-1", "node-2", "node-3", "node-4", "node-5", "node-6", "node-7", "node-8"],
        request,
    )

    assert [node.role_replica_index for node in topology.pods[0].prefill_nodes] == [0, 0, 1, 1]
    assert [node.role_replica_rank for node in topology.pods[0].prefill_nodes] == [0, 1, 0, 1]
    assert [node.role_replica_index for node in topology.pods[0].decode_nodes] == [0, 0, 1, 1]
    assert [node.role_replica_rank for node in topology.pods[0].decode_nodes] == [0, 1, 0, 1]


def test_build_topology_places_cold_prefill_nodes_first():
    request = FromSlurmRequest(
        job_id=3781,
        num_cold_prefill_nodes_per_pod=2,
        num_prefill_nodes_per_pod=2,
        num_decode_nodes_per_pod=2,
        num_replicas=1,
    )

    topology = build_topology(
        ["node-1", "node-2", "node-3", "node-4", "node-5", "node-6"],
        request,
    )

    assert topology.total_inference_nodes == 6
    assert topology.pods[0].router_hostname == "node-1"
    assert [node.hostname for node in topology.pods[0].cold_prefill_nodes] == ["node-1", "node-2"]
    assert [node.hostname for node in topology.pods[0].prefill_nodes] == ["node-3", "node-4"]
    assert [node.hostname for node in topology.pods[0].decode_nodes] == ["node-5", "node-6"]


def test_build_topology_rejects_uneven_role_replica_division():
    request = FromSlurmRequest(
        job_id=3781,
        num_prefill_nodes_per_pod=3,
        num_decode_nodes_per_pod=2,
        num_prefill_replicas_per_pod=2,
        num_replicas=1,
    )

    try:
        build_topology(["node-1", "node-2", "node-3", "node-4", "node-5"], request)
    except ValueError as exc:
        assert "evenly divide" in str(exc)
    else:
        raise AssertionError("expected build_topology to reject uneven role replica division")

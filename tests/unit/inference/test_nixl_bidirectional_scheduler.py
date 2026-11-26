"""P/D KV transfer through vLLM's real scheduler: a declined or failed D->P read under bidirectional P/D, and a P->D
read whose lease may have lapsed.

The engines' kv_transfer_config comes from prime-rl's config builder, so these tests cover what the launcher deploys:
bidirectional NIXL with ``kv_load_failure_policy = "recompute"`` on prefill, plus the
``monkey_patch_nixl_bidirectional_single_pull`` backport. The scheduler helpers are trimmed from vLLM's
tests/v1/kv_connector/unit/utils.py (v0.34.0), which the vLLM wheel does not ship.
"""

import copy
import time

import pytest
import torch
from vllm import SamplingParams
from vllm.config import CacheConfig, DeviceConfig, KVTransferConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import pull_scheduler
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import NixlBaseConnectorScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import NixlConnectorMetadata
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.patches import (
    monkey_patch_nixl_bidirectional_single_pull,
    monkey_patch_nixl_prefill_lapsed_remote_kv_locally,
)

BLOCK_SIZE = 16
EOS_TOKEN_ID = 50256


def _prefill_kv_transfer_config() -> dict:
    """The prefill engine's kv_transfer_config as prime-rl builds it for bidirectional P/D (NIXL only)."""
    config = InferenceConfig(deployment={"type": "disaggregated", "bidirectional_kv_transfer": {}}, slurm={})
    return config.for_pd_role("prefill").build_kv_transfer_config()


def _scheduler(kv_transfer: dict, kv_role: str = "kv_producer") -> Scheduler:
    model_config = ModelConfig(model="facebook/opt-125m", dtype="float16", seed=42)
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=10000,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(block_size=BLOCK_SIZE, gpu_memory_utilization=0.9, enable_prefix_caching=True)
    cache_config.kv_cache_layout = "LBNHC"
    extra = {**kv_transfer["kv_connector_extra_config"], "kv_recompute_threshold": 0}
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        kv_transfer_config=KVTransferConfig(
            kv_connector=kv_transfer["kv_connector"],
            kv_role=kv_role,
            kv_connector_extra_config=extra,
            kv_load_failure_policy=kv_transfer.get("kv_load_failure_policy", "fail"),
        ),
        device_config=DeviceConfig("cpu"),
    )
    num_blocks = 10000
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32)
            )
        ],
    )
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )


def _turn2_request(request_id: int, num_tokens: int, num_remote_blocks: int = 3) -> Request:
    """A prefill-side request for the next turn, carrying the decode instance's retained blocks."""
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id=f"id-{request_id}",
        prompt_token_ids=[i * request_id for i in range(num_tokens)],
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    request.kv_transfer_params = {
        "do_remote_prefill": False,
        "do_remote_decode": True,
        "remote_block_ids": [list(range(num_remote_blocks))],
        "remote_num_tokens": num_remote_blocks * BLOCK_SIZE,
        "remote_engine_id": "decode-engine",
        "remote_request_id": f"decode-{request_id}",
        "remote_host": "decode-host",
        "remote_port": 5678,
    }
    return request


def _output(request: Request, sampled: bool = True) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[request.request_id],
        sampled_token_ids=[[0]] if sampled else [[]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
    )


@pytest.fixture
def single_pull_patch(monkeypatch):
    cls = pull_scheduler.NixlPullConnectorScheduler
    monkeypatch.setattr(cls, "get_num_new_matched_tokens", cls.get_num_new_matched_tokens)
    monkey_patch_nixl_bidirectional_single_pull()


def test_prefill_role_recomputes_failed_loads():
    config = InferenceConfig(deployment={"type": "disaggregated", "bidirectional_kv_transfer": {}}, slurm={})
    prefill = config.for_pd_role("prefill").build_kv_transfer_config()
    decode = config.for_pd_role("decode").build_kv_transfer_config()
    assert prefill["kv_load_failure_policy"] == "recompute"
    assert "kv_load_failure_policy" not in decode
    # With a second connector the policy sits on the top-level MultiConnector, where the scheduler reads it.
    mooncake = {"type": "mooncake", "cpu": {"num_bytes": 1000}}
    config = InferenceConfig(
        deployment={"type": "disaggregated", "bidirectional_kv_transfer": {}}, slurm={}, kv_cache_offload=mooncake
    )
    prefill = config.for_pd_role("prefill").build_kv_transfer_config()
    assert prefill["kv_connector"] == "MultiConnector" and prefill["kv_load_failure_policy"] == "recompute"
    assert KVTransferConfig(**prefill).kv_load_failure_policy == "recompute"
    # An explicit setting wins, and one-way P/D keeps vLLM's default.
    config = InferenceConfig(
        deployment={"type": "disaggregated", "bidirectional_kv_transfer": {}}, slurm={}, kv_load_failure_policy="fail"
    )
    assert config.for_pd_role("prefill").build_kv_transfer_config()["kv_load_failure_policy"] == "fail"
    config = InferenceConfig(deployment={"type": "disaggregated"}, slurm={})
    assert "kv_load_failure_policy" not in config.for_pd_role("prefill").build_kv_transfer_config()


def test_declined_decode_read_prefills_locally(single_pull_patch):
    """Prefill declines (or fails) the D->P read, e.g. because decode's blocks are about to expire. With prime-rl's
    prefill config the request then prefills locally and finishes, and no blocks leak."""
    kv_transfer = _prefill_kv_transfer_config()
    assert kv_transfer["kv_connector_extra_config"]["bidirectional_kv_xfer"]
    scheduler = _scheduler(kv_transfer)
    request = _turn2_request(507, int(BLOCK_SIZE * 2.5))
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.request_id in output.kv_connector_metadata.reqs_to_recv

    # The NIXL worker reports a declined read as finished and failed, with the request's blocks invalid.
    blocks = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
    runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    runner_output.kv_connector_output = KVConnectorOutput(
        finished_recving={request.request_id}, failed_recving={request.request_id}, invalid_block_ids=set(blocks)
    )
    scheduler.update_from_output(output, runner_output)

    # Scheduled locally, without a second read.
    output = scheduler.schedule()
    assert isinstance(output.kv_connector_metadata, NixlConnectorMetadata)
    assert request.request_id not in output.kv_connector_metadata.reqs_to_recv
    assert request.request_id in output.num_scheduled_tokens, f"{request.request_id} is {request.status.name}"

    for _ in range(4):
        scheduler.update_from_output(
            output, _output(request, sampled=request.num_computed_tokens >= request.num_tokens)
        )
        if request.is_finished():
            break
        output = scheduler.schedule()
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED

    # Prefill keeps the finished request's blocks for decode to read, then frees them once decode reports the read.
    output = scheduler.schedule()
    scheduler.update_from_output(output, EMPTY_MODEL_RUNNER_OUTPUT)
    output = scheduler.schedule()
    runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    runner_output.kv_connector_output = KVConnectorOutput(finished_sending={request.request_id})
    scheduler.update_from_output(output, runner_output)
    assert not scheduler.requests and not scheduler.running and len(scheduler.waiting) == 0
    assert all(block.ref_cnt == 0 for block in scheduler.kv_cache_manager.block_pool.blocks)


def test_keep_pause_does_not_wait_for_held_kv():
    """A prefill that finished for remote decode holds its blocks until decode pulls them. While the engine is paused
    with ``keep`` (PAUSED_ALL), that hold no longer counts as work, so the pause completes; it counts again on resume."""
    from vllm.v1.core.sched.interface import PauseState

    from prime_rl.inference.patches import monkey_patch_keep_pause_ignores_held_kv

    monkey_patch_keep_pause_ignores_held_kv()
    scheduler = _scheduler(_prefill_kv_transfer_config())
    request = _turn2_request(508, BLOCK_SIZE * 2, num_remote_blocks=0)
    request.kv_transfer_params = {"do_remote_decode": True, "do_remote_prefill": False}
    scheduler.add_request(request)

    output = scheduler.schedule()
    scheduler.update_from_output(output, _output(request))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    output = scheduler.schedule()  # reports the finished request to the workers
    scheduler.update_from_output(output, EMPTY_MODEL_RUNNER_OUTPUT)
    assert request.request_id in scheduler.requests, "prefill holds the blocks for decode"
    assert scheduler.has_requests()

    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    assert not scheduler.has_requests()
    assert request.request_id in scheduler.requests, "the blocks stay held while paused"

    scheduler.set_pause_state(PauseState.UNPAUSED)
    assert scheduler.has_requests()
    output = scheduler.schedule()
    runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    runner_output.kv_connector_output = KVConnectorOutput(finished_sending={request.request_id})
    scheduler.update_from_output(output, runner_output)
    assert not scheduler.has_requests()


def _decode_request(request_id: int, num_tokens: int) -> Request:
    """A decode-side request whose prompt a prefill engine computed, and holds the blocks of for it."""
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=8)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id=f"id-{request_id}",
        prompt_token_ids=[i * request_id for i in range(num_tokens)],
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": [list(range(num_tokens // BLOCK_SIZE))],
        "remote_engine_id": "prefill-engine",
        "remote_request_id": f"prefill-{request_id}",
        "remote_host": "prefill-host",
        "remote_port": 5600,
        "tp_size": 1,
    }
    return request


def test_decode_prefills_locally_once_the_kv_lease_may_have_lapsed(monkeypatch):
    """A decode engine that did not step, e.g. paused for a weight update while the prefill engines kept serving, sent
    no heartbeats, so the prefill engine may have reused the blocks it held: decode prefills that request itself
    instead of reading them. A request that arrived since still reads its prompt's KV from the prefill engine."""
    for cls, name in (
        (pull_scheduler.NixlPullConnectorScheduler, "get_num_new_matched_tokens"),
        (NixlBaseConnectorScheduler, "on_new_request"),
    ):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    monkey_patch_nixl_prefill_lapsed_remote_kv_locally()
    config = InferenceConfig(deployment={"type": "disaggregated"}, slurm={})
    scheduler = _scheduler(config.for_pd_role("decode").build_kv_transfer_config(), kv_role="kv_both")
    nixl = scheduler.connector.connector_scheduler
    now = [time.perf_counter()]
    monkeypatch.setattr(time, "perf_counter", lambda: now[0])

    paused_through, arrived_since = _decode_request(1, 3 * BLOCK_SIZE), _decode_request(2, 3 * BLOCK_SIZE)
    scheduler.add_request(paused_through)
    now[0] += nixl._kv_lease_duration
    scheduler.add_request(arrived_since)
    output = scheduler.schedule()

    assert paused_through.status == RequestStatus.RUNNING
    assert output.num_scheduled_tokens[paused_through.request_id] == 3 * BLOCK_SIZE
    assert paused_through.request_id not in nixl._heartbeat_req_engine
    assert arrived_since.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert arrived_since.request_id in nixl._heartbeat_req_engine

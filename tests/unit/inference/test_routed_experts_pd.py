from types import SimpleNamespace

import numpy as np

from prime_rl.inference.patches import (
    _emit_floors,
    monkey_patch_routed_experts_with_pd_connectors,
    skip_remote_prefill_partial_block,
)


def test_remote_prefill_rows_start_at_first_forward():
    from vllm.distributed.aux_output_connector.connector import AuxOutputSchedulerConnector

    monkey_patch_routed_experts_with_pd_connectors()
    connector = AuxOutputSchedulerConnector()
    request = SimpleNamespace(
        request_id="r",
        block_hashes=[b"a" * 32],
        num_tokens=6,
        num_output_tokens=0,
        num_computed_tokens=5,  # tokens [0, 5) were loaded from the prefill instance
        sampling_params=SimpleNamespace(routed_experts_prompt_start=0),
        kv_transfer_params={"do_remote_prefill": False, "remote_engine_id": "p0"},
        is_finished=lambda: False,
    )

    def emit_start():
        step = SimpleNamespace(num_scheduled_tokens={"r": request.num_tokens - request.num_computed_tokens})
        return connector.build_connector_meta(step, {"r": request}).requests["r"]

    assert emit_start() == 5
    # A preempted request recomputes from scratch but still emits from its first forward.
    connector.request_finished(request)
    request.num_computed_tokens = 0
    assert emit_start() == 5
    request.is_finished = lambda: True
    connector.request_finished(request)
    assert not _emit_floors(connector)


def test_remote_prefill_partial_first_block_is_not_buffered():
    from vllm.distributed.aux_output_connector.routed_experts import RoutedExpertsBuffer
    from vllm.distributed.aux_output_connector.worker import _WorkerRequestState

    rows = np.arange(12 * 3 * 2, dtype=np.uint8).reshape(12, 3, 2)
    buffer = RoutedExpertsBuffer(np.dtype("uint8"), (3, 2), 4, 1, 8, 2)
    state = _WorkerRequestState(emit_cursor=6)
    worker = SimpleNamespace(_buffer=buffer, _requests={"r": state})
    skip_remote_prefill_partial_block(worker)

    # The first forward starts mid-block at token 6; without the skip the buffer asserts.
    assert buffer.capture("r", 6, rows[6:7]) == []
    state.capture_cursor = 7
    assert buffer.capture("r", 7, rows[7:9]) == []
    state.capture_cursor = 9
    [(start, block)] = buffer.capture("r", 9, rows[9:12])
    assert start == 8
    np.testing.assert_array_equal(block, rows[8:12])

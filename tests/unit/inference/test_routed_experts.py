import pickle

import numpy as np
from vllm.distributed.aux_output_connector import worker as aux_worker
from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
from vllm.distributed.aux_output_connector.store import BlockObjectStore

from prime_rl.inference.patches import AUX_OUTPUT_STORE_HEADROOM, monkey_patch_aux_output_store
from prime_rl.inference.vllm.routed_experts import PackedAuxOutputs


def test_packed_aux_outputs_round_trip():
    rng = np.random.default_rng(0)
    outputs = {
        f"req-{i}": AuxRequestOutput(start, rng.integers(0, 128, (n, 3, 2), dtype=np.uint8))
        for i, (start, n) in enumerate([(7, 1), (0, 5), (40, 0), (12, 2)])
    }

    packed = pickle.loads(pickle.dumps(PackedAuxOutputs.pack(outputs), protocol=pickle.HIGHEST_PROTOCOL))

    assert "req-9" not in packed
    for request_id, output in outputs.items():
        assert request_id in packed
        assert packed[request_id].token_start == output.token_start
        np.testing.assert_array_equal(packed[request_id].rows, output.rows)


def test_aux_output_store_has_headroom(monkeypatch):
    monkeypatch.setattr(aux_worker, "BlockObjectStore", BlockObjectStore)
    monkey_patch_aux_output_store()
    store = aux_worker.BlockObjectStore(max_bytes=40, object_nbytes=1)
    assert isinstance(store, BlockObjectStore)
    assert store.num_slots == AUX_OUTPUT_STORE_HEADROOM * 40

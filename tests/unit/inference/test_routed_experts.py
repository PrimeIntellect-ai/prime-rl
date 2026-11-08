import pickle
import random

import numpy as np
from vllm.distributed.aux_output_connector import worker as aux_worker
from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
from vllm.distributed.aux_output_connector.store import BlockObject, BlockObjectStore, BlockObjectStoreError

from prime_rl.inference.patches import monkey_patch_aux_output_store
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


def test_aux_output_store_eviction_matches_upstream(monkeypatch):
    upstream_evict = BlockObjectStore._evict_to_fit
    monkeypatch.setattr(BlockObjectStore, "_evict_to_fit", upstream_evict)
    monkeypatch.setattr(aux_worker, "BlockObjectStore", BlockObjectStore)
    monkey_patch_aux_output_store()
    upstream = type("Upstream", (BlockObjectStore,), {"_evict_to_fit": upstream_evict})
    rng = random.Random(0)
    for _ in range(50):
        stores = [upstream(max_bytes=40, object_nbytes=1), BlockObjectStore(max_bytes=40, object_nbytes=1)]
        references: dict[str, int] = {}
        for _ in range(200):
            objects = [BlockObject(f"k{rng.randrange(120)}", b"x") for _ in range(rng.randrange(6))]
            retain = [f"k{rng.randrange(120)}" for _ in range(rng.randrange(4))]
            release = [key for key, count in references.items() if count and rng.random() < 0.15]
            for key in retain:
                references[key] = references.get(key, 0) + 1
            for key in release:
                references[key] -= 1
            failed = []
            for store in stores:
                try:
                    store.put(objects, retain_keys=retain, release_keys=release)
                    failed.append(False)
                except BlockObjectStoreError:
                    failed.append(True)
            a, b = stores
            assert failed[0] == failed[1]
            assert set(a._lru) == set(b._lru)
            # Only the order of unreferenced keys decides future evictions.
            assert [k for k in a._lru if k not in a._references] == [k for k in b._lru if k not in b._references]

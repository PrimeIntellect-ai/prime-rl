from types import SimpleNamespace

import pytest

BLOCK = 256


class FakeLookupClient:
    """Stands in for LookupKeyClient: counts submissions, returns ``result`` once ``ready``."""

    def __init__(self, hit_length):
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import MooncakeLookupResult

        self.result = MooncakeLookupResult(hit_length)
        self.ready = True
        self.submitted = []
        self.pending = set()
        self.discarded = []

    def lookup(self, req_id, num_tokens, block_hashes, non_block=False):
        if req_id not in self.pending:
            self.submitted.append((req_id, num_tokens))
            self.pending.add(req_id)
        if non_block and not self.ready:
            return None
        self.pending.discard(req_id)
        return self.result

    def discard(self, req_id):
        self.pending.discard(req_id)
        self.discarded.append(req_id)


def _scheduler(hit_length, min_gain=None, ttl=None, timeout=None, lookup_async=True):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler

    from prime_rl.inference.patches import monkey_patch_mooncake_lookup_economy

    monkey_patch_mooncake_lookup_economy()
    s = MooncakeStoreScheduler.__new__(MooncakeStoreScheduler)
    s.enable_lookup = True
    s.lookup_async = lookup_async
    s.load_async = True
    s.load_specs = {}
    s.enable_partial_hash_hits = False
    s._block_size = BLOCK
    s._hash_block_size = 64
    s._store_coord = SimpleNamespace(align_lookup_length=lambda n: n // BLOCK * BLOCK)
    s.client = FakeLookupClient(hit_length)
    # What the patched __init__ derives from kv_connector_extra_config.
    s._prime_rl_min_gain = None if min_gain is None else max(min_gain, BLOCK)
    s._prime_rl_memo_ttl_s = ttl
    s._prime_rl_timeout_s = timeout
    s._prime_rl_economy = any(x is not None for x in (min_gain, ttl, timeout))
    s._prime_rl_memo = {}
    s._prime_rl_submitted = {}
    return s


def _request(num_tokens, rid="r"):
    return SimpleNamespace(request_id=rid, num_tokens=num_tokens, block_hashes=[b"h"] * (num_tokens // 64))


def test_off_by_default_looks_up_every_call():
    s = _scheduler(hit_length=768)
    assert s.get_num_new_matched_tokens(_request(1000), 512) == (256, True)
    assert s.get_num_new_matched_tokens(_request(1000), 512) == (256, True)
    assert len(s.client.submitted) == 2


@pytest.mark.parametrize(
    ("num_tokens", "local", "min_gain", "looked_up"),
    [
        (1000, 768, 1, False),  # align(999) = 768: nothing past the local hit can load (floor = block size)
        (1000, 512, 1, True),  # one block could load
        (1025, 768, 1, True),  # align(1024) = 1024: one block
        (1024, 768, 1, False),  # a full-length hit is re-derived one token short: 768
        (10_000, 7_936, 4096, False),  # 1,792 tokens < min gain: recompute instead
        (10_000, 4_096, 4096, True),
    ],
)
def test_min_gain_skips_lookups_that_cannot_pay(num_tokens, local, min_gain, looked_up):
    s = _scheduler(hit_length=0, min_gain=min_gain)
    assert s.get_num_new_matched_tokens(_request(num_tokens), local) == (0, False)
    assert bool(s.client.submitted) == looked_up


def test_memo_reuses_a_lookup_until_admission():
    s = _scheduler(hit_length=768, ttl=60.0)
    req = _request(1000)
    assert s.get_num_new_matched_tokens(req, 256) == (512, True)
    # Deferred steps: no new lookup, and the load is re-derived from the current local hit.
    assert s.get_num_new_matched_tokens(req, 512) == (256, True)
    assert s.load_specs["r"].vllm_cached_tokens == 512
    assert s.get_num_new_matched_tokens(req, 768) == (0, False)
    assert len(s.client.submitted) == 1
    # A longer request (new turn under the same id) looks up again.
    assert s.get_num_new_matched_tokens(_request(1300), 256) == (512, True)
    assert len(s.client.submitted) == 2
    # Admission drops the memo.
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler

    s._unfinished_requests, s._unfinished_request_ids = {}, set()
    MooncakeStoreScheduler.update_state_after_alloc(s, SimpleNamespace(request_id="r"), None, 0)
    assert "r" not in s._prime_rl_memo


def test_memo_expires(monkeypatch):
    import prime_rl.inference.patches as patches

    now = [100.0]
    monkeypatch.setattr(patches.time, "monotonic", lambda: now[0])
    s = _scheduler(hit_length=768, ttl=5.0)
    s.get_num_new_matched_tokens(_request(1000), 0)
    now[0] += 4.0
    s.get_num_new_matched_tokens(_request(1000), 0)
    assert len(s.client.submitted) == 1
    now[0] += 2.0
    s.get_num_new_matched_tokens(_request(1000), 0)
    assert len(s.client.submitted) == 2


def test_timeout_admits_on_local_hit(monkeypatch):
    import prime_rl.inference.patches as patches

    now = [100.0]
    monkeypatch.setattr(patches.time, "monotonic", lambda: now[0])
    s = _scheduler(hit_length=768, timeout=2.0)
    s.client.ready = False
    req = _request(1000)
    assert s.get_num_new_matched_tokens(req, 0) == (None, False)
    now[0] += 1.0
    assert s.get_num_new_matched_tokens(req, 0) == (None, False)
    now[0] += 1.5
    assert s.get_num_new_matched_tokens(req, 0) == (0, False)
    assert s.client.discarded == ["r"]
    # Given up: no resubmission while it waits, even once the store answers.
    s.client.ready = True
    assert s.get_num_new_matched_tokens(req, 0) == (0, False)
    assert len(s.client.submitted) == 1


def test_finished_requests_are_forgotten():
    s = _scheduler(hit_length=768, ttl=60.0, timeout=2.0)
    s.client.ready = False
    s.get_num_new_matched_tokens(_request(1000, "a"), 0)
    s.client.ready = True
    s.get_num_new_matched_tokens(_request(1000, "b"), 0)
    assert set(s._prime_rl_submitted) == {"a"} and set(s._prime_rl_memo) == {"b"}
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler

    # The cleanup runs before upstream's body, which needs a full scheduler state and fails on this stub.
    with pytest.raises(AttributeError):
        MooncakeStoreScheduler.build_connector_meta(s, SimpleNamespace(finished_req_ids={"a", "b"}))
    assert not s._prime_rl_submitted and not s._prime_rl_memo
    assert "a" in s.client.discarded


def test_config_fields_reach_extra_config():
    from prime_rl.configs.inference import InferenceConfig

    off = InferenceConfig(kv_cache_offload={"type": "mooncake", "cpu": {"num_bytes": 1000}})
    assert off.build_kv_transfer_config()["kv_connector_extra_config"] == {}
    assert off.build_kv_transfer_config()["kv_load_failure_policy"] == "recompute"
    on = InferenceConfig(
        kv_cache_offload={
            "type": "mooncake",
            "cpu": {"num_bytes": 1000},
            "lookup_async": True,
            "save_decode_cache": True,
            "min_lookup_gain_tokens": 4096,
            "lookup_memo_ttl_s": 10,
            "lookup_timeout_s": 2,
        }
    )
    assert on.build_kv_transfer_config()["kv_connector_extra_config"] == {
        "lookup_async": True,
        "save_decode_cache": True,
        "prime_rl_min_lookup_gain_tokens": 4096,
        "prime_rl_lookup_memo_ttl_s": 10.0,
        "prime_rl_lookup_timeout_s": 2.0,
    }

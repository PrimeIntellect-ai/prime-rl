from types import SimpleNamespace

from vllm.model_executor.layers.fused_moe import routed_experts_capturer
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MLAAttentionSpec

from prime_rl.inference.patches import monkey_patch_routed_experts_skip_host_kv_groups


def test_routed_experts_slots_skip_hisparse_host_group():
    monkey_patch_routed_experts_skip_host_kv_groups()
    spec = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=576, dtype="bfloat16")
    groups = [
        KVCacheGroupSpec(["source"], spec, host_resident=True),
        KVCacheGroupSpec(["indexer"], spec),
    ]
    kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    assert routed_experts_capturer.get_routed_experts_attn_gid(kv_cache_config) == 1

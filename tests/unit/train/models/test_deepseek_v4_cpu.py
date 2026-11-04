from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config


def test_deepseek_v4_config_translates_legacy_compress_ratios():
    """Real checkpoints ship the V3-flavoured legacy `compress_ratios`/`num_hash_layers` schema
    instead of `layer_types`, which is what prime-rl's model code reads, so the
    config has to translate between them. Loading the real checkpoint without this built the
    wrong per-layer attention schedule outright.
    """
    config = DeepseekV4Config(num_hidden_layers=6, compress_ratios=[0, 0, 4, 128, 4, 128], num_hash_layers=2)

    assert config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert config.num_hash_layers == 2

from prime_rl.trainer.models.qwen3_5.converting_qwen3_5 import is_hf_state_dict, is_prime_state_dict


def test_fp8_release_is_hf_format_only():
    """Dense HF and PrimeRL names coincide, so a dense release passes as PrimeRL format and loads as
    is. Its `-FP8` twin must pass as HF only, with or without the `mtp.*` keys official releases
    ship, since the HF->PrimeRL conversion is what dequantizes it."""
    layer = "model.language_model.layers.0"
    dense = dict.fromkeys([f"{layer}.linear_attn.A_log", f"{layer}.mlp.up_proj.weight"])
    fp8 = dense | dict.fromkeys([f"{layer}.mlp.up_proj.weight_scale_inv"])

    assert is_prime_state_dict(dense)
    for snapshot in (fp8, fp8 | dict.fromkeys(["mtp.fc.weight"])):
        assert is_hf_state_dict(snapshot)
        assert not is_prime_state_dict(snapshot)

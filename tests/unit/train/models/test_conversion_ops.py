import torch

from prime_rl.trainer.models.conversion_ops import Copy, Rename, apply_hf_to_prime, apply_prime_to_hf


def test_copy_creates_the_target_on_the_way_to_prime_and_drops_it_on_the_way_back():
    ops = [Rename("embed.weight", "model.embed_tokens.weight"), Copy("model.embed_tokens.weight", "lm_head.weight")]
    embed = torch.randn(4, 3)

    prime = apply_hf_to_prime({"embed.weight": embed.clone()}, ops)
    assert set(prime) == {"model.embed_tokens.weight", "lm_head.weight"}
    assert torch.equal(prime["lm_head.weight"], embed)
    assert (
        prime["lm_head.weight"].untyped_storage().data_ptr()
        != prime["model.embed_tokens.weight"].untyped_storage().data_ptr()
    )

    hf = apply_prime_to_hf(prime, ops)
    assert set(hf) == {"embed.weight"}
    assert torch.equal(hf["embed.weight"], embed)


def test_copy_without_its_source_or_target_does_nothing():
    op = Copy("source", "target")
    state_dict = {"other": torch.zeros(2)}
    op.hf_to_prime(state_dict)
    op.prime_to_hf(state_dict)
    assert set(state_dict) == {"other"}

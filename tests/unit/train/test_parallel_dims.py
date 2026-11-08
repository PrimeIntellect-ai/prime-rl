from prime_rl.trainer.parallel_dims import auto_ep


def test_auto_ep_divides_island_and_experts():
    assert auto_ep(fsdp_island_size=6, num_experts=128, cp=1) == 2
    assert auto_ep(fsdp_island_size=16, num_experts=128, cp=1) == 8
    assert auto_ep(fsdp_island_size=8, num_experts=128, cp=4) == 8
    assert auto_ep(fsdp_island_size=6, num_experts=128, cp=3) == 1

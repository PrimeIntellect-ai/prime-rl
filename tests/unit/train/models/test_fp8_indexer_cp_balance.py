from prime_rl.trainer.models.kernels.fp8_indexer import cp_balance_plan


def test_cp_balance_plan_moves_trailing_chunks_from_heavy_to_light_rank():
    light, heavy = [1, 2, 3, 4], [13, 14, 15, 16]
    assert cp_balance_plan(heavy, light) == (2, 0)
    assert cp_balance_plan(light, heavy) == (0, 2)


def test_cp_balance_plan_keeps_balanced_ranks():
    assert cp_balance_plan([5, 5], [5, 5]) == (0, 0)
    assert cp_balance_plan([6, 4], [5, 5]) == (0, 0)

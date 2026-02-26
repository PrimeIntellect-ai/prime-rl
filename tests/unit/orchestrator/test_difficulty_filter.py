from prime_rl.orchestrator.difficulty_filter import get_training_rollout_mask


def test_get_training_rollout_mask_filters_zero_advantages_when_verification_enabled():
    advantages = [0.2, 0.0, -1e-9, -0.4]
    keep_mask = get_training_rollout_mask(advantages=advantages, skip_verification=False)
    assert keep_mask == [True, False, False, True]


def test_get_training_rollout_mask_keeps_all_when_skip_verification_enabled():
    advantages = [0.0, 0.0, 0.0]
    keep_mask = get_training_rollout_mask(advantages=advantages, skip_verification=True)
    assert keep_mask == [True, True, True]


def test_get_training_rollout_mask_respects_custom_epsilon():
    advantages = [1e-4, 5e-5]
    keep_mask = get_training_rollout_mask(
        advantages=advantages,
        skip_verification=False,
        zero_advantage_eps=1e-4,
    )
    assert keep_mask == [False, False]

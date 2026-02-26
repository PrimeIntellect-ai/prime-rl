"""Unit tests for eval scheduling logic - specifically the range check
that detects when ckpt_step jumps over eval interval boundaries."""

from prime_rl.orchestrator.eval_utils import compute_eval_ckpt_step


class TestEvalSchedulingRangeCheck:
    """Tests for the range-based eval scheduling that handles ckpt_step jumps."""

    def test_exact_hit(self):
        """ckpt_step lands exactly on interval - should trigger."""
        result = compute_eval_ckpt_step(ckpt_step=25, prev_ckpt_step=24, last_eval_step=0, interval=25)
        assert result == 25

    def test_jump_over_interval(self):
        """ckpt_step jumps from 24 to 26, skipping interval step 25 - should still trigger."""
        result = compute_eval_ckpt_step(ckpt_step=26, prev_ckpt_step=24, last_eval_step=0, interval=25)
        assert result == 25

    def test_no_interval_crossed(self):
        """ckpt_step advances within an interval - should not trigger."""
        result = compute_eval_ckpt_step(ckpt_step=23, prev_ckpt_step=22, last_eval_step=0, interval=25)
        assert result is None

    def test_base_model_eval_at_step_0(self):
        """Step 0 with eval_base_model=True - should trigger."""
        result = compute_eval_ckpt_step(
            ckpt_step=0, prev_ckpt_step=-1, last_eval_step=-1, interval=25, eval_base_model=True
        )
        assert result == 0

    def test_base_model_eval_disabled(self):
        """Step 0 with eval_base_model=False - should not trigger."""
        result = compute_eval_ckpt_step(
            ckpt_step=0, prev_ckpt_step=-1, last_eval_step=-1, interval=25, eval_base_model=False
        )
        assert result is None

    def test_no_double_eval(self):
        """Same ckpt_step as last_eval_step - should not trigger again."""
        result = compute_eval_ckpt_step(ckpt_step=25, prev_ckpt_step=24, last_eval_step=25, interval=25)
        assert result is None

    def test_no_change_in_ckpt_step(self):
        """ckpt_step unchanged - should not trigger."""
        result = compute_eval_ckpt_step(ckpt_step=25, prev_ckpt_step=25, last_eval_step=0, interval=25)
        assert result is None

    def test_multiple_intervals_crossed(self):
        """ckpt_step jumps from 24 to 76 - should trigger at 75 (highest interval in range)."""
        result = compute_eval_ckpt_step(ckpt_step=76, prev_ckpt_step=24, last_eval_step=0, interval=25)
        assert result == 75

    def test_second_interval(self):
        """After evaluating at step 25, ckpt_step reaches 50 - should trigger."""
        result = compute_eval_ckpt_step(ckpt_step=50, prev_ckpt_step=49, last_eval_step=25, interval=25)
        assert result == 50

    def test_jump_across_second_interval(self):
        """After step 25 eval (last_eval=25), ckpt_step jumps from 48 to 51 - should trigger at 50."""
        result = compute_eval_ckpt_step(ckpt_step=51, prev_ckpt_step=48, last_eval_step=25, interval=25)
        assert result == 50

    def test_production_scenario_step25_skipped(self):
        """Reproduces the actual bug from run c14miuyha2yhxkw1z3eqgyub.

        Old code: ckpt_step=26, 26 % 25 != 0 -> eval skipped forever.
        New code: highest interval in (24, 26] = 25 -> eval triggers.
        """
        result = compute_eval_ckpt_step(ckpt_step=26, prev_ckpt_step=24, last_eval_step=0, interval=25)
        assert result == 25

    def test_production_scenario_step50_exact(self):
        """Step 50 lands exactly - normal case."""
        result = compute_eval_ckpt_step(ckpt_step=50, prev_ckpt_step=49, last_eval_step=26, interval=25)
        assert result == 50

    def test_simulate_full_run(self):
        """Simulate a sequence of ckpt_step values and verify evals trigger correctly."""
        ckpt_steps = [0, 0, 3, 5, 10, 15, 20, 24, 26, 30, 35, 40, 48, 51, 60, 70, 74, 76]
        interval = 25
        last_eval_step = -1
        prev_ckpt_step = -1
        eval_triggered_at = []

        for ckpt_step in ckpt_steps:
            result = compute_eval_ckpt_step(ckpt_step, prev_ckpt_step, last_eval_step, interval)
            if result is not None:
                eval_triggered_at.append(result)
                last_eval_step = ckpt_step
            prev_ckpt_step = ckpt_step

        # Should trigger at 0 (base), 25 (via jump 24->26), 50 (via jump 48->51), 75 (via jump 74->76)
        assert eval_triggered_at == [0, 25, 50, 75]

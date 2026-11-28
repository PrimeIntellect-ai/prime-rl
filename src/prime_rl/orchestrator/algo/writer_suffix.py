"""Writer-only recurrent memory learning from subsequent task outcomes."""

from prime_rl.orchestrator.algo.rime import RIMEAlgorithm
from prime_rl.orchestrator.algo.rime_credit import assign_suffix_credit


class WriterSuffixAlgorithm(RIMEAlgorithm):
    writer_reward_metric = "rime/writer_future_return"

    def credit(self, task, exploration, well_formed):
        return assign_suffix_credit(task, exploration, well_formed)

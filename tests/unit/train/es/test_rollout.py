from datasets import Dataset

from prime_rl.configs.orchestrator import TrainSamplingConfig
from prime_rl.trainer.es.rollout import sample_examples


class DummyEnv:
    def get_dataset(self, seed: int):
        return Dataset.from_list([{"idx": 0}, {"idx": 1}])


def test_sample_examples_uses_replacement_when_count_exceeds_dataset():
    rows = sample_examples(DummyEnv(), count=8, seed=123)

    assert len(rows) == 8
    assert {row["idx"] for row in rows}.issubset({0, 1})


def test_train_sampling_config_can_disable_logprobs():
    args = TrainSamplingConfig(logprobs=False, max_completion_tokens=64).to_sampling_args()

    assert args["logprobs"] is False
    assert args["max_completion_tokens"] == 64

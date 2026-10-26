from types import SimpleNamespace

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.types import WeightVersionSpan

from prime_rl.orchestrator.utils import Staleness, rollout_staleness


def test_rollout_staleness_is_max_token_staleness():
    spans = [
        WeightVersionSpan(version="default", start=0, end=4),
        WeightVersionSpan(version="7", start=4, end=10),
        WeightVersionSpan(version="9", start=10, end=12),
    ]
    trace = SimpleNamespace(
        nodes=[
            SimpleNamespace(mask=np.array([False, True]), weight_versions=None),
            SimpleNamespace(mask=np.array([False] + [True] * 12), weight_versions=spans),
        ]
    )
    work = vf.TrainWorkInfo(step=7, policy=vf.PolicySpan(start=6, end=9))
    episode = SimpleNamespace(run=vf.TrainRunInfo(id="run", work=work), traces=[trace])

    # Batch 11 trains on v10. Unlabelled tokens take the dispatch version v6:
    # 5 tokens of staleness 4, 6 of staleness 3, and 2 of staleness 1. The
    # rollout completed at v9, so 1 of its 4 steps of staleness is in-queue.
    assert rollout_staleness(episode, 11) == Staleness(rollout=4, in_flight=3, in_queue=1, token_sum=40, token_count=13)

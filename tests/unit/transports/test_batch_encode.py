import pytest

from prime_rl.transports.batch.filesystem import FileSystemBatchSender
from prime_rl.transports.batch.types import MicroBatch


@pytest.mark.parametrize("n", [0, 15, 16, 2**16 - 1, 2**16])
def test_encode_matches_whole_list_encode(tmp_path, n):
    sender = FileSystemBatchSender(tmp_path, data_world_size=1)
    micro_batch = MicroBatch([1, 2], [True, False], [0.5, 0.0], [-0.1, -0.2], [0, 1], [2], [1.0, 1.0], ["env"], [2])
    micro_batches = [micro_batch] * n
    assert sender.encode(micro_batches) == sender.encoder.encode(micro_batches)

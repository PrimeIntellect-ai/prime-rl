"""Replay one saved trainer batch without an orchestrator or inference workers."""

from functools import partial
import hashlib
import json
from pathlib import Path
import sys

import msgspec

from prime_rl.trainer.rl.data import DataLoader
from prime_rl.trainer.rl.trainer_trace_probe import run_with_diagnostics
from prime_rl.trainer.world import get_world
from prime_rl.transports.batch.types import MicroBatch
from prime_rl.utils.logger import get_logger


class ReplayDataLoader(DataLoader):
    def __init__(self, fake_config, seq_len, dp_world_size, *, batch_dir):
        self.world = get_world()
        assert self.world.world_size == dp_world_size == 16
        assert seq_len == 131072
        self.batch_dir = Path(batch_dir)
        self.used = False

    def wait_for_batch(self):
        assert not self.used, "This diagnostic replays exactly one saved batch"

    def get_batch(self):
        assert not self.used
        self.used = True
        manifest = json.loads((self.batch_dir / "replay-summary.json").read_text())
        record = next(item for item in manifest["ranks"] if item["rank"] == self.world.rank)
        payload = (self.batch_dir / f"rank-{self.world.rank}.msgpack").read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["sha256"], "Replay batch checksum mismatch"
        batches = msgspec.msgpack.decode(payload, type=list[MicroBatch])
        assert len(batches) == record["microbatches"] == 14
        assert [len(batch.input_ids) for batch in batches] == record["lengths"]
        get_logger().info(f"Replaying saved production batch: rank={self.world.rank}, microbatches={len(batches)}")
        return [self._micro_batch_to_tensor(batch) for batch in batches]


def run(train, config):
    assert config.data.fake is not None and config.max_steps == 1
    assert config.model.cp == 1
    module = sys.modules[train.__module__]
    original = module.FakeDataLoader
    module.FakeDataLoader = partial(ReplayDataLoader, batch_dir=config.env_vars["PRIME_REPLAY_BATCH_DIR"])
    try:
        return run_with_diagnostics(train, config)
    finally:
        module.FakeDataLoader = original

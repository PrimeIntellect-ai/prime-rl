import asyncio
from pathlib import Path

from prime_rl.orchestrator.draft import DraftLoader


class FakeAdminPlane:
    def __init__(self, broken: set[Path], split: set[Path]):
        self.broken, self.split = broken, split
        self.loaded: list[Path] = []

    async def update_draft_weights(self, weight_dir: Path) -> None:
        self.loaded.append(weight_dir)
        if weight_dir in self.broken:
            raise RuntimeError("load failed")

    async def draft_digests(self) -> list[str]:
        return ["a", "b"] if self.loaded[-1] in self.split else ["a", "a"]


def test_failed_draft_loads_restore_the_last_good_draft():
    v1, v2, v3, v4 = (Path(f"/drafts/v{i}") for i in range(1, 5))
    admin = FakeAdminPlane(broken={v2}, split={v3})
    loader = DraftLoader(admin, served=v1, draft=v1)
    asyncio.run(loader.sync())
    assert admin.loaded == []

    for rejected in (v2, v3):
        loader.draft = rejected
        asyncio.run(loader.sync())
        assert admin.loaded[-2:] == [rejected, v1]
        assert (loader.served, loader.draft, loader.take_rejected()) == (v1, v1, rejected)
    assert loader.take_rejected() is None

    loader.draft = v4
    asyncio.run(loader.sync())
    assert (loader.served, loader.good) == (v4, v4)
    assert (loader.loads, loader.failures) == (3, 2)


def test_draft_count_switch_applies_once_and_retries_after_a_failure():
    switched = []
    failures = [True]

    class SwitchingAdminPlane(FakeAdminPlane):
        async def set_num_speculative_tokens(self, num: int | None) -> None:
            switched.append(num)
            if failures and failures.pop():
                raise RuntimeError("engine unreachable")

    draft = Path("/drafts/v1")
    admin = SwitchingAdminPlane(broken=set(), split=set())
    loader = DraftLoader(admin, served=draft, draft=draft, launch_num_speculative_tokens=3)
    asyncio.run(loader.sync())
    assert switched == [] and loader.gauges()["speculator/num_speculative_tokens"] == 3.0

    loader.num_speculative_tokens = 0
    asyncio.run(loader.sync())
    assert switched == [0] and "speculator/num_speculative_tokens" not in loader.gauges()
    asyncio.run(loader.sync())
    asyncio.run(loader.sync())
    assert switched == [0, 0] and loader.gauges()["speculator/num_speculative_tokens"] == 0.0

    loader.num_speculative_tokens = None
    asyncio.run(loader.sync())
    assert switched[-1] is None and loader.gauges()["speculator/num_speculative_tokens"] == 3.0

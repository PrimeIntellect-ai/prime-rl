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

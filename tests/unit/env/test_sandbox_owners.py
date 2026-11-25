from unittest.mock import AsyncMock, patch

import pytest
from prime_sandboxes import AsyncSandboxClient

from prime_rl.env.sandbox_owners import reap_previous_sandbox_pools, register_sandbox_pool


@pytest.mark.asyncio
async def test_resume_claims_only_dead_pools_of_earlier_attempts(tmp_path):
    old, new = (tmp_path / f"attempt_{n}" / "resolved" for n in (1, 2))
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    markers = old / "sandbox-owners"
    with register_sandbox_pool(new), patch.object(AsyncSandboxClient, "bulk_delete", new_callable=AsyncMock) as delete:
        with register_sandbox_pool(old) as label:
            # A pool that still serves keeps its sandboxes.
            await reap_previous_sandbox_pools(new)
            delete.assert_not_awaited()

        # A failed delete does not stop the resume, and the next resume tries again.
        delete.side_effect = RuntimeError("unavailable")
        await reap_previous_sandbox_pools(new)
        delete.side_effect = None
        delete.return_value.failed = [{"error": "unavailable"}]
        await reap_previous_sandbox_pools(new)
        assert len(list(markers.iterdir())) == 1
        delete.return_value.failed = []
        delete.return_value.succeeded = ["old-box"]
        await reap_previous_sandbox_pools(new)
        assert not list(markers.iterdir())
        await reap_previous_sandbox_pools(new)
        assert delete.await_count == 3
        delete.assert_awaited_with(labels=[label])

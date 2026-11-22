"""Run a random rollout for every env and print the violators of [-1, 1]."""
from __future__ import annotations

import pathlib
import sys

# Allow ``from tests._reward_helpers import _random_rollout_return`` when this
# script is invoked directly (i.e. without pytest, which already adds the repo
# root to sys.path).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._reward_helpers import _random_rollout_return  # noqa: E402

import glyphbench  # noqa: E402, F401 — registers all envs
from glyphbench.core.task_selection import list_task_ids  # noqa: E402


def main() -> int:
    violators: list[tuple[str, float, int]] = []
    for env_id in list_task_ids():
        try:
            total, length = _random_rollout_return(env_id, seed=0)
        except Exception as e:
            print(f"ERROR {env_id}: {e}", file=sys.stderr)
            continue
        if not (-1.0 - 1e-6 <= total <= 1.0 + 1e-6):
            violators.append((env_id, total, length))
    print("Violators (cumulative reward outside [-1, 1] under random policy, seed=0):")
    for env_id, total, length in sorted(violators):
        print(f"  {env_id}: {total:+.4f} ({length} steps)")
    print(f"\nTotal violators: {len(violators)}")
    print("\nKNOWN_REWARD_VIOLATORS = {")
    for env_id, _, _ in sorted(violators):
        print(f"    {env_id!r},")
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

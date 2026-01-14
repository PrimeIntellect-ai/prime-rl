from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from prime_rl.utils.hf_hub import upload_folder_to_hub


def main() -> int:
    # argv[1] is a json payload
    payload = json.loads(sys.argv[1])
    upload_folder_to_hub(
        repo_id=payload["repo_id"],
        folder_path=Path(payload["folder_path"]),
        path_in_repo=payload["path_in_repo"],
        commit_message=payload["commit_message"],
        create_repo=True,
        private=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)


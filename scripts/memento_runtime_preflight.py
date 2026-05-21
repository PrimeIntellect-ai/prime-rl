#!/usr/bin/env python
"""Runtime preflight for the Memento block-masking smoke job."""

import os
import sys
from importlib.metadata import version

import prime_rl
import vllm


def main() -> None:
    actual = version("vllm")
    expected_prefix = os.environ.get("EXPECTED_VLLM_VERSION_PREFIX")

    print(f"vLLM runtime version: {actual}")
    print(f"vLLM module path: {vllm.__file__}")
    print(f"prime_rl module path: {prime_rl.__file__}")
    print(f"Python executable: {sys.executable}")
    print(f"UV_PROJECT_ENVIRONMENT: {os.environ.get('UV_PROJECT_ENVIRONMENT')}")
    for package in ("prime-rl", "torch", "transformers"):
        print(f"{package} version: {version(package)}")

    if expected_prefix and not actual.startswith(expected_prefix):
        raise SystemExit(
            "Expected vLLM version prefix "
            f"{expected_prefix!r}, got {actual!r}. "
            "Sync/install the matching vLLM wheel before applying the block "
            "masking overlay."
        )


if __name__ == "__main__":
    main()

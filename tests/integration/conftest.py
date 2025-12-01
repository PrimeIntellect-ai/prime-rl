import os
import subprocess
from typing import Callable

import pytest


@pytest.fixture(scope="session")
def user() -> str:
    """Get currrent user from environment."""
    return os.environ.get("USERNAME_CI", os.environ.get("USER", "none"))


@pytest.fixture(scope="session")
def branch_name() -> str:
    """Get current branch name"""
    branch_name_ = os.environ.get("GITHUB_REF_NAME", None)

    if branch_name_ is None:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    else:
        branch_name = branch_name_.replace("/merge", "")
        branch_name = f"pr-{branch_name}"
    return branch_name


@pytest.fixture(scope="session")
def commit_hash() -> str:
    """Get current commit hash"""
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()


@pytest.fixture(scope="session")
def get_wandb_project(user: str) -> Callable[[str], str]:
    """Factory function to get W&B project name"""

    def _get_wandb_project(name: str) -> str:
        project = f"ci-{name}"
        if user != "CI_RUNNER":
            project += "-local"
        return project

    return _get_wandb_project

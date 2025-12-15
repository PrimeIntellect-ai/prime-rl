from typing import Callable

import pytest


@pytest.fixture(scope="module")
def wandb_project(get_wandb_project: Callable[[str], str]) -> str:
    """Fixture for W&B project name for integration tests. This means all integration tests will use the same W&B project."""
    return get_wandb_project("nightly-tests")

"""Dummy test-fixture env. Importing this module registers the env."""

from glyphbench.core.registry import register_env
from glyphbench.envs.dummy.env import DummyEnv

register_env("glyphbench/__dummy-v0", DummyEnv, default_use_memory=False)

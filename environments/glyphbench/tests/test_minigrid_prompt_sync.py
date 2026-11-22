"""Each minigrid env must include a [Legend] block in the per-turn obs,
not in the system prompt."""

from __future__ import annotations

import re

import pytest

from glyphbench.core.registry import all_glyphbench_env_ids, make_env
from glyphbench.envs import _import_all_suites

# Shared legend-block detector (mirrors test_legend_dedup._has_legend_block).
_LEGEND_LINE_RE = re.compile(r"^\s+\S\s+[=:\-]\s+\S", re.MULTILINE)


def _has_legend_block(text: str) -> bool:
    lines = text.splitlines()
    streak = 0
    for line in lines:
        if _LEGEND_LINE_RE.match(line):
            streak += 1
            if streak >= 3:
                return True
        else:
            streak = 0
    return False


@pytest.fixture(scope="module", autouse=True)
def _envs_loaded():
    _import_all_suites()


def _minigrid_env_ids():
    return [eid for eid in all_glyphbench_env_ids() if eid.startswith("glyphbench/minigrid-")]


def test_system_prompt_does_not_embed_legend_for_minigrid():
    for env_id in _minigrid_env_ids():
        env = make_env(env_id)
        env.reset(seed=0)
        sp = env.system_prompt()
        assert not _has_legend_block(sp), (
            f"{env_id}: system_prompt contains a glyph-enumeration legend block. "
            f"Move legend material to the per-turn observation."
        )


def test_per_turn_obs_has_legend_for_minigrid():
    for env_id in _minigrid_env_ids():
        env = make_env(env_id)
        obs_text, _ = env.reset(seed=0)
        assert "[Legend]" in obs_text, f"{env_id}: per-turn obs missing legend"

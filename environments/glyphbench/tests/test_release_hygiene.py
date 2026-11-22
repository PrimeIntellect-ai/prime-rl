from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def hygiene():
    script = Path(__file__).parents[1] / "scripts" / "check_release_hygiene.py"
    spec = importlib.util.spec_from_file_location("check_release_hygiene", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [".env", ".env.production", "id_ed25519", "job.sbatch"])
def test_rejects_sensitive_filenames(hygiene, tmp_path, name) -> None:
    path = tmp_path / name
    path.write_text("placeholder")
    assert hygiene.check_file(path)


@pytest.mark.parametrize(
    "content,label",
    [
        ("sk-" + "a" * 24, "provider token"),
        ("https://" + "user:password@example.com", "credentials in URL"),
        ("/" + "home/researcher/checkpoint", "absolute user home"),
        ("-----BEGIN " + "PRIVATE KEY-----", "private key"),
    ],
)
def test_rejects_sensitive_contents(hygiene, tmp_path, content, label) -> None:
    path = tmp_path / "config.txt"
    path.write_text(content)
    assert hygiene.check_file(path) == [label]


def test_allows_env_template(hygiene, tmp_path) -> None:
    path = tmp_path / ".env.example"
    path.write_text("OPENAI_API_KEY_LOCAL=EMPTY\n# OPENAI_API_KEY=\n")
    assert hygiene.check_file(path) == []


def test_rejects_notebook_outputs(hygiene, tmp_path) -> None:
    path = tmp_path / "analysis.ipynb"
    path.write_text(json.dumps({"cells": [{"outputs": [{"text": "run output"}]}]}))
    assert hygiene.check_file(path) == ["notebook contains execution state or outputs"]

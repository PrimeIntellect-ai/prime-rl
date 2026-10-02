import pytest

from scripts.docker_application import copy_installation_artifacts, editable_paths, set_fallback_version
from scripts.docker_dependencies import dependency_fingerprint


@pytest.fixture
def image_context(tmp_path):
    for name in ("Dockerfile.cuda", ".dockerignore", "uv.lock", "pyproject.toml", "scripts/docker_dependencies.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return tmp_path


def test_dependency_fingerprint_ignores_source_and_git(image_context):
    before = dependency_fingerprint(image_context)
    for name in ("src/model.py", ".git/config", ".venv/package/pyproject.toml"):
        path = image_context / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed")
    assert dependency_fingerprint(image_context) == before


@pytest.mark.parametrize("name", ["uv.lock", "Dockerfile.cuda", ".dockerignore", "deps/env/pyproject.toml"])
def test_dependency_fingerprint_tracks_inputs(image_context, name):
    before = dependency_fingerprint(image_context)
    path = image_context / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("changed")
    assert dependency_fingerprint(image_context) != before


def test_dependency_fingerprint_tracks_manifest_paths(image_context):
    first = image_context / "deps/first/pyproject.toml"
    first.parent.mkdir(parents=True)
    first.write_text("same content")
    before = dependency_fingerprint(image_context)
    second = image_context / "deps/second/pyproject.toml"
    second.parent.mkdir(parents=True)
    first.rename(second)
    assert dependency_fingerprint(image_context) != before


def test_editable_paths_include_local_packages_outside_workspace(tmp_path):
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "root"\nsource = { editable = "." }\n'
        '[[package]]\nname = "local"\nsource = { editable = "deps/local" }\n'
        '[[package]]\nname = "external"\nsource = { registry = "https://pypi.org/simple" }\n'
    )
    assert editable_paths(tmp_path) == [tmp_path, tmp_path / "deps/local"]


def test_editable_paths_reject_external_paths(tmp_path):
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "external"\nsource = { editable = "../external" }\n')
    with pytest.raises(ValueError, match="outside the application"):
        editable_paths(tmp_path)


def test_fallback_version_preserves_other_metadata(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.hatch.version]\nsource = "vcs"\nfallback-version = "0.0.0"\n')
    set_fallback_version(pyproject, "0.3.2.dev165")
    assert pyproject.read_text() == '[tool.hatch.version]\nsource = "vcs"\nfallback-version = "0.3.2.dev165"\n'


def test_application_artifacts_exclude_interpreter_and_bootstrap_files(tmp_path):
    venv = tmp_path / "venv"
    site = venv / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    (venv / "bin").mkdir()
    (venv / "bin/python").symlink_to("/usr/bin/python3.12")
    (venv / "pyvenv.cfg").write_text("interpreter configuration")
    (site / "_virtualenv.pth").write_text("bootstrap")
    baseline = {path.relative_to(venv) for path in venv.rglob("*")}
    (venv / "bin/rl").write_text("#!/app/.venv/bin/python\n")
    (site / "prime_rl.pth").write_text("/app/src\n")
    metadata = site / "prime_rl-0.9.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: prime-rl\nVersion: 0.9.0\n")
    output = tmp_path / "output"
    copy_installation_artifacts(venv, baseline, output)
    files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert files == {
        ".venv/bin/rl",
        ".venv/lib/python3.12/site-packages/prime_rl.pth",
        ".venv/lib/python3.12/site-packages/prime_rl-0.9.0.dist-info/METADATA",
    }

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def dependency_fingerprint(root: Path) -> str:
    inputs = [root / path for path in ("Dockerfile.cuda", ".dockerignore", "uv.lock", "scripts/docker_dependencies.py")]
    for directory, children, files in os.walk(root):
        children[:] = [name for name in children if name not in {".git", ".venv", "outputs"}]
        if "pyproject.toml" in files:
            inputs.append(Path(directory) / "pyproject.toml")
    digest = hashlib.sha256()
    for path in sorted(inputs):
        name = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def registry_digest(image: str) -> str | None:
    registry, _, reference = image.partition("/")
    if registry != "ghcr.io":
        raise ValueError("Dependency images must use ghcr.io")
    repository, separator, tag = reference.rpartition(":")
    if not separator or not repository or not tag:
        raise ValueError("Expected ghcr.io/<owner>/<image>:<tag>")

    query = urlencode({"service": registry, "scope": f"repository:{repository}:pull"})
    headers = {}
    if token := os.environ.get("GH_TOKEN"):
        credentials = f"{os.environ['GITHUB_ACTOR']}:{token}".encode()
        headers["Authorization"] = f"Basic {base64.b64encode(credentials).decode()}"
    request = Request(f"https://{registry}/token?{query}", headers=headers)
    with urlopen(request, timeout=30) as response:
        token = json.load(response)["token"]
    request = Request(
        f"https://{registry}/v2/{repository}/manifests/{tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json",
        },
        method="HEAD",
    )
    try:
        with urlopen(request, timeout=30) as response:
            digest = response.headers["Docker-Content-Digest"]
            if not digest or not digest.startswith("sha256:"):
                raise ValueError(f"Missing or invalid image digest for {image}")
            return digest
    except HTTPError as error:
        if error.code != 404:
            raise
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fingerprint and resolve Docker dependency images")
    commands = parser.add_subparsers(dest="command", required=True)
    fingerprint = commands.add_parser("fingerprint")
    fingerprint.add_argument("root", type=Path)
    resolve = commands.add_parser("resolve")
    resolve.add_argument("image")
    args = parser.parse_args()
    if args.command == "fingerprint":
        print(dependency_fingerprint(args.root.resolve()))
    else:
        print(registry_digest(args.image) or "")


if __name__ == "__main__":
    main()

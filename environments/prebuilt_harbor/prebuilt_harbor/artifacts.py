"""Move artifacts into the verifier box the way Harbor's separate verifier does.

Harbor uploads each declared root in turn, in declared order: a directory root is
emptied in place (`empty_dirs`: the directory itself is kept; a symlink or
non-directory there becomes a fresh directory) and filled with
`docker compose cp <host>/. <root>`; a file root's parent is created (`ensure_dirs`)
and the file copied to it. `docker compose cp`
brings everything in root-owned with modes kept, and every symlink verbatim, absolute
and chained ones included. A root missing from the agent box is skipped, a symlinked
directory root arrives as its target's content, and a root reached through another
(`/opt/x -> /app/x` next to `/app`) sees that root already in place.

One deliberate difference: Harbor 0.24 also opens each directory root, and each file
root's parent, to 0777. Tasks converted from Harbor's shared mode, where the verifier
saw the agent's box as it was, can have verifiers that reject a group- or
world-writable directory as tampering. A directory root here keeps the owner and mode
it had in the agent box, and a file root's parent is left as the verifier image has it.

verifiers' `restore` instead clears every root up front, extracts every archive, and
refuses absolute and chained symlinks (extracting through one could write outside the
root). So `PrebuiltHarborTask.finalize` stages each root: its archive is renamed under
its own scratch key (`STAGED_DIR/<n>/<root>`), made root-owned, and stripped of its
symlinks, which are recorded instead. `restore` then only extracts into the scratch
keys, and `PrebuiltHarborTask.stage_verifier` recreates the links there and moves each
root into place in
declared order (`place_script`). Nothing is ever extracted through a symlink, and the
scratch keys are plain paths whatever the agent named its files. A link must sit
inside its root, may not collide with another member, and may not be the parent of
any member; whatever it points to is kept, as under Harbor. Link paths and targets
are carried as raw bytes (hex), so names that are not UTF-8 survive and the trace
stays serializable.
"""

import io
import os
import tarfile
from pathlib import PurePosixPath

from verifiers.v1.errors import TaskError

INFO_KEY = "prebuilt_harbor_symlinks"
"""`trace.info` key holding the links taken out of each staged archive."""
ROOTS_KEY = "prebuilt_harbor_artifact_roots"
"""`trace.info` key: the staged roots in declared order, each
`{"root", "kind": "dir" | "file", "scratch"}` (see `place_script`)."""
STAGED_DIR = "/tmp/prebuilt-harbor-staged"
"""Where `restore` extracts each root, at `STAGED_DIR/<n>/<root>` (so saved eval
artifacts land under `tmp/prebuilt-harbor-staged/<n>/<root>`)."""
EMPTY_KEY = "/tmp/prebuilt-harbor-empty"
"""The one artifact key left when no root travels: `restore` clears and fills only
this scratch path, and an empty dict would make the caller collect raw archives
again."""


def _relative_root(root: str) -> PurePosixPath:
    path = PurePosixPath(root)
    if path.anchor != "/" or ".." in path.parts or path == PurePosixPath("/"):
        raise TaskError(f"artifact root {root!r} must be an absolute path below '/'")
    return path.relative_to("/")


def _member_path(name: str, root: str, relative_root: PurePosixPath) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or not (path == relative_root or path.is_relative_to(relative_root))
    ):
        raise TaskError(f"artifact member {name!r} is outside declared root {root!r}")
    return path


def _rewrite(source: tarfile.TarFile, members: list[tarfile.TarInfo]) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:", format=source.format) as target:
        for member in members:
            target.addfile(member, source.extractfile(member) if member.isfile() else None)
    return out.getvalue()


def _link(path: PurePosixPath, target: str, member: tarfile.TarInfo) -> dict:
    # tarfile decodes names with surrogateescape; fsencode gives back the raw bytes.
    return {
        "path_hex": os.fsencode(str(path)).hex(),
        "target_hex": os.fsencode(target).hex(),
        "uid": member.uid,
        "gid": member.gid,
    }


def split_symlinks(root: str, archive: bytes) -> tuple[bytes, list[dict]]:
    """`archive` without its symlinks, and those links as
    `{"path_hex", "target_hex", "uid", "gid"}` (path relative to `/`), in archive order.
    A hard link to a symlink is carried as a second symlink with the same target."""
    relative_root = _relative_root(root)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            members = tar.getmembers()
            symlinks: dict[PurePosixPath, tarfile.TarInfo] = {}
            for member in members:
                if not member.issym():
                    continue
                path = _member_path(member.name, root, relative_root)
                if path in symlinks:
                    raise TaskError(f"duplicate artifact symlink {member.name!r}")
                if not member.linkname or "\0" in member.linkname:
                    raise TaskError(f"artifact symlink {member.name!r} has no target")
                symlinks[path] = member
            if not symlinks:
                return archive, []
            carried: dict[PurePosixPath, tuple[str, tarfile.TarInfo]] = {
                path: (member.linkname, member) for path, member in symlinks.items()
            }
            kept = []
            for member in members:
                if member.issym():
                    continue
                path = _member_path(member.name, root, relative_root)
                if member.islnk() and PurePosixPath(member.linkname) in symlinks:
                    if path in carried:
                        raise TaskError(f"duplicate artifact link {member.name!r}")
                    target = symlinks[PurePosixPath(member.linkname)].linkname
                    carried[path] = (target, member)
                    continue
                if path in carried:
                    raise TaskError(f"artifact member {member.name!r} is also a symlink")
                kept.append(member)
            for member in members:
                path = PurePosixPath(member.name)
                if any(parent in carried for parent in path.parents):
                    raise TaskError(f"artifact member {member.name!r} traverses a symlink")
            rewritten = _rewrite(tar, kept)
    except tarfile.TarError as exc:
        raise TaskError(f"unreadable artifact archive for {root!r}: {exc}") from exc
    return rewritten, [_link(path, target, member) for path, (target, member) in carried.items()]


def root_is_symlink(root: str, archive: bytes) -> bool:
    """Whether `archive` holds `root` itself as a symlink (a symlinked root)."""
    relative_root = _relative_root(root)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            return any(member.issym() and PurePosixPath(member.name) == relative_root for member in tar)
    except tarfile.TarError as exc:
        raise TaskError(f"unreadable artifact archive for {root!r}: {exc}") from exc


def rebase(root: str, archive: bytes) -> bytes:
    """An archive of `tar -C <root>/ .` (the content of a symlinked root) renamed to
    `<root>/...`, as Harbor's `cp <root>/.` lays it out. Drops FIFOs and device nodes
    and hard links to them, as verifiers' own collection does."""
    relative_root = _relative_root(root)

    def renamed(name: str) -> str:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise TaskError(f"artifact member {name!r} escapes {root!r}")
        return str(relative_root / path) if str(path) != "." else str(relative_root)

    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            members = tar.getmembers()
            special = {m.name for m in members if m.isfifo() or m.ischr() or m.isblk()}
            kept = []
            for member in members:
                if member.name in special or (member.islnk() and member.linkname in special):
                    continue
                # A pax path/linkpath record would override the renamed attributes.
                member.pax_headers = {
                    key: value for key, value in member.pax_headers.items() if key not in ("path", "linkpath")
                }
                if member.islnk():
                    member.linkname = renamed(member.linkname)
                member.name = renamed(member.name)
                kept.append(member)
            return _rewrite(tar, kept)
    except tarfile.TarError as exc:
        raise TaskError(f"unreadable artifact archive for {root!r}: {exc}") from exc


def root_entry(root: str, archive: bytes) -> dict | None:
    """What `archive` holds at `root` itself: `{"kind": "dir" | "file", "uid", "gid",
    "mode"}` (mode as permission bits), None if absent."""
    relative_root = _relative_root(root)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar:
                if PurePosixPath(member.name) == relative_root:
                    return {
                        "kind": "dir" if member.isdir() else "file",
                        "uid": member.uid,
                        "gid": member.gid,
                        "mode": member.mode & 0o7777,
                    }
    except tarfile.TarError as exc:
        raise TaskError(f"unreadable artifact archive for {root!r}: {exc}") from exc
    return None


def owned_by_root(root: str, archive: bytes) -> bytes:
    """`archive` with every member owned by root, as Harbor's `docker compose cp` into
    the verifier container creates them (it keeps modes, not owners). A file that
    changes owner loses its setuid and setgid bits, as `chown` clears them: a setuid
    binary the agent made must not become setuid-root where the verifier runs."""
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            members = tar.getmembers()
            if all(m.uid == 0 and m.gid == 0 for m in members):
                return archive
            for member in members:
                if (member.uid or member.gid) and not member.isdir():
                    member.mode &= ~0o6000
                member.uid = member.gid = 0
                member.uname = member.gname = "root"
                for key in ("uid", "gid", "uname", "gname"):
                    member.pax_headers.pop(key, None)
            return _rewrite(tar, members)
    except tarfile.TarError as exc:
        raise TaskError(f"unreadable artifact archive for {root!r}: {exc}") from exc


def _quote(value: bytes) -> bytes:
    return b"'" + value.replace(b"'", b"'\"'\"'") + b"'"


def link_script(carried: dict[str, list[dict]]) -> bytes:
    """A POSIX sh script recreating `carried` links after `restore`. Fails, rather than
    replacing anything, if a path already exists."""
    lines = [b"set -e"]
    for links in carried.values():
        for link in links:
            raw = b"/" + bytes.fromhex(link["path_hex"])
            path = _quote(raw)
            target = _quote(bytes.fromhex(link["target_hex"]))
            owner = f"{int(link['uid'])}:{int(link['gid'])}".encode()
            lines += [
                b"if [ -e " + path + b" ] || [ -L " + path + b" ]; then "
                b"echo 'artifact symlink already exists: '" + path + b" >&2; exit 1; fi",
                b"mkdir -p -- " + _quote(os.path.dirname(raw)),
                b"ln -s -- " + target + b" " + path,
                b"chown -h " + owner + b" " + path,
            ]
    return b"\n".join(lines) + b"\n"


def empty_archive() -> bytes:
    """A valid tar archive with no members."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:"):
        pass
    return out.getvalue()


def relocate(root: str, scratch: str, archive: bytes) -> bytes:
    """`archive` of `root` renamed to `scratch`: `<root>[/...]` becomes
    `<scratch>[/...]`, hard links included. Every member must sit at or under `root`."""
    old = _relative_root(root)
    new = _relative_root(scratch)

    def renamed(name: str) -> str:
        path = _member_path(name, root, old)
        return str(new) if path == old else str(new / path.relative_to(old))

    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            members = tar.getmembers()
            for member in members:
                # A pax path/linkpath record would override the renamed attributes.
                member.pax_headers = {
                    key: value for key, value in member.pax_headers.items() if key not in ("path", "linkpath")
                }
                if member.islnk():
                    member.linkname = renamed(member.linkname)
                member.name = renamed(member.name)
            return _rewrite(tar, members)
    except tarfile.TarError as exc:
        raise TaskError(f"unreadable artifact archive for {root!r}: {exc}") from exc


def place_script(roots: list[dict]) -> bytes:
    """A POSIX sh script moving each staged root (`{"root", "kind", "scratch"}`, plus
    `"uid", "gid", "mode"` for a directory) into place, in order, as Harbor uploads it
    (see the module docstring), then removing `STAGED_DIR`. A directory root is emptied
    in place, given the owner and mode it had in the agent box, and its staged children
    are moved in; a staged file replaces the file root (or, as `docker cp` does, goes
    inside an existing directory there)."""
    lines = [b"set -e"]
    for entry in roots:
        path = PurePosixPath(entry["root"])
        _relative_root(entry["root"])
        root = _quote(os.fsencode(entry["root"]))
        staged = _quote(os.fsencode(entry["scratch"]))
        if entry["kind"] == "dir":
            lines += [
                b"if [ -L "
                + root
                + b" ] || { [ -e "
                + root
                + b" ] && [ ! -d "
                + root
                + b" ]; }; then rm -rf -- "
                + root
                + b"; fi",
                b"mkdir -p -- " + root,
                b"find " + root + b" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +",
                # Same filesystem: rename each child in. Another one (Prime's /run
                # tmpfs): one copy, which keeps hard links between children.
                b'if [ "$(stat -c %d ' + staged + b')" = "$(stat -c %d ' + root + b')" ]; then',
                b"    for f in " + staged + b"/* " + staged + b"/.[!.]* " + staged + b"/..?*; do",
                b'        if [ -e "$f" ] || [ -L "$f" ]; then mv -- "$f" ' + root + b"/; fi",
                b"    done",
                b"else",
                b"    cp -a -- " + staged + b"/. " + root + b"/",
                b"fi",
                f"chown {int(entry['uid'])}:{int(entry['gid'])} ".encode() + root,
                f"chmod {int(entry['mode']) & 0o7777:o} ".encode() + root,
            ]
        elif entry["kind"] == "file":
            parent = _quote(os.fsencode(str(path.parent)))
            inside = root + b"/" + _quote(os.fsencode(path.name))
            lines += [
                b"mkdir -p -- " + parent,
                b"if [ -d "
                + root
                + b" ] && [ ! -L "
                + root
                + b" ]; then mv -- "
                + staged
                + b" "
                + inside
                + b"; else rm -f -- "
                + root
                + b"; mv -- "
                + staged
                + b" "
                + root
                + b"; fi",
            ]
        else:
            raise TaskError(f"artifact root {entry['root']!r} has unknown kind {entry['kind']!r}")
    lines.append(b"rm -rf -- " + _quote(os.fsencode(STAGED_DIR)))
    return b"\n".join(lines) + b"\n"

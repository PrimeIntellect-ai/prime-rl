import json
from pathlib import Path

# Provenance, not architecture. Loads cast to the served dtype.
PROVENANCE_KEYS = ("_name_or_path", "transformers_version", "speculators_version", "dtype", "torch_dtype")
# Served drafts share the target's embedding and LM head and loads skip them, so checkpoints may leave them out.
TARGET_SHARED_TENSORS = ("embed_tokens.weight", "lm_head.weight")


def local_draft_dir(name: str, revision: str | None = None) -> Path:
    """A draft directory, or a Hub id resolved from the local Hugging Face cache."""
    path = Path(name)
    if path.is_dir() or path.is_absolute():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name, revision=revision, local_files_only=True))


def draft_architecture(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text())
    for key in PROVENANCE_KEYS:
        config.pop(key, None)
    # The verifier path and proposal methods change between refreshes; serving sets its own.
    config["speculators_config"] = config.get("speculators_config", {}).get("algorithm")
    return config


def tensor_shapes(directory: Path) -> dict[str, list[int]]:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors files in {directory}")
    shapes = {}
    for path in files:
        with path.open("rb") as file:
            header = json.loads(file.read(int.from_bytes(file.read(8), "little")))
        header.pop("__metadata__", None)
        shapes.update((name, tensor["shape"]) for name, tensor in header.items())
    return shapes


def check_draft_swap(served: Path, new: Path) -> None:
    """Raise unless ``new`` can replace the served draft in place: same architecture, same tensors."""
    served_config, new_config = draft_architecture(served), draft_architecture(new)
    changed = sorted(k for k in served_config.keys() | new_config.keys() if served_config.get(k) != new_config.get(k))
    if changed:
        raise ValueError(f"Draft {new} differs from the served draft {served} in {changed}")
    served_shapes, new_shapes = tensor_shapes(served), tensor_shapes(new)
    optional = {
        k for k in served_shapes.keys() ^ new_shapes.keys() if k.removeprefix("model.") in TARGET_SHARED_TENSORS
    }
    names = (served_shapes.keys() | new_shapes.keys()) - optional
    changed = sorted(k for k in names if served_shapes.get(k) != new_shapes.get(k))
    if changed:
        raise ValueError(f"Draft {new} tensors differ from the served draft {served}: {changed[:8]}")

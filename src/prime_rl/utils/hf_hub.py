from __future__ import annotations

from pathlib import Path

from prime_rl.utils.logger import get_logger


def should_upload_step(step: int, keep_interval: int | None) -> bool:
    """
    Decide whether a given checkpoint step should be uploaded to HF Hub.

    We intentionally key this off `keep_interval` only (per request), not `keep_last`.
    """
    if keep_interval is None:
        return False
    if keep_interval <= 0:
        return False
    return step % keep_interval == 0


LARGE_UPLOAD_THRESHOLD_BYTES = 5 * 1024**3


def _path_in_repo(repo_prefix: str, kind: str, step: int) -> str:
    repo_prefix = repo_prefix.strip().strip("/")
    if repo_prefix:
        return f"{repo_prefix}/{kind}/step_{step}"
    return f"{kind}/step_{step}"


def _should_use_large_upload(folder_path: Path) -> bool:
    total_size = 0
    for path in folder_path.rglob("*"):
        if path.is_file():
            total_size += path.stat().st_size
            if total_size >= LARGE_UPLOAD_THRESHOLD_BYTES:
                return True
    return False

def upload_folder_to_hub(
    *,
    repo_id: str,
    folder_path: Path,
    path_in_repo: str,
    commit_message: str,
    create_repo: bool = True,
    private: bool = True,
    repo_type: str = "model",
) -> None:
    """
    Upload a local folder to the Hugging Face Hub.

    This is intentionally "upload from disk" (no in-memory serialization).
    """
    logger = get_logger()

    import huggingface_hub as hf_hub

    api = hf_hub.HfApi()
    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=private)

    logger.info(f"Uploading {folder_path} to hf://{repo_id}/{path_in_repo}")
    use_large_upload = _should_use_large_upload(folder_path)
    upload_fn = api.upload_folder
    if use_large_upload and hasattr(api, "upload_large_folder"):
        upload_fn = api.upload_large_folder
    upload_fn(
        repo_id=repo_id,
        folder_path=str(folder_path),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        repo_type=repo_type,
    )


def build_training_path_in_repo(repo_prefix: str, step: int) -> str:
    return _path_in_repo(repo_prefix, "checkpoints", step)


def build_weights_path_in_repo(repo_prefix: str, step: int) -> str:
    return _path_in_repo(repo_prefix, "weights", step)


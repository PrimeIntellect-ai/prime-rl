import asyncio
import hashlib
import time
from argparse import Namespace
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State, UploadFile
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.serving import OpenAIServingModels

from prime_rl.utils.logger import get_logger

logger = get_logger()
router = APIRouter()

WEIGHT_UPDATE_MODES = {"full", "delta"}
STAGE_READ_CHUNK_BYTES = 16 * 1024 * 1024


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def _ensure_weight_staging_state(state: State) -> None:
    if not hasattr(state, "staged_versions"):
        state.staged_versions = {}
    if not hasattr(state, "stage_uploads"):
        state.stage_uploads = {}
    if not hasattr(state, "finalized_stage_uploads"):
        state.finalized_stage_uploads = {}
    if not hasattr(state, "active_version"):
        state.active_version = "base"
    if not hasattr(state, "weight_update_lock"):
        state.weight_update_lock = asyncio.Lock()
    if not hasattr(state, "stage_upload_lock"):
        state.stage_upload_lock = asyncio.Lock()
    if not hasattr(state, "weights_dirty"):
        state.weights_dirty = False
    if not hasattr(state, "staging_dir"):
        state.staging_dir = Path("staging")
    state.staging_dir.mkdir(parents=True, exist_ok=True)


def _safe_path_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return safe or "unknown"


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _normalize_admin_url(url: str) -> str:
    return url.rstrip("/").removesuffix("/v1")


def _relay_peer_url(peer: str, route: str) -> str:
    return f"{_normalize_admin_url(peer)}/{route.lstrip('/')}"


def _relay_requested(fields: dict[str, Any]) -> bool:
    return _coerce_bool(fields.get("relay"), default=True)


def _relay_peers(state: State) -> list[str]:
    if not bool(getattr(state, "relay_enabled", False)):
        return []
    return list(getattr(state, "relay_peers", []) or [])


def _relay_transport(state: State, peer: str) -> httpx.AsyncBaseTransport | None:
    transports = getattr(state, "relay_transports", None)
    if not transports:
        return None
    return transports.get(peer) or transports.get(_normalize_admin_url(peer))


async def _relay_post(
    state: State,
    peer: str,
    route: str,
    *,
    timeout_s: float,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    content: bytes | None = None,
) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout_s, transport=_relay_transport(state, peer)) as client:
        response = await client.post(
            _relay_peer_url(peer, route),
            params=params,
            data=data,
            json=json,
            files=files,
            content=content,
        )
        response.raise_for_status()
        return response


def _relay_failure_response(state: State, operation: str, stats: dict[str, float | None]) -> JSONResponse | None:
    if not bool(getattr(state, "relay_fail_on_peer_error", True)):
        return None
    failed_peers = [peer for peer, duration in stats.items() if duration is None]
    if not failed_peers:
        return None
    return JSONResponse(
        {
            "error": f"relay {operation} failed for {len(failed_peers)} peer(s)",
            "failed_peers": failed_peers,
            "relay": stats,
        },
        status_code=502,
    )


async def _relay_pause_peer(state: State, peer: str, timeout_s: float) -> None:
    await _relay_post(state, peer, "/pause", timeout_s=timeout_s, params={"mode": "keep", "clear_cache": "false"})


async def _relay_resume_peer(state: State, peer: str, timeout_s: float) -> None:
    await _relay_post(state, peer, "/resume", timeout_s=timeout_s)


def _stage_relay_file(path: Path, mode: str) -> Path:
    if path.is_file():
        return path
    if mode == "delta" and path.is_dir():
        stream_path = path / "delta.stream"
        return stream_path if stream_path.exists() else path / "delta.safetensors"
    return path


async def _fan_out_stage_file(
    state: State,
    *,
    path: Path,
    version: str,
    mode: str,
    base_version: Any,
    relay: bool,
) -> dict[str, float | None]:
    peers = _relay_peers(state)
    if not relay or not peers:
        return {}

    upload_path = _stage_relay_file(path, mode)
    if not upload_path.is_file():
        logger.warning(
            "[relay][stage] cannot fan-out %s weights version %s from non-file path %s",
            mode,
            version,
            path.as_posix(),
        )
        return {peer: None for peer in peers}

    data: dict[str, Any] = {"version": version, "mode": mode, "relay": "false"}
    if base_version is not None:
        data["base_version"] = str(base_version)

    results: dict[str, float | None] = {}
    logger.info("[relay][stage] fan-out start v%s (%s) peers=%s", version, mode, peers)

    async def _send(peer: str) -> None:
        start = time.perf_counter()
        try:
            with upload_path.open("rb") as f:
                files = {"file": (upload_path.name, f, "application/octet-stream")}
                await _relay_post(
                    state,
                    peer,
                    "/stage",
                    timeout_s=getattr(state, "relay_stage_timeout_s", 3600.0),
                    data=data,
                    files=files,
                )
        except Exception as exc:
            results[peer] = None
            logger.warning("[relay][stage] v%s -> %s failed: %s", version, peer, exc)
            return
        results[peer] = time.perf_counter() - start
        logger.info("[relay][stage] v%s -> %s in %.2fs", version, peer, results[peer])

    await asyncio.gather(*(_send(peer) for peer in peers))
    return results


async def _fan_out_stage_chunk(
    state: State,
    *,
    fields: dict[str, Any],
    filename: str,
    chunk: bytes,
    relay: bool,
) -> dict[str, float | None]:
    peers = _relay_peers(state)
    if not relay or not peers:
        return {}

    data = {key: str(value) for key, value in fields.items() if key != "relay"}
    data["relay"] = "false"
    results: dict[str, float | None] = {}
    version = str(fields.get("version", "unknown"))

    async def _send(peer: str) -> None:
        start = time.perf_counter()
        try:
            files = {"file": (filename, chunk, "application/octet-stream")}
            await _relay_post(
                state,
                peer,
                "/stage_chunk",
                timeout_s=getattr(state, "relay_stage_timeout_s", 3600.0),
                data=data,
                files=files,
            )
        except Exception as exc:
            results[peer] = None
            logger.warning("[relay][stage_chunk] v%s -> %s failed: %s", version, peer, exc)
            return
        results[peer] = time.perf_counter() - start

    await asyncio.gather(*(_send(peer) for peer in peers))
    return results


async def _fan_out_stage_finalize(
    state: State,
    *,
    fields: dict[str, Any],
    relay: bool,
) -> dict[str, float | None]:
    peers = _relay_peers(state)
    if not relay or not peers:
        return {}

    data = {key: str(value) for key, value in fields.items() if key != "relay"}
    data["relay"] = "false"
    results: dict[str, float | None] = {}
    version = str(fields.get("version", "unknown"))
    logger.info("[relay][stage_finalize] fan-out start v%s peers=%s", version, peers)

    async def _send(peer: str) -> None:
        start = time.perf_counter()
        try:
            await _relay_post(
                state,
                peer,
                "/stage_finalize",
                timeout_s=getattr(state, "relay_stage_timeout_s", 3600.0),
                data=data,
            )
        except Exception as exc:
            results[peer] = None
            logger.warning("[relay][stage_finalize] v%s -> %s failed: %s", version, peer, exc)
            return
        results[peer] = time.perf_counter() - start
        logger.info("[relay][stage_finalize] v%s -> %s in %.2fs", version, peer, results[peer])

    await asyncio.gather(*(_send(peer) for peer in peers))
    return results


async def _fan_out_stream_init(
    state: State,
    *,
    fields: dict[str, Any],
    filename: str,
    relay: bool,
) -> tuple[dict[str, str], dict[str, float | None]]:
    peers = _relay_peers(state)
    if not relay or not peers:
        return {}, {}

    payload = {key: value for key, value in fields.items() if key != "relay"}
    payload["filename"] = filename
    payload["relay"] = False
    relay_uploads: dict[str, str] = {}
    results: dict[str, float | None] = {}
    version = str(fields.get("version", "unknown"))
    logger.info("[relay][stage_stream_init] fan-out start v%s peers=%s", version, peers)

    async def _send(peer: str) -> None:
        start = time.perf_counter()
        try:
            response = await _relay_post(
                state,
                peer,
                "/stage_stream_init",
                timeout_s=getattr(state, "relay_stage_timeout_s", 3600.0),
                json=payload,
            )
            upload_id = response.json().get("upload_id")
            if not upload_id:
                raise RuntimeError("stage_stream_init response did not include upload_id")
        except Exception as exc:
            results[peer] = None
            logger.warning("[relay][stage_stream_init] v%s -> %s failed: %s", version, peer, exc)
            return
        relay_uploads[peer] = str(upload_id)
        results[peer] = time.perf_counter() - start

    await asyncio.gather(*(_send(peer) for peer in peers))
    return relay_uploads, results


async def _fan_out_stream_chunk(
    state: State,
    *,
    upload: dict[str, Any],
    upload_id: str,
    offset: int,
    chunk: bytes,
) -> dict[str, float | None]:
    relay_uploads = upload.get("relay_uploads") or {}
    if not upload.get("relay_requested", True) or not relay_uploads:
        return {}

    results: dict[str, float | None] = {}
    version = str(upload.get("version", "unknown"))

    async def _send(peer: str, peer_upload_id: str) -> None:
        start = time.perf_counter()
        try:
            await _relay_post(
                state,
                peer,
                "/stage_stream_chunk",
                timeout_s=getattr(state, "relay_stage_timeout_s", 3600.0),
                params={"upload_id": peer_upload_id, "offset": offset},
                content=chunk,
            )
        except Exception as exc:
            results[peer] = None
            logger.warning("[relay][stage_stream_chunk] v%s -> %s failed: %s", version, peer, exc)
            return
        results[peer] = time.perf_counter() - start

    await asyncio.gather(*(_send(peer, peer_upload_id) for peer, peer_upload_id in relay_uploads.items()))
    return results


async def _fan_out_stream_finalize(
    state: State,
    *,
    upload: dict[str, Any],
    fields: dict[str, Any],
) -> dict[str, float | None]:
    relay_uploads = upload.get("relay_uploads") or {}
    if not upload.get("relay_requested", True) or not relay_uploads:
        return {}

    data = {key: str(value) for key, value in fields.items() if key != "upload_id"}
    results: dict[str, float | None] = {}
    version = str(upload.get("version", "unknown"))
    logger.info("[relay][stage_stream_finalize] fan-out start v%s peers=%s", version, list(relay_uploads))

    async def _send(peer: str, peer_upload_id: str) -> None:
        start = time.perf_counter()
        try:
            await _relay_post(
                state,
                peer,
                "/stage_stream_finalize",
                timeout_s=getattr(state, "relay_stage_timeout_s", 3600.0),
                data={**data, "upload_id": peer_upload_id},
            )
        except Exception as exc:
            results[peer] = None
            logger.warning("[relay][stage_stream_finalize] v%s -> %s failed: %s", version, peer, exc)
            return
        results[peer] = time.perf_counter() - start
        logger.info("[relay][stage_stream_finalize] v%s -> %s in %.2fs", version, peer, results[peer])

    await asyncio.gather(*(_send(peer, peer_upload_id) for peer, peer_upload_id in relay_uploads.items()))
    return results


async def _fan_out_commit(
    state: State,
    *,
    version: str,
    mode: str,
    relay: bool,
) -> dict[str, float | None]:
    peers = _relay_peers(state)
    if not relay or not peers:
        return {}

    results: dict[str, float | None] = {}
    timeout_s = getattr(state, "relay_commit_timeout_s", 600.0)
    data = {"version": version, "mode": mode, "relay": "false"}
    logger.info("[relay][commit] fan-out start v%s (%s) peers=%s", version, mode, peers)

    async def _send(peer: str) -> None:
        start = time.perf_counter()
        try:
            await _relay_pause_peer(state, peer, timeout_s)
            try:
                await _relay_post(state, peer, "/commit", timeout_s=timeout_s, data=data)
            finally:
                await _relay_resume_peer(state, peer, timeout_s)
        except Exception as exc:
            results[peer] = None
            logger.warning("[relay][commit] v%s -> %s failed: %s", version, peer, exc)
            return
        results[peer] = time.perf_counter() - start
        logger.info("[relay][commit] v%s -> %s in %.2fs", version, peer, results[peer])

    await asyncio.gather(*(_send(peer) for peer in peers))
    return results


async def _fan_out_reload(state: State, *, relay: bool) -> dict[str, float | None]:
    peers = _relay_peers(state)
    if not relay or not peers:
        return {}

    results: dict[str, float | None] = {}
    timeout_s = getattr(state, "relay_reload_timeout_s", 600.0)
    logger.info("[relay][reload_weights] fan-out start peers=%s", peers)

    async def _send(peer: str) -> None:
        start = time.perf_counter()
        try:
            await _relay_pause_peer(state, peer, timeout_s)
            try:
                await _relay_post(state, peer, "/reload_weights", timeout_s=timeout_s, data={"relay": "false"})
            finally:
                await _relay_resume_peer(state, peer, timeout_s)
        except Exception as exc:
            results[peer] = None
            logger.warning("[relay][reload_weights] -> %s failed: %s", peer, exc)
            return
        results[peer] = time.perf_counter() - start
        logger.info("[relay][reload_weights] -> %s in %.2fs", peer, results[peer])

    await asyncio.gather(*(_send(peer) for peer in peers))
    return results


async def _read_request_fields(request: Request) -> tuple[dict[str, Any], UploadFile | None]:
    fields: dict[str, Any] = dict(request.query_params)
    uploaded_file = None
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        data = await request.json()
        if isinstance(data, dict):
            fields.update({key: value for key, value in data.items() if value is not None})
    elif content_type.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        form = await request.form()
        for key, value in form.items():
            if isinstance(value, UploadFile):
                if key == "file":
                    uploaded_file = value
                continue
            fields[key] = value

    return fields, uploaded_file


def _cleanup_staged_file(state: State, entry: dict[str, Any]) -> None:
    if not entry.get("owned"):
        return
    path = Path(entry["path"])
    staging_dir = Path(state.staging_dir)
    if path.is_file() and path.parent == staging_dir:
        path.unlink()


def _cleanup_stage_upload(state: State, upload: dict[str, Any]) -> None:
    path = Path(upload["path"])
    staging_dir = Path(state.staging_dir)
    if path.is_file() and path.parent == staging_dir:
        path.unlink()


async def _register_staged_version(state: State, version: str, entry: dict[str, Any]) -> JSONResponse | None:
    async with state.weight_update_lock:
        metadata = _validate_stage_metadata(state, {**entry, "version": version})
        if isinstance(metadata, JSONResponse):
            _cleanup_staged_file(state, entry)
            return metadata
        previous = state.staged_versions.get(version)
        if previous is not None and previous["path"] != entry["path"]:
            _cleanup_staged_file(state, previous)
        state.staged_versions[version] = entry
    return None


def _stage_upload_key(version: str, mode: str) -> str:
    return f"{version}:{mode}"


def _stage_stream_upload_key(upload_id: str) -> str:
    return f"stream:{upload_id}"


def _parse_upload_id(fields: dict[str, Any]) -> str | JSONResponse:
    upload_id = str(fields.get("upload_id") or uuid4().hex)
    if len(upload_id) > 128 or _safe_path_component(upload_id) != upload_id:
        return _error_response(400, "upload_id must contain only letters, numbers, '.', '_', or '-'")
    return upload_id


def _parse_non_negative_int(fields: dict[str, Any], name: str) -> int | JSONResponse:
    value = fields.get(name)
    if value is None:
        return _error_response(400, f"{name} is required")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return _error_response(400, f"{name} must be an integer")
    if parsed < 0:
        return _error_response(400, f"{name} must be non-negative")
    return parsed


def _validate_stage_metadata(state: State, fields: dict[str, Any]) -> tuple[str, str, Any, Any] | JSONResponse:
    version = str(fields.get("version", "unknown"))
    mode = str(fields.get("mode", "full"))
    base_version = fields.get("base_version")
    if base_version is not None:
        base_version = str(base_version)
    active_version = state.active_version

    if mode not in WEIGHT_UPDATE_MODES:
        return _error_response(400, f"unsupported weight update mode: {mode}")
    if getattr(state, "api_server_count", 1) != 1:
        return _error_response(400, "stage_commit requires a single API server to own the version state")
    if mode == "delta":
        if state.weights_dirty:
            return _error_response(409, "weights require reload after a failed update")
        current_version = active_version or "0"
        if base_version is None:
            return _error_response(400, "base_version is required for delta updates")
        if base_version != current_version:
            return _error_response(409, f"base_version mismatch: current={current_version}, got={base_version}")
        if version == base_version:
            return _error_response(409, "delta version must differ from base_version")

    return version, mode, base_version, active_version


def _merge_received_range(ranges: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    if start >= end:
        return ranges
    ranges.append((start, end))
    ranges.sort()

    merged: list[tuple[int, int]] = []
    for range_start, range_end in ranges:
        if not merged or range_start > merged[-1][1]:
            merged.append((range_start, range_end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, range_end))
    return merged


def _received_bytes(ranges: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in ranges)


def _upload_complete(ranges: list[tuple[int, int]], total_size: int) -> bool:
    if total_size == 0:
        return ranges == []
    return len(ranges) == 1 and ranges[0] == (0, total_size)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(STAGE_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


@router.post("/stage")
async def stage_weights(request: Request):
    """Stage full or delta weights on the local inference server."""
    _ensure_weight_staging_state(request.app.state)
    fields, uploaded_file = await _read_request_fields(request)
    relay_requested = _relay_requested(fields)
    metadata = _validate_stage_metadata(request.app.state, fields)
    if isinstance(metadata, JSONResponse):
        return metadata
    version, mode, base_version, active_version = metadata

    ingest_start = time.perf_counter()
    suffix = "delta" if mode == "delta" else "full"
    if uploaded_file is not None:
        filename = _safe_path_component(Path(uploaded_file.filename or "weights").name)
        staged_path = request.app.state.staging_dir / f"{uuid4().hex}_{suffix}_{filename}"
        with staged_path.open("wb") as f:
            while chunk := await uploaded_file.read(STAGE_READ_CHUNK_BYTES):
                f.write(chunk)
        owned = True
    else:
        path = fields.get("path")
        if path is None:
            return _error_response(400, "No file uploaded or path provided")
        staged_path = Path(str(path)).expanduser()
        if not staged_path.exists():
            return _error_response(404, f"staged path does not exist: {staged_path}")
        owned = False

    error = await _register_staged_version(
        request.app.state,
        version,
        {"path": staged_path, "mode": mode, "base_version": base_version, "owned": owned},
    )
    if error is not None:
        return error
    ingest_ms = (time.perf_counter() - ingest_start) * 1000
    logger.info(
        "Staged %s weights version %s from %s (base_version=%s active_version=%s owned=%s ingest_ms=%.2f)",
        mode,
        version,
        staged_path.as_posix(),
        base_version,
        active_version,
        owned,
        ingest_ms,
    )
    fanout_start = time.perf_counter()
    relay_stats = await _fan_out_stage_file(
        request.app.state,
        path=staged_path,
        version=version,
        mode=mode,
        base_version=base_version,
        relay=relay_requested,
    )
    fanout_ms = (time.perf_counter() - fanout_start) * 1000
    failure_response = _relay_failure_response(request.app.state, "stage", relay_stats)
    if failure_response is not None:
        return failure_response

    return {
        "status": "ok",
        "version": version,
        "mode": mode,
        "path": staged_path.as_posix(),
        "ingest_ms": ingest_ms,
        "relay": relay_stats,
        "fanout_ms": fanout_ms,
    }


@router.post("/stage_stream_init")
async def stage_stream_init(request: Request):
    """Initialize a streaming staged upload."""
    _ensure_weight_staging_state(request.app.state)
    fields, _ = await _read_request_fields(request)
    relay_requested = _relay_requested(fields)
    metadata = _validate_stage_metadata(request.app.state, fields)
    if isinstance(metadata, JSONResponse):
        return metadata
    version, mode, base_version, active_version = metadata

    filename = fields.get("filename")
    if filename is None:
        return _error_response(400, "filename is required")
    filename = _safe_path_component(Path(str(filename)).name)

    upload_id = _parse_upload_id(fields)
    if isinstance(upload_id, JSONResponse):
        return upload_id
    upload_key = _stage_stream_upload_key(upload_id)

    async with request.app.state.stage_upload_lock:
        finalized = request.app.state.finalized_stage_uploads.get(upload_key)
        if finalized is not None:
            upload = finalized["upload"]
            if (
                upload["filename"] != filename
                or upload["mode"] != mode
                or upload["version"] != version
                or upload["base_version"] != base_version
            ):
                return _error_response(409, "upload_id is already finalized with different metadata")
            return {
                "status": "ok",
                "upload_id": upload_id,
                "version": version,
                "mode": mode,
                "finalized": True,
                "relay": {},
            }

        existing = request.app.state.stage_uploads.get(upload_key)
        if existing is not None:
            if (
                existing["filename"] != filename
                or existing["mode"] != mode
                or existing["version"] != version
                or existing["base_version"] != base_version
            ):
                return _error_response(409, "upload_id is already in use with different metadata")
            return {
                "status": "ok",
                "upload_id": upload_id,
                "version": version,
                "mode": mode,
                "relay": existing["init_relay"],
            }

        suffix = "delta" if mode == "delta" else "full"
        temp_path = request.app.state.staging_dir / f"{upload_id}_{suffix}_{filename}.stream.part"
        staged_path = request.app.state.staging_dir / f"{upload_id}_{suffix}_{filename}"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with temp_path.open("wb"):
            pass

        relay_uploads, relay_stats = await _fan_out_stream_init(
            request.app.state,
            fields=fields,
            filename=filename,
            relay=relay_requested,
        )
        failure_response = _relay_failure_response(request.app.state, "stage_stream_init", relay_stats)
        if failure_response is not None:
            _cleanup_stage_upload(request.app.state, {"path": temp_path})
            return failure_response

        request.app.state.stage_uploads[upload_key] = {
            "path": temp_path,
            "staged_path": staged_path,
            "filename": filename,
            "mode": mode,
            "version": version,
            "base_version": base_version,
            "ranges": [],
            "chunks": 0,
            "start_time": time.perf_counter(),
            "streaming": True,
            "relay_requested": relay_requested,
            "relay_uploads": relay_uploads,
            "init_relay": relay_stats,
        }
    logger.info(
        "Initialized streaming stage for %s weights version %s (base_version=%s active_version=%s upload_id=%s)",
        mode,
        version,
        base_version,
        active_version,
        upload_id,
    )
    return {"status": "ok", "upload_id": upload_id, "version": version, "mode": mode, "relay": relay_stats}


@router.post("/stage_stream_chunk")
async def stage_stream_chunk(request: Request):
    """Upload one raw streaming chunk for a staged weight file."""
    _ensure_weight_staging_state(request.app.state)
    fields = dict(request.query_params)
    upload_id = fields.get("upload_id")
    if upload_id is None:
        return _error_response(400, "upload_id is required")

    offset = _parse_non_negative_int(fields, "offset")
    if isinstance(offset, JSONResponse):
        return offset

    upload = request.app.state.stage_uploads.get(_stage_stream_upload_key(str(upload_id)))
    if upload is None:
        return _error_response(404, f"streaming upload {upload_id} not found")

    chunk = await request.body()
    if not chunk:
        return _error_response(400, "chunk body is required")
    end_offset = offset + len(chunk)

    temp_path = Path(upload["path"])
    with temp_path.open("r+b" if temp_path.exists() else "w+b") as f:
        f.seek(offset)
        f.write(chunk)

    upload["ranges"] = _merge_received_range(upload["ranges"], offset, end_offset)
    upload["chunks"] += 1
    relay_stats = await _fan_out_stream_chunk(
        request.app.state,
        upload=upload,
        upload_id=str(upload_id),
        offset=offset,
        chunk=chunk,
    )
    failure_response = _relay_failure_response(request.app.state, "stage_stream_chunk", relay_stats)
    if failure_response is not None:
        return failure_response
    return {
        "status": "ok",
        "upload_id": upload_id,
        "version": upload["version"],
        "mode": upload["mode"],
        "received_bytes": _received_bytes(upload["ranges"]),
        "relay": relay_stats,
    }


@router.post("/stage_stream_finalize")
async def stage_stream_finalize(request: Request):
    """Finalize a streaming staged upload and make it available for /commit."""
    _ensure_weight_staging_state(request.app.state)
    fields, _ = await _read_request_fields(request)
    upload_id = fields.get("upload_id")
    if upload_id is None:
        return _error_response(400, "upload_id is required")

    final_size = _parse_non_negative_int(fields, "final_size")
    if isinstance(final_size, JSONResponse):
        return final_size
    expected_sha256 = fields.get("sha256")
    if expected_sha256 is not None:
        expected_sha256 = str(expected_sha256)

    upload_key = _stage_stream_upload_key(str(upload_id))
    async with request.app.state.stage_upload_lock:
        finalized = request.app.state.finalized_stage_uploads.get(upload_key)
        if finalized is not None:
            if finalized["final_size"] != final_size or finalized["sha256"] != expected_sha256:
                return _error_response(409, "finalize metadata does not match completed upload")
            upload = finalized["upload"]
            response = finalized["response"]
        else:
            upload = request.app.state.stage_uploads.get(upload_key)
            if upload is None:
                return _error_response(404, f"streaming upload {upload_id} not found")

            metadata = _validate_stage_metadata(request.app.state, upload)
            if isinstance(metadata, JSONResponse):
                return metadata

            ranges = upload["ranges"]
            if not _upload_complete(ranges, final_size):
                return _error_response(409, f"upload incomplete: received={_received_bytes(ranges)} total={final_size}")

            temp_path = Path(upload["path"])
            if not temp_path.exists():
                return _error_response(404, f"streaming upload file for {upload_id} not found")
            actual_size = temp_path.stat().st_size
            if actual_size != final_size:
                return _error_response(409, f"final_size mismatch: expected={final_size}, got={actual_size}")

            if expected_sha256 is not None:
                actual_sha256 = _file_sha256(temp_path)
                if actual_sha256 != expected_sha256:
                    _cleanup_stage_upload(request.app.state, upload)
                    request.app.state.stage_uploads.pop(upload_key, None)
                    return _error_response(409, f"sha256 mismatch: expected={expected_sha256}, got={actual_sha256}")

            staged_path = Path(upload["staged_path"])
            if staged_path.exists():
                staged_path.unlink()
            temp_path.replace(staged_path)

            version = str(upload["version"])
            error = await _register_staged_version(
                request.app.state,
                version,
                {
                    "path": staged_path,
                    "mode": upload["mode"],
                    "base_version": upload["base_version"],
                    "owned": True,
                },
            )
            if error is not None:
                return error
            request.app.state.stage_uploads.pop(upload_key, None)
            ingest_ms = (time.perf_counter() - upload["start_time"]) * 1000
            logger.info(
                "Staged %s weights version %s from %s via streaming upload "
                "(base_version=%s active_version=%s owned=True chunks=%s ingest_ms=%.2f)",
                upload["mode"],
                version,
                staged_path.as_posix(),
                upload["base_version"],
                request.app.state.active_version,
                upload["chunks"],
                ingest_ms,
            )
            response = {
                "status": "ok",
                "version": version,
                "mode": upload["mode"],
                "path": staged_path.as_posix(),
                "ingest_ms": ingest_ms,
            }
            request.app.state.finalized_stage_uploads[upload_key] = {
                "version": version,
                "final_size": final_size,
                "sha256": expected_sha256,
                "upload": upload,
                "response": response,
            }

    fanout_start = time.perf_counter()
    relay_stats = await _fan_out_stream_finalize(request.app.state, upload=upload, fields=fields)
    fanout_ms = (time.perf_counter() - fanout_start) * 1000
    failure_response = _relay_failure_response(request.app.state, "stage_stream_finalize", relay_stats)
    if failure_response is not None:
        return failure_response

    return {**response, "relay": relay_stats, "fanout_ms": fanout_ms}


@router.post("/stage_chunk")
async def stage_chunk(request: Request):
    """Upload one chunk of a staged weight file to the local inference server."""
    _ensure_weight_staging_state(request.app.state)
    fields, uploaded_file = await _read_request_fields(request)
    relay_requested = _relay_requested(fields)
    metadata = _validate_stage_metadata(request.app.state, fields)
    if isinstance(metadata, JSONResponse):
        return metadata
    version, mode, base_version, _active_version = metadata

    if uploaded_file is None:
        return _error_response(400, "file chunk is required")

    offset = _parse_non_negative_int(fields, "offset")
    if isinstance(offset, JSONResponse):
        return offset
    total_size = _parse_non_negative_int(fields, "total_size")
    if isinstance(total_size, JSONResponse):
        return total_size

    expected_sha256 = fields.get("sha256")
    if expected_sha256 is None:
        return _error_response(400, "sha256 is required")
    expected_sha256 = str(expected_sha256)

    filename = _safe_path_component(Path(str(fields.get("filename") or uploaded_file.filename or "weights")).name)
    chunk = await uploaded_file.read()
    end_offset = offset + len(chunk)
    if end_offset > total_size:
        return _error_response(400, f"chunk exceeds total_size: offset={offset}, size={len(chunk)}, total={total_size}")

    suffix = "delta" if mode == "delta" else "full"
    key = _stage_upload_key(version, mode)
    finalized = request.app.state.finalized_stage_uploads.get(f"chunk:{key}")
    if finalized is not None:
        if (
            finalized["filename"] != filename
            or finalized["total_size"] != total_size
            or finalized["sha256"] != expected_sha256
            or finalized["base_version"] != base_version
        ):
            return _error_response(409, "chunk metadata does not match completed upload")
        return {
            "status": "ok",
            "version": version,
            "mode": mode,
            "received_bytes": total_size,
            "total_size": total_size,
            "finalized": True,
            "relay": {},
        }

    upload = request.app.state.stage_uploads.get(key)
    if upload is None:
        temp_path = request.app.state.staging_dir / f"{_safe_path_component(version)}_{suffix}_{filename}.part"
        upload = {
            "path": temp_path,
            "filename": filename,
            "mode": mode,
            "base_version": base_version,
            "total_size": total_size,
            "sha256": expected_sha256,
            "ranges": [],
            "chunks": 0,
            "start_time": time.perf_counter(),
        }
        request.app.state.stage_uploads[key] = upload
    elif (
        upload["filename"] != filename
        or upload["total_size"] != total_size
        or upload["sha256"] != expected_sha256
        or upload["base_version"] != base_version
    ):
        return _error_response(409, "chunk metadata does not match active upload")

    temp_path = Path(upload["path"])
    with temp_path.open("r+b" if temp_path.exists() else "w+b") as f:
        f.seek(offset)
        f.write(chunk)

        upload["ranges"] = _merge_received_range(upload["ranges"], offset, end_offset)
    upload["chunks"] += 1
    relay_stats = await _fan_out_stage_chunk(
        request.app.state,
        fields=fields,
        filename=filename,
        chunk=chunk,
        relay=relay_requested,
    )
    failure_response = _relay_failure_response(request.app.state, "stage_chunk", relay_stats)
    if failure_response is not None:
        return failure_response
    return {
        "status": "ok",
        "version": version,
        "mode": mode,
        "received_bytes": _received_bytes(upload["ranges"]),
        "total_size": total_size,
        "relay": relay_stats,
    }


@router.post("/stage_finalize")
async def stage_finalize(request: Request):
    """Finalize a chunked staged upload and make it available for /commit."""
    _ensure_weight_staging_state(request.app.state)
    fields, _ = await _read_request_fields(request)
    relay_requested = _relay_requested(fields)
    metadata = _validate_stage_metadata(request.app.state, fields)
    if isinstance(metadata, JSONResponse):
        return metadata
    version, mode, base_version, active_version = metadata

    total_size = _parse_non_negative_int(fields, "total_size")
    if isinstance(total_size, JSONResponse):
        return total_size
    expected_sha256 = fields.get("sha256")
    if expected_sha256 is None:
        return _error_response(400, "sha256 is required")
    expected_sha256 = str(expected_sha256)

    filename = _safe_path_component(Path(str(fields.get("filename") or "weights")).name)
    key = _stage_upload_key(version, mode)
    finalized_key = f"chunk:{key}"
    async with request.app.state.stage_upload_lock:
        finalized = request.app.state.finalized_stage_uploads.get(finalized_key)
        if finalized is not None:
            if (
                finalized["filename"] != filename
                or finalized["total_size"] != total_size
                or finalized["sha256"] != expected_sha256
                or finalized["base_version"] != base_version
            ):
                return _error_response(409, "finalize metadata does not match completed upload")
            response = finalized["response"]
        else:
            upload = request.app.state.stage_uploads.get(key)
            if upload is None:
                return _error_response(404, f"upload for version {version} not found")
            if (
                upload["filename"] != filename
                or upload["total_size"] != total_size
                or upload["sha256"] != expected_sha256
                or upload["base_version"] != base_version
            ):
                return _error_response(409, "finalize metadata does not match active upload")

            ranges = upload["ranges"]
            if not _upload_complete(ranges, total_size):
                return _error_response(409, f"upload incomplete: received={_received_bytes(ranges)} total={total_size}")

            temp_path = Path(upload["path"])
            actual_sha256 = _file_sha256(temp_path)
            if actual_sha256 != expected_sha256:
                _cleanup_stage_upload(request.app.state, upload)
                request.app.state.stage_uploads.pop(key, None)
                return _error_response(409, f"sha256 mismatch: expected={expected_sha256}, got={actual_sha256}")

            suffix = "delta" if mode == "delta" else "full"
            staged_path = request.app.state.staging_dir / f"{uuid4().hex}_{suffix}_{filename}"
            if staged_path.exists():
                staged_path.unlink()
            temp_path.replace(staged_path)

            error = await _register_staged_version(
                request.app.state,
                version,
                {"path": staged_path, "mode": mode, "base_version": base_version, "owned": True},
            )
            if error is not None:
                return error
            request.app.state.stage_uploads.pop(key, None)
            ingest_ms = (time.perf_counter() - upload["start_time"]) * 1000
            logger.info(
                "Staged %s weights version %s from %s via chunked upload "
                "(base_version=%s active_version=%s owned=True chunks=%s ingest_ms=%.2f)",
                mode,
                version,
                staged_path.as_posix(),
                base_version,
                active_version,
                upload["chunks"],
                ingest_ms,
            )
            response = {
                "status": "ok",
                "version": version,
                "mode": mode,
                "path": staged_path.as_posix(),
                "ingest_ms": ingest_ms,
            }
            request.app.state.finalized_stage_uploads[finalized_key] = {
                "version": version,
                "filename": filename,
                "total_size": total_size,
                "sha256": expected_sha256,
                "base_version": base_version,
                "response": response,
            }

    fanout_start = time.perf_counter()
    relay_stats = await _fan_out_stage_finalize(request.app.state, fields=fields, relay=relay_requested)
    fanout_ms = (time.perf_counter() - fanout_start) * 1000
    failure_response = _relay_failure_response(request.app.state, "stage_finalize", relay_stats)
    if failure_response is not None:
        return failure_response

    return {**response, "relay": relay_stats, "fanout_ms": fanout_ms}


@router.post("/commit")
async def commit_weights(request: Request):
    """Commit a staged full checkpoint or sparse delta."""
    _ensure_weight_staging_state(request.app.state)
    async with request.app.state.weight_update_lock:
        return await _commit_weights(request)


async def _commit_weights(request: Request):
    fields, _ = await _read_request_fields(request)
    relay_requested = _relay_requested(fields)
    version = fields.get("version")
    if version is None:
        return _error_response(400, "version is required")
    version = str(version)

    staged_versions = request.app.state.staged_versions
    if version not in staged_versions:
        return _error_response(404, f"version {version} not found in staging")

    entry = staged_versions[version]
    mode = entry["mode"]
    requested_mode = fields.get("mode")
    if requested_mode is not None and str(requested_mode) != mode:
        return _error_response(409, f"staged mode mismatch: staged={mode}, requested={requested_mode}")

    path = Path(entry["path"])
    state = request.app.state
    if state.weights_dirty:
        return _error_response(409, "weights require reload after a failed update")
    if state.active_version != version:
        metadata = _validate_stage_metadata(state, {**entry, "version": version})
        if isinstance(metadata, JSONResponse):
            return metadata
        state.weights_dirty = True
        method = "update_weights_from_delta_path" if mode == "delta" else "update_weights_from_path"
        await engine_client(request).collective_rpc(method, args=(path.as_posix(),))
        state.active_version = version
        state.weights_dirty = False
    for staged_version, old_entry in list(staged_versions.items()):
        if staged_version == version:
            continue
        _cleanup_staged_file(request.app.state, old_entry)
        staged_versions.pop(staged_version, None)

    fanout_start = time.perf_counter()
    relay_stats = await _fan_out_commit(
        request.app.state,
        version=version,
        mode=mode,
        relay=relay_requested,
    )
    fanout_ms = (time.perf_counter() - fanout_start) * 1000
    failure_response = _relay_failure_response(request.app.state, "commit", relay_stats)
    if failure_response is not None:
        return failure_response
    for upload_key, upload in list(state.finalized_stage_uploads.items()):
        if upload["version"] == version:
            state.finalized_stage_uploads.pop(upload_key, None)

    logger.info("Committed %s weights version %s from %s", mode, version, path.as_posix())
    return {"status": "ok", "active_version": version, "mode": mode, "relay": relay_stats, "fanout_ms": fanout_ms}


@router.post("/reload_weights")
async def reload_weights(request: Request):
    """Reload the base model weights and clear local staging state."""
    _ensure_weight_staging_state(request.app.state)
    async with request.app.state.weight_update_lock:
        return await _reload_weights(request)


async def _reload_weights(request: Request):
    fields, _ = await _read_request_fields(request)
    relay_requested = _relay_requested(fields)
    request.app.state.weights_dirty = True
    await engine_client(request).collective_rpc("reload_weights")
    for entry in list(request.app.state.staged_versions.values()):
        _cleanup_staged_file(request.app.state, entry)
    for upload in list(request.app.state.stage_uploads.values()):
        _cleanup_stage_upload(request.app.state, upload)
    request.app.state.active_version = "base"
    request.app.state.weights_dirty = False
    request.app.state.staged_versions.clear()
    request.app.state.stage_uploads.clear()
    request.app.state.finalized_stage_uploads.clear()
    fanout_start = time.perf_counter()
    relay_stats = await _fan_out_reload(request.app.state, relay=relay_requested)
    fanout_ms = (time.perf_counter() - fanout_start) * 1000
    failure_response = _relay_failure_response(request.app.state, "reload_weights", relay_stats)
    if failure_response is not None:
        return failure_response
    logger.info("Reloaded base weights and cleared staged weight versions")
    return {"status": "ok", "relay": relay_stats, "fanout_ms": fanout_ms}


def initialize_delta_sync_state(state: State, args: Namespace) -> None:
    state.staging_dir = Path(getattr(args, "staging_dir", "staging"))
    state.staged_versions = {}
    state.stage_uploads = {}
    state.finalized_stage_uploads = {}
    state.active_version = "base"
    state.weights_dirty = False
    state.weight_update_lock = asyncio.Lock()
    state.stage_upload_lock = asyncio.Lock()
    state.api_server_count = int(getattr(args, "api_server_count", 1))
    state.staging_dir.mkdir(parents=True, exist_ok=True)
    state.relay_enabled = bool(getattr(args, "relay_enabled", False))
    state.relay_peers = list(getattr(args, "relay_peers", []) or [])
    state.relay_fail_on_peer_error = bool(getattr(args, "relay_fail_on_peer_error", True))
    state.relay_stage_timeout_s = float(getattr(args, "relay_stage_timeout_s", 3600.0))
    state.relay_commit_timeout_s = float(getattr(args, "relay_commit_timeout_s", 600.0))
    state.relay_reload_timeout_s = float(getattr(args, "relay_reload_timeout_s", 600.0))
    if state.relay_enabled:
        logger.info("Inference relay enabled for peers: %s", state.relay_peers)

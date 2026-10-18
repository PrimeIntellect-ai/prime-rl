"""Where the sparse weight broadcast's data lives: the broadcast dir on a shared filesystem (a
``Path``), or an S3-compatible bucket (an ``S3Path``, from an ``s3://bucket/prefix`` URL) so
trainer and engines can be in different regions. ``S3Path`` implements the few ``Path`` methods
the transport uses; ``read_range``/``remove`` work on both. Configure the endpoint
and credentials the usual boto3 way (e.g. ``AWS_ENDPOINT_URL``, ``AWS_ACCESS_KEY_ID``).
"""

import fnmatch
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

PART_SIZE = 64 * 2**20


@cache
def _client():
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(max_pool_connections=64, retries={"mode": "standard"}))


class S3Path:
    def __init__(self, url: str):
        self.bucket, _, self.key = url.removeprefix("s3://").partition("/")
        self.key = self.key.strip("/")

    def __truediv__(self, name: str) -> "S3Path":
        return S3Path(f"s3://{self.bucket}/{self.key}/{name}")

    def __repr__(self) -> str:
        return self.as_posix()

    def as_posix(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def name(self) -> str:
        return self.key.rsplit("/", 1)[-1]

    @property
    def parent(self) -> "S3Path":
        return S3Path(f"s3://{self.bucket}/{self.key.rpartition('/')[0]}")

    @property
    def _prefix(self) -> str:
        return f"{self.key}/" if self.key else ""

    def _list(self, delimiter: str = "/") -> tuple[list[str], list[str]]:
        """Object keys and sub-prefixes below this prefix."""
        keys, prefixes = [], []
        pages = (
            _client()
            .get_paginator("list_objects_v2")
            .paginate(Bucket=self.bucket, Prefix=self._prefix, Delimiter=delimiter)
        )
        for page in pages:
            keys.extend(item["Key"] for item in page.get("Contents", []))
            prefixes.extend(item["Prefix"].rstrip("/") for item in page.get("CommonPrefixes", []))
        return keys, prefixes

    def exists(self) -> bool:
        from botocore.exceptions import ClientError

        try:
            _client().head_object(Bucket=self.bucket, Key=self.key)
            return True
        except ClientError:
            return self.is_dir()

    def is_dir(self) -> bool:
        response = _client().list_objects_v2(Bucket=self.bucket, Prefix=self._prefix, MaxKeys=1)
        return response.get("KeyCount", 0) > 0

    def glob(self, pattern: str) -> list["S3Path"]:
        keys, prefixes = self._list()
        return [
            S3Path(f"s3://{self.bucket}/{key}")
            for key in keys + prefixes
            if fnmatch.fnmatch(key.rsplit("/", 1)[-1], pattern)
        ]

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        pass

    def read_text(self) -> str:
        return _client().get_object(Bucket=self.bucket, Key=self.key)["Body"].read().decode()


StoragePath = Path | S3Path


def storage_path(location: str | Path) -> StoragePath:
    location = str(location)
    return S3Path(location) if location.startswith("s3://") else Path(location)


def read_range(path: StoragePath, offset: int, size: int) -> bytes:
    """``size`` bytes at ``offset`` (a ranged GET on S3)."""
    if isinstance(path, S3Path):
        if size == 0:
            return b""
        response = _client().get_object(Bucket=path.bucket, Key=path.key, Range=f"bytes={offset}-{offset + size - 1}")
        return response["Body"].read()
    with open(path, "rb") as file:
        data = os.pread(file.fileno(), size, offset)
        # Broadcast data is read once; don't let it fill the inference host's page cache.
        os.posix_fadvise(file.fileno(), offset, size, os.POSIX_FADV_DONTNEED)
    return data


def upload(path: S3Path, buffers: list) -> None:
    """Upload ``buffers`` back to back as one object: a single PUT, or a multipart upload in parts of
    ``PART_SIZE`` (S3 allows up to 10k); only the pieces of one part are copied together."""
    buffers = [memoryview(buffer).cast("B") for buffer in buffers]
    parts, current, size = [], [], 0
    for buffer in buffers:
        while len(buffer):
            piece, buffer = buffer[: PART_SIZE - size], buffer[PART_SIZE - size :]
            current.append(piece)
            size += len(piece)
            if size == PART_SIZE:
                parts.append(current)
                current, size = [], 0
    if current or not parts:
        parts.append(current)
    client = _client()
    if len(parts) == 1:
        client.put_object(Bucket=path.bucket, Key=path.key, Body=b"".join(parts[0]))
        return
    upload_id = client.create_multipart_upload(Bucket=path.bucket, Key=path.key)["UploadId"]

    def upload_part(number: int) -> dict:
        body = b"".join(parts[number - 1])
        response = client.upload_part(
            Bucket=path.bucket, Key=path.key, UploadId=upload_id, PartNumber=number, Body=body
        )
        return {"ETag": response["ETag"], "PartNumber": number}

    try:
        with ThreadPoolExecutor(8) as pool:
            uploaded = list(pool.map(upload_part, range(1, len(parts) + 1)))
        client.complete_multipart_upload(
            Bucket=path.bucket, Key=path.key, UploadId=upload_id, MultipartUpload={"Parts": uploaded}
        )
    except BaseException:
        client.abort_multipart_upload(Bucket=path.bucket, Key=path.key, UploadId=upload_id)
        raise


def remove(path: StoragePath) -> None:
    """Delete a dir and everything below it."""
    if isinstance(path, S3Path):
        keys, _ = path._list(delimiter="")
        for start in range(0, len(keys), 1000):
            objects = [{"Key": key} for key in keys[start : start + 1000]]
            _client().delete_objects(Bucket=path.bucket, Delete={"Objects": objects, "Quiet": True})
    else:
        shutil.rmtree(path, ignore_errors=True)

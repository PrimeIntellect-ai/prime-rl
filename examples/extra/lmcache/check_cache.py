"""Check external prefix reuse and salt isolation on an idle, single-engine server."""

import argparse
import json
import time
from urllib.request import Request, urlopen
from uuid import uuid4


def post(base_url: str, path: str, body: dict) -> dict:
    request = Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def check_cache(base_url: str, model: str) -> None:
    salt = uuid4().hex
    prompt = "The quick brown fox jumps over the lazy dog. " * 256
    tokens = post(base_url, "/tokenize", {"model": model, "prompt": prompt})["tokens"]
    for label, cache_salt, expect_hit in (
        ("cold policy A", salt + ":A", False),
        ("warm policy A", salt + ":A", True),
        ("cold policy B", salt + ":B", False),
        ("warm policy B", salt + ":B", True),
    ):
        # Async offload can temporarily hold blocks after a request completes.
        deadline = time.monotonic() + 30
        while not post(base_url, "/reset_prefix_cache?reset_external=false", {})["success"]:
            if time.monotonic() >= deadline:
                raise RuntimeError("Local prefix cache reset timed out; use an idle, single-engine server.")
            time.sleep(0.1)

        started = time.monotonic()
        result = post(
            base_url,
            "/inference/v1/generate",
            {
                "request_id": uuid4().hex,
                "model": model,
                "token_ids": tokens,
                "cache_salt": cache_salt,
                "sampling_params": {"temperature": 0, "max_tokens": 1},
            },
        )
        usage = result["usage"]
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        assert isinstance(cached, int) and 0 <= cached <= usage["prompt_tokens"], usage
        assert (cached > 0) == expect_hit, f"{label}: expected hit={expect_hit}, got {cached} cached tokens"
        print(f"{label}: {cached}/{usage['prompt_tokens']} cached tokens, {time.monotonic() - started:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Direct engine URL, without /v1 (requires VLLM_SERVER_DEV_MODE=1)")
    parser.add_argument("model", help="Served model name")
    args = parser.parse_args()
    check_cache(args.base_url, args.model)

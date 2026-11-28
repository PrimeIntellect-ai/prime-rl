"""Swap the checkpoint a whole deployment serves: every DP group installs a ModelExpress version (one a checkpoint
publisher offers, see tools/publish_checkpoint.py) through `POST /swap-checkpoint` on the group's first rank.

    uv run python tools/swap_checkpoint.py (--router URL | --servers URL...) (--step N | --version-uid UID)
        [--mode fail|drain|abort] [--drain-timeout-s 300] [--rolling]

The engines come from the router's worker list (vllm-router `GET /list_workers`) or from --servers. Each reports its
DP group (`GET /swap-checkpoint`); the group's first rank swaps all of its ranks. By default every group swaps at
once, so the whole deployment pauses together for about the same time. With --rolling, groups swap one at a time:
each leaves the router's rotation first (`/remove_worker`), so the router sends new requests to the others, and
rejoins once it serves the new checkpoint; a mode other than `drain` makes little sense then.

Sends the admin token from $PRIME_RL_ADMIN_TOKEN (--admin-token-var). Prints one line per group and a JSON summary,
and exits 0 only if every group serves the target.
"""

import argparse
import asyncio
import json
import os
import sys
import time

import httpx

ADMIN_TOKEN_HEADER = "X-Prime-Admin-Token"


def base(url: str) -> str:
    return url.rstrip("/").removesuffix("/v1")


async def router_workers(client: httpx.AsyncClient, router: str) -> list[str]:
    response = await client.get(f"{router}/list_workers")
    response.raise_for_status()
    body = response.json()
    urls = body.get("urls") if isinstance(body, dict) else body
    if urls is None and isinstance(body, dict):
        urls = [worker["url"] for worker in body.get("workers", [])]
    return [base(url) for url in urls]


async def discover_groups(client: httpx.AsyncClient, servers: list[str]) -> dict[str, list[str]]:
    """Group leader -> the servers of its DP group. Fails if a server belongs to no group that can swap."""

    async def info(server: str) -> dict:
        response = await client.get(f"{server}/swap-checkpoint")
        response.raise_for_status()
        return response.json()

    infos = dict(zip(servers, await asyncio.gather(*map(info, servers))))
    groups = {
        server: [base(url) for url in info["group"]] if info["group"] else [server]
        for server, info in infos.items()
        if info["available"]
    }
    covered = {member for members in groups.values() for member in members}
    if orphans := [server for server in servers if server not in covered]:
        reasons = "; ".join(f"{server}: {infos[server].get('reason')}" for server in orphans)
        raise SystemExit(f"No group leader swaps these servers: {reasons}")
    return groups


async def swap_group(client: httpx.AsyncClient, leader: str, body: dict) -> dict:
    start = time.perf_counter()
    response = await client.post(f"{leader}/swap-checkpoint", json=body)
    try:
        result = response.json()
    except ValueError:
        result = {"error": response.text[:2000]}
    return {"leader": leader, "status": response.status_code, "seconds": time.perf_counter() - start, **result}


async def wait_healthy(client: httpx.AsyncClient, servers: list[str], timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    for server in servers:
        while True:
            try:
                if (await client.get(f"{server}/health")).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{server} is not healthy")
            await asyncio.sleep(1)


async def swap_rolling(client: httpx.AsyncClient, router: str, groups: dict, body: dict) -> list[dict]:
    results = []
    for leader, members in groups.items():
        for member in members:
            (await client.post(f"{router}/remove_worker", params={"url": member})).raise_for_status()
        result = await swap_group(client, leader, body)
        # Only a failed install (500) leaves a group paused and unhealthy: it stays out of the rotation
        if result["status"] != 500:
            await wait_healthy(client, members)
            for member in members:
                (await client.post(f"{router}/add_worker", params={"url": member})).raise_for_status()
        print(line(result), flush=True)
        results.append(result)
        if result["status"] != 200:
            break
    return results


def line(result: dict) -> str:
    if result["status"] == 200:
        what = "unchanged" if result.get("unchanged") else f"timings {json.dumps(result.get('timings'))}"
        return f"ok   {result['leader']}: step {result['step']} ({result['version_uid']}), {what}"
    return f"FAIL {result['leader']}: HTTP {result['status']}: {result.get('error') or result.get('detail')}"


async def main_async(args: argparse.Namespace) -> int:
    headers = {}
    if token := os.environ.get(args.admin_token_var):
        headers[ADMIN_TOKEN_HEADER] = token
    body = {"mode": args.mode, "drain_timeout_s": args.drain_timeout_s}
    body.update({"step": args.step} if args.step is not None else {"version_uid": args.version_uid})
    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(args.timeout)) as client:
        servers = (
            [base(url) for url in args.servers] if args.servers else await router_workers(client, base(args.router))
        )
        groups = await discover_groups(client, servers)
        print(f"{len(servers)} servers in {len(groups)} DP groups: {json.dumps(groups)}", flush=True)
        if args.rolling:
            if not args.router:
                raise SystemExit("--rolling takes the groups out of the router's rotation: it needs --router")
            results = await swap_rolling(client, base(args.router), groups, body)
        else:
            results = await asyncio.gather(*(swap_group(client, leader, body) for leader in groups))
            for result in results:
                print(line(result), flush=True)
    ok = len(results) == len(groups) and all(result["status"] == 200 for result in results)
    print(json.dumps({"ok": ok, "groups": results}, indent=2))
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--router", help="vllm-router URL; its workers are the engines")
    where.add_argument("--servers", nargs="+", help="every engine server's URL")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--step", type=int, help="serve the newest READY version of this step")
    target.add_argument("--version-uid", help="serve this ModelExpress version")
    parser.add_argument("--mode", choices=["fail", "drain", "abort"], default="fail")
    parser.add_argument("--drain-timeout-s", type=float, default=300.0)
    parser.add_argument("--rolling", action="store_true", help="swap one group at a time, out of the rotation")
    parser.add_argument("--admin-token-var", default="PRIME_RL_ADMIN_TOKEN")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per HTTP request")
    sys.exit(asyncio.run(main_async(parser.parse_args())))


if __name__ == "__main__":
    main()

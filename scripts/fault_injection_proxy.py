#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import urlparse

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class FaultState:
    def __init__(self, *, fail_path: str, fail_method: str, fail_count: int, fail_status: int):
        self.fail_path = fail_path
        self.fail_method = fail_method.upper()
        self.fail_count = fail_count
        self.fail_status = fail_status
        self.failed = 0
        self.lock = threading.Lock()

    def should_fail(self, *, method: str, path: str) -> bool:
        with self.lock:
            if self.failed >= self.fail_count:
                return False
            if method.upper() != self.fail_method:
                return False
            if path != self.fail_path:
                return False
            self.failed += 1
            return True

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "fail_path": self.fail_path,
                "fail_method": self.fail_method,
                "fail_count": self.fail_count,
                "failed": self.failed,
                "remaining": max(self.fail_count - self.failed, 0),
            }


class FaultProxyHandler(BaseHTTPRequestHandler):
    target = urlparse("http://127.0.0.1:8002")
    state: ClassVar[FaultState]
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[fault-proxy] {self.address_string()} - {fmt % args}", flush=True)

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def do_OPTIONS(self) -> None:
        self._handle()

    def _handle(self) -> None:
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/__faults":
            self._send_json(200, self.state.snapshot())
            return

        body = self._read_body()
        if self.state.should_fail(method=self.command, path=parsed_path.path):
            print(
                f"[fault-proxy] injecting {self.state.fail_status} for {self.command} {parsed_path.path}",
                flush=True,
            )
            self._send_json(
                self.state.fail_status,
                {
                    "error": "fault injection",
                    "method": self.command,
                    "path": parsed_path.path,
                    "state": self.state.snapshot(),
                },
            )
            return

        self._forward(body)

    def _read_body(self) -> bytes:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            return b""
        return self.rfile.read(int(content_length))

    def _forward(self, body: bytes) -> None:
        target = self.target
        if target.scheme == "https":
            connection = http.client.HTTPSConnection(
                target.hostname,
                target.port or 443,
                context=ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(target.hostname, target.port or 80)

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        headers["Host"] = target.netloc
        headers["Content-Length"] = str(len(body))
        forward_path = self.path
        try:
            connection.request(self.command, forward_path, body=body, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
        finally:
            connection.close()

        self.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            if key.lower() in HOP_BY_HOP_HEADERS or key.lower() == "content-length":
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small HTTP fault-injection proxy for local Prime-RL E2E runs.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-base-url", required=True)
    parser.add_argument("--fail-path", default="/stage")
    parser.add_argument("--fail-method", default="POST")
    parser.add_argument("--fail-count", type=int, default=1)
    parser.add_argument("--fail-status", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FaultProxyHandler.target = urlparse(args.target_base_url.rstrip("/"))
    FaultProxyHandler.state = FaultState(
        fail_path=args.fail_path,
        fail_method=args.fail_method,
        fail_count=args.fail_count,
        fail_status=args.fail_status,
    )
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), FaultProxyHandler)
    print(
        "[fault-proxy] listening on "
        f"http://{args.listen_host}:{args.listen_port}, target={args.target_base_url}, "
        f"fail={args.fail_method.upper()} {args.fail_path} x{args.fail_count}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

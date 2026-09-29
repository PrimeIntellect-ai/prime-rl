"""Find distinct free TCP ports on the node that will run a distributed service."""

import argparse
import errno
import random
import socket
from collections.abc import Iterable


def find_available_ports(
    host: str, count: int, excluded: Iterable[int] = (), *, start: int = 29500, stop: int = 30000
) -> list[int]:
    if count < 1 or stop - start < count:
        raise ValueError("Port range cannot satisfy the requested count")
    address = socket.gethostbyname(host)
    if address == "0.0.0.0":
        raise ValueError("Port discovery requires a specific host address")
    addresses = (address,) if address == "127.0.0.1" else (address, "127.0.0.1")
    reserved = set(excluded)
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for port in random.SystemRandom().sample(range(start, stop), stop - start):
            if port in reserved:
                continue
            candidates: list[socket.socket] = []
            try:
                for bind_address in addresses:
                    candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        candidate.bind((bind_address, port))
                    except OSError:
                        candidate.close()
                        raise
                    candidates.append(candidate)
            except OSError as error:
                for candidate in candidates:
                    candidate.close()
                if error.errno in (errno.EADDRINUSE, errno.EACCES):
                    continue
                raise
            sockets.extend(candidates)
            ports.append(port)
            if len(ports) == count:
                return ports
    finally:
        for sock in sockets:
            sock.close()
    raise RuntimeError(f"Could not find {count} free TCP ports in [{start}, {stop})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("count", type=int)
    parser.add_argument("excluded", type=int, nargs="*")
    args = parser.parse_args()
    print(*find_available_ports(args.host, args.count, args.excluded))


if __name__ == "__main__":
    main()

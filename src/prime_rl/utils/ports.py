"""Find distinct free TCP ports on the node that will run a distributed service."""

import argparse
import errno
import random
import socket
from collections.abc import Iterable


def find_available_ports(
    count: int, excluded: Iterable[int] = (), *, start: int = 29500, stop: int = 30000
) -> list[int]:
    if count < 1 or stop - start < count:
        raise ValueError("Port range cannot satisfy the requested count")
    reserved = set(excluded)
    sockets: list[socket.socket] = []
    try:
        for port in random.SystemRandom().sample(range(start, stop), stop - start):
            if port in reserved:
                continue
            candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                candidate.bind(("0.0.0.0", port))
            except OSError as error:
                candidate.close()
                if error.errno in (errno.EADDRINUSE, errno.EACCES):
                    continue
                raise
            sockets.append(candidate)
            if len(sockets) == count:
                return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()
    raise RuntimeError(f"Could not find {count} free TCP ports in [{start}, {stop})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("count", type=int)
    parser.add_argument("excluded", type=int, nargs="*")
    args = parser.parse_args()
    print(*find_available_ports(args.count, args.excluded))


if __name__ == "__main__":
    main()

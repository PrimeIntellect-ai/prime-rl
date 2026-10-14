"""Fake environment worker: sends real saved frames from a separate process."""

import json
import sys
from pathlib import Path

import zmq

root = Path(sys.argv[3])
meta = json.loads((root / "fixture.json").read_text())
wires = [(root / "frames" / r["file"]).read_bytes() for r in meta["frames"]]
ctx = zmq.Context()
tx = ctx.socket(zmq.PAIR)
tx.setsockopt(zmq.SNDHWM, 8)
tx.connect(sys.argv[1])
count = int(sys.argv[2])
for wire in wires:
    for i in range(count):
        tx.send_multipart([str(i).encode(), b"delta", wire])
for i in range(count):
    tx.send_multipart([str(i).encode(), b"reply", b""])
assert tx.recv() == b"consumed"
tx.close()
ctx.term()

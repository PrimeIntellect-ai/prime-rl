"""Step plans: the orchestrator of an ``sft`` or ``pretrain`` run tells the trainer what to do at each step.

At the start of every step the trainer master sends a ``StepRequest`` and waits for the step's ``StepPlan``. The
orchestrator binds an OS-assigned port next to the trainer master and publishes its address to
``orchestrator.address`` in the attempt's config dir, so two runs on one node never race for a port."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import msgspec
import zmq
import zmq.asyncio

from prime_rl.utils.logger import get_logger

ADDRESS_FILE = "orchestrator.address"

# The launcher starts the orchestrator next to the trainer; it binds within seconds.
ORCHESTRATOR_STARTUP_TIMEOUT_S = 600


class StepRequest(msgspec.Struct):
    """The trainer starting a step."""

    step: int
    stop: Literal["finish", "pause"] | None = None
    """Set on the trainer's last step: it reached ``max_steps`` or a control action paused it."""


class StepPlan(msgspec.Struct):
    """What the trainer does at a step."""

    validate: bool = False
    """Validate the weights the step starts from."""

    broadcast: bool = False
    """Broadcast the weights the step produces to inference; on the last step, after the final checkpoint."""


class StepPlanServer:
    """The orchestrator's end: answers every step request of the trainer."""

    def __init__(self, config_dir: Path):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        path = config_dir / ADDRESS_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(f"tcp://127.0.0.1:{port}")
        tmp.replace(path)

    async def serve(self, plan: Callable[[StepRequest], StepPlan]) -> None:
        """Answer step requests with ``plan`` until the trainer's last step."""
        decoder, encoder = msgspec.msgpack.Decoder(type=StepRequest), msgspec.msgpack.Encoder()
        while True:
            request = decoder.decode(await self.socket.recv())
            await self.socket.send(encoder.encode(plan(request)))
            if request.stop is not None:
                return

    def close(self) -> None:
        """Close once the reply to the trainer's last step is out."""
        self.socket.close()
        self.context.term()


class StepPlanClient:
    """The trainer master's end. Waits for the orchestrator's address, which it publishes once it has bound."""

    def __init__(self, config_dir: Path):
        path = config_dir / ADDRESS_FILE
        get_logger().info(f"Waiting for the orchestrator's address at {path}")
        deadline = time.monotonic() + ORCHESTRATOR_STARTUP_TIMEOUT_S
        while not path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"No orchestrator published its address to {path} within {ORCHESTRATOR_STARTUP_TIMEOUT_S}s. "
                    "The trainer of an sft or pretrain run needs the run's orchestrator: launch it with `uv run sft` "
                    "or `uv run pretrain`, or start `uv run python -m prime_rl.orchestrator.supervised "
                    f"@ {config_dir / 'orchestrator.json'}`."
                )
            time.sleep(0.5)
        self.socket = zmq.Context.instance().socket(zmq.REQ)
        self.socket.connect(path.read_text())
        self.decoder, self.encoder = msgspec.msgpack.Decoder(type=StepPlan), msgspec.msgpack.Encoder()

    def plan(self, request: StepRequest) -> StepPlan:
        self.socket.send(self.encoder.encode(request))
        return self.decoder.decode(self.socket.recv())

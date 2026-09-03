from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, cast

import httpx
import torch
import torch.nn as nn
from modelexpress.client import MxClient
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import NCCLWeightBroadcastConfig
from prime_rl.inference.dynamo import (
    NATIVE_NCCL_ROUTES,
    DynamoAdminClients,
    DynamoDiscoveryPending,
    DynamoWorker,
    client_headers,
    discover_dynamo_workers,
    topology_fingerprint,
)
from prime_rl.trainer.conversion_utils import get_max_layer_num
from prime_rl.trainer.utils import get_world
from prime_rl.transports.weights.base import FINISHED_MARKER, WeightReceiver, WeightSender
from prime_rl.transports.weights.nccl import filter_state_dict_by_layers, preprocess_layer_checkpoint
from prime_rl.transports.weights.nixl.agent import (
    NixlAgent,
    NixlPeer,
    make_agent_name,
    policy_notification,
    set_ucx_env_defaults,
)
from prime_rl.transports.weights.nixl.model_express import ModelExpressSession
from prime_rl.transports.weights.nixl.trainer_tensor_table import TrainerTensorTable
from prime_rl.utils.pathing import wait_for_path
from prime_rl.utils.vlm import get_layer_prefix


class DynamoVLLMWeightSyncClient:
    """Synchronous client required by vLLM's trainer-side transfer engine."""

    def __init__(self, workers: tuple[DynamoWorker, ...], headers: dict[str, str], timeout: float) -> None:
        self.workers = workers
        self.clients = [
            httpx.Client(base_url=worker.system_url.rstrip("/"), headers=headers, timeout=timeout) for worker in workers
        ]

    @staticmethod
    def _validate(response: httpx.Response, path: str) -> None:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Dynamo route {path} returned a non-object response")
        if payload.get("status") == "error":
            raise RuntimeError(payload.get("message", f"Dynamo route {path} failed"))

    def _fanout(self, path: str, bodies: list[dict[str, Any]]) -> None:
        def request(client: httpx.Client, body: dict[str, Any]) -> None:
            self._validate(client.post(path, json=body), path)

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = [executor.submit(request, client, body) for client, body in zip(self.clients, bodies)]
            for future in futures:
                future.result()

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        rank_offset = 1
        bodies = []
        for worker in self.workers:
            if worker.world_size is None:
                raise ValueError("Dynamo native NCCL requires each worker's world_size")
            bodies.append({"init_info": {**init_info, "rank_offset": rank_offset}})
            rank_offset += worker.world_size
        self._fanout("/engine/update/init_weight_transfer_engine", bodies)

    def start_weight_update(self) -> None:
        self._fanout("/engine/update/start_weight_update", [{} for _ in self.clients])

    def update_weights(self, update_info: dict[str, Any]) -> None:
        self._fanout("/engine/update/update_weights", [{"update_info": update_info} for _ in self.clients])

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        self._fanout(
            "/engine/update/finish_weight_update",
            [{"weight_version": weight_version} for _ in self.clients],
        )

    def update_weight_version(self, weight_version: str) -> None:
        self._fanout(
            "/engine/update/update_weight_version",
            [{"new_version": weight_version} for _ in self.clients],
        )


class PrimeWeightSource:
    """Expose Prime checkpoint-format tensors through vLLM's WeightSource protocol."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.model: nn.Module | None = None
        self.dtype = dtype
        self._metadata = None

    def set_model(self, model: nn.Module) -> None:
        if model is not self.model:
            self._metadata = None
        self.model = model

    def _items(self) -> Iterator[tuple[str, Tensor]]:
        if self.model is None:
            raise RuntimeError("Prime weight source has no model")
        state_dict = self.model.state_dict()
        layer_prefix = get_layer_prefix(self.model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        for layer_idx, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            resolved = {}
            for name, tensor in layer_state_dict.items():
                if isinstance(tensor, DTensor):
                    tensor = cast(DTensor, tensor.to(self.dtype)).full_tensor()
                resolved[name] = tensor
            yield from preprocess_layer_checkpoint(self.model, resolved, layer_idx).items()

    def metadata(self):
        from vllm.distributed.weight_transfer import ParamMeta

        if self._metadata is None:
            if self.model is None:
                raise RuntimeError("Prime weight source has no model")
            state_dict = {
                name: torch.empty(
                    tuple(tensor.shape),
                    dtype=self.dtype if isinstance(tensor, DTensor) else tensor.dtype,
                    device="meta",
                )
                for name, tensor in self.model.state_dict().items()
            }
            layer_prefix = get_layer_prefix(self.model.config)
            num_layers = get_max_layer_num(state_dict, layer_prefix)
            metadata = []
            for layer_idx, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
                converted = preprocess_layer_checkpoint(self.model, layer_state_dict, layer_idx)
                metadata.extend(
                    ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in converted.items()
                )
            self._metadata = metadata
        return list(self._metadata)

    def __iter__(self) -> Iterator[tuple[str, Tensor]]:
        yield from self._items()


def _discover(config: NCCLWeightBroadcastConfig) -> tuple[DynamoWorker, ...]:
    if config.dynamo is None:
        raise ValueError("Dynamo native NCCL requires Dynamo discovery configuration")
    headers = client_headers(config.dynamo.headers, config.dynamo.headers_from_env, config.dynamo.api_key_var)
    deadline = time.monotonic() + config.timeout
    previous_fingerprint = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            workers = discover_dynamo_workers(
                config.dynamo.discovery_url,
                config.dynamo.model_name,
                headers=headers,
                timeout=min(30.0, max(1.0, deadline - time.monotonic())),
            )
            discovered_world_size = sum(worker.world_size or 0 for worker in workers)
            if discovered_world_size != config.inference_world_size:
                raise DynamoDiscoveryPending(
                    f"Dynamo world size {discovered_world_size} does not match Prime inference capacity "
                    f"{config.inference_world_size}"
                )
            for worker in workers:
                missing = NATIVE_NCCL_ROUTES.difference(worker.routes)
                if missing:
                    raise DynamoDiscoveryPending(
                        f"Dynamo worker {worker.component}/{worker.instance_id} is missing native NCCL routes: "
                        f"{sorted(missing)}"
                    )
            fingerprint = topology_fingerprint(workers)
            if fingerprint == previous_fingerprint:
                return workers
            previous_fingerprint = fingerprint
        except (DynamoDiscoveryPending, httpx.TransportError) as error:
            previous_fingerprint = None
            last_error = error
        time.sleep(1)
    raise TimeoutError("Dynamo workers did not become ready before trainer initialization") from last_error


class DynamoNCCLWeightSender(WeightSender):
    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(output_dir, config.timeout)
        if config.quantize_in_weight_transfer:
            raise ValueError("Dynamo native NCCL does not support Prime's custom quantized transfer")

        from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

        from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

        self.world = get_world()
        workers = _discover(config)
        assert config.dynamo is not None
        headers = client_headers(config.dynamo.headers, config.dynamo.headers_from_env, config.dynamo.api_key_var)
        self.client = DynamoVLLMWeightSyncClient(workers, headers, config.timeout)
        self.source = PrimeWeightSource(dtype)
        disable_nccl_p2p_if_unavailable()
        self.engine = WeightTransferTrainerFactory.trainer_init(
            NCCLTrainerInitInfo(
                master_address=config.host,
                master_port=config.port,
                world_size=1 + sum(worker.world_size or 0 for worker in workers),
                rank=self.world.rank,
            ),
            client=self.client,
            source=self.source,
        )

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        del step_dir
        self.source.set_model(model)
        self.engine.send_weights()
        if self.world.is_master:
            self.client.update_weight_version(str(step))


class DynamoWeightReceiver(WeightReceiver):
    def __init__(self, broadcast_dir, config, admin_clients, model_name, admin_plane: DynamoAdminClients) -> None:
        super().__init__(broadcast_dir, config, admin_clients, model_name)
        self.admin_plane = admin_plane
        self.nixl_agent: NixlAgent | None = None
        self.nixl_session: ModelExpressSession | None = None
        self.nixl_trainer_peer: NixlPeer | None = None

    def _world_size(self) -> int:
        sizes = [worker.world_size for worker in self.admin_plane.workers]
        if any(size is None for size in sizes):
            raise ValueError("Dynamo in-memory weight transfer requires every worker's world_size")
        return sum(cast(int, size) for size in sizes)

    async def initialize(self) -> None:
        if self.config.type == "nccl":
            discovered = self._world_size()
            if discovered != self.config.inference_world_size:
                raise ValueError(
                    f"Configured inference_world_size={self.config.inference_world_size} does not match Dynamo "
                    f"world size {discovered}"
                )
            for worker in self.admin_plane.workers:
                missing = NATIVE_NCCL_ROUTES.difference(worker.routes)
                if missing:
                    raise ValueError(
                        f"Dynamo worker {worker.component}/{worker.instance_id} is missing native NCCL routes: "
                        f"{sorted(missing)}"
                    )
        elif self.config.type == "nixl":
            discovered = self._world_size()
            if discovered != self.config.inference_world_size:
                raise ValueError(
                    f"Configured inference_world_size={self.config.inference_world_size} does not match Dynamo "
                    f"world size {discovered}"
                )
            rank_offset = 0
            bodies = []
            for worker in self.admin_plane.workers:
                assert worker.world_size is not None
                bodies.append(
                    {
                        "method": "init_broadcaster",
                        "timeout": self.config.timeout,
                        "args": [
                            self.config.host,
                            self.config.port,
                            rank_offset,
                            discovered,
                            self.config.timeout,
                            False,
                            self.config.session_id,
                        ],
                        "kwargs": {},
                    }
                )
                rank_offset += worker.world_size
            await self.admin_plane.fanout_collective(bodies)
            set_ucx_env_defaults(0)
            self.nixl_agent = NixlAgent(make_agent_name("orchestrator", 0))
            self.nixl_session = ModelExpressSession(
                client=MxClient(server_url=f"{self.config.host}:{self.config.port}"),
                role="orchestrator",
                rank=0,
                session_id=self.config.session_id,
                worker_id="orchestrator",
            )
            self.nixl_session.publish(nixl_metadata=self.nixl_agent.get_metadata())

    async def _wait_for_nixl_ready(self, step: int) -> None:
        if self.nixl_agent is None or self.nixl_session is None:
            raise RuntimeError("Dynamo NIXL receiver was not initialized")
        if self.nixl_trainer_peer is None:
            trainer_refs = await asyncio.to_thread(
                self.nixl_session.wait_for,
                "trainer",
                count=1,
                timeout=self.config.timeout,
            )
            trainer_worker = await asyncio.to_thread(self.nixl_session.fetch, trainer_refs[0])
            trainer_table = TrainerTensorTable.decode(trainer_worker.nixl_metadata)
            self.nixl_trainer_peer = self.nixl_agent.add_remote_agent(trainer_table.agents[0].metadata)
            self.nixl_agent.make_connection(self.nixl_trainer_peer)
        await asyncio.to_thread(
            self.nixl_agent.wait_for_notification,
            [self.nixl_trainer_peer],
            policy_notification(step, "ready"),
            timeout=self.config.timeout,
        )

    def _complete_nixl_receive(self, step: int) -> None:
        if self.nixl_agent is None or self.nixl_trainer_peer is None:
            raise RuntimeError("Dynamo NIXL receiver has no trainer peer")
        self.nixl_agent.send_notification(
            self.nixl_trainer_peer,
            policy_notification(step, "complete"),
        )

    async def _wait_for_version(self, step: int) -> None:
        expected = str(step)
        deadline = time.monotonic() + self.config.timeout
        while time.monotonic() < deadline:
            if all(version == expected for version in await self.admin_plane.weight_versions()):
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Dynamo workers did not commit weight version {expected}; engines remain paused")

    async def receive(self, step: int) -> None:
        await self.admin_plane.pause()
        if not await self.admin_plane.is_paused():
            raise RuntimeError("Dynamo did not confirm every pinned worker was paused")

        if self.config.type == "nccl":
            self._ack(step)
        elif self.config.type == "filesystem":
            self._ack(step)
            weights_path = self.step_dir(step)
            await wait_for_path(weights_path / FINISHED_MARKER)
            await self.admin_plane.fanout_collective(
                [
                    {
                        "method": "reload_weights",
                        "timeout": self.config.timeout,
                        "args": [],
                        "kwargs": {"weights_path": weights_path.as_posix()},
                    }
                    for _ in self.admin_plane.workers
                ]
            )
            await self.admin_plane.update_weight_version(str(step))
        elif self.config.type == "nixl":
            self._ack(step)
            await self._wait_for_nixl_ready(step)
            await self.admin_plane.fanout_collective(
                [
                    {
                        "method": "update_weights_from_path",
                        "timeout": self.config.timeout,
                        "args": [None],
                        "kwargs": {},
                    }
                    for _ in self.admin_plane.workers
                ]
            )
            self._complete_nixl_receive(step)
            await self.admin_plane.update_weight_version(str(step))
        else:
            raise ValueError(f"Unsupported Dynamo weight transfer type: {self.config.type}")

        await self._wait_for_version(step)
        await self.admin_plane.resume()

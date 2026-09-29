"""Critic score packing and distributed dummy-model data flow."""

import asyncio
import os
import socket
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import torch
import torch.distributed as dist
import zmq

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.model import freeze_attention_modules
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.trainer.rl.value import ServiceState, _score_batch, _server_handler, _train_batch, pack_score_requests
from prime_rl.transports.batch import setup_batch_receiver, setup_batch_sender
from prime_rl.transports.batch.types import TrainingSample


def test_score_bins_preserve_sequences_and_replica_steps():
    sequences = [[1, 2, 3], [4] * 8, [5, 6], [7] * 7, [8], [9] * 5]
    for dp, cp in ((1, 1), (2, 2), (4, 4)):
        ranks = pack_score_requests(sequences, seq_len=16, dp=dp, cp=cp)
        assert len(ranks) == dp
        assert len({len(bins) for bins in ranks}) == 1
        actual = {}
        for bins in ranks:
            for score_bin in bins:
                assert len(score_bin.token_ids) <= 16
                offset = 0
                for index, length in zip(score_bin.indices, score_bin.lengths, strict=True):
                    actual[index] = score_bin.token_ids[offset : offset + length]
                    offset += length
        assert actual == dict(enumerate(sequences))


def test_score_http_accepts_batches_and_legacy_sequences():
    state = ServiceState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _server_handler(state, max_seq_len=2))
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()

    def serve_one():
        request = state.requests.get(timeout=5)
        request.values = [[float(token) for token in ids] for ids in request.batched_token_ids]
        request.bootstrap_values = [float(ids[-1]) for ids in request.batched_token_ids]
        request.done.set()

    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as client:
            for body, expected in (
                (
                    {"batched_token_ids": [[1, 2], [3]]},
                    {"values": [[1.0, 2.0], [3.0]], "bootstrap_values": [2.0, 3.0]},
                ),
                ({"token_ids": [4, 5]}, {"values": [4.0, 5.0], "bootstrap_value": 5.0}),
            ):
                worker = threading.Thread(target=serve_one)
                worker.start()
                response = client.post("/score", json=body)
                worker.join(timeout=5)
                assert response.status_code == 200 and response.json() == expected
            assert client.post("/score", json={"batched_token_ids": [[1], []]}).status_code == 400
            assert client.post("/score", json={"batched_token_ids": [[1, 2, 3]]}).status_code == 400
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


def test_freeze_attention_follows_modules_instead_of_model_type():
    class MixedBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Linear(2, 2)
            self.linear_attn = torch.nn.Linear(2, 2)
            self.mlp = torch.nn.Linear(2, 2)

    model = MixedBackbone()
    freeze_attention_modules(model)
    assert not any(param.requires_grad for param in model.self_attn.parameters())
    assert not any(param.requires_grad for param in model.linear_attn.parameters())
    assert all(param.requires_grad for param in model.mlp.parameters())


class DummyValueModel:
    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, *, input_ids, position_ids, seq_lens, **kwargs):
        self.calls += 1
        assert input_ids.shape == position_ids.shape
        return {"values": input_ids.float() * 0.25 + position_ids.float() * 0.125}


class TrainableDummyValueModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.01, device="cuda"))

    def forward(self, *, input_ids, position_ids, seq_lens, **kwargs):
        return {"values": input_ids.float() * self.weight + position_ids.float() * 0.0}


def _expected(ids):
    raw = [token * 0.25 + index * 0.125 for index, token in enumerate(ids)]
    return [0.0] + raw[:-1], raw[-1]


def _free_transport_port():
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        port = first.getsockname()[1]
        second.bind(("127.0.0.1", port + 1))
    return port


def _check_training_traffic(critic_dp, world_size, dims):
    rank = dist.get_rank()
    policy_dp = 1 if critic_dp > 1 else world_size
    samples = []
    for index, length in enumerate((9, 7, 5, 3, 2)):
        samples.append(
            TrainingSample(
                token_ids=list(range(10 * index, 10 * index + length)),
                mask=[False] * (length - 1) + [True],
                logprobs=[0.0] * length,
                temperatures=[1.0] * length,
                env_name="test",
                advantages=[0.0] * (length - 1) + [1.0],
                old_values=[0.25] * length,
                value_targets=[0.5] * length,
                value_mask=[False] * (length - 1) + [True],
                trace_id=str(index),
                branch_index=0,
            )
        )
    policy_grid = prepare_batch(samples, 16, policy_dp, sum, pad_to_multiple_of=world_size // policy_dp)
    critic_grid = prepare_batch(samples, 8, critic_dp, sum, pad_to_multiple_of=world_size // critic_dp, for_value=True)
    ports = [_free_transport_port(), _free_transport_port()] if rank == 0 else [0, 0]
    dist.broadcast_object_list(ports, src=0, device=torch.device("cuda"))
    policy_transport = ZMQTransportConfig(host="127.0.0.1", port=ports[0])
    critic_transport = ZMQTransportConfig(host="127.0.0.1", port=ports[1])
    senders = (
        [
            setup_batch_sender(Path("/tmp"), policy_dp, 0, policy_transport),
            setup_batch_sender(Path("/tmp"), critic_dp, 0, critic_transport),
        ]
        if rank == 0
        else []
    )
    policy_rank = rank // (world_size // policy_dp)
    critic_rank = rank // (world_size // critic_dp)
    receivers = [
        setup_batch_receiver(Path("/tmp"), policy_rank, 0, policy_transport),
        setup_batch_receiver(Path("/tmp"), critic_rank, 0, critic_transport),
    ]
    for receiver in receivers:
        receiver.socket.setsockopt(zmq.RCVTIMEO, 20000)
    dist.barrier()
    if rank == 0:

        async def send_both():
            await asyncio.wait_for(senders[0].send(policy_grid), 20)
            await asyncio.wait_for(senders[1].send(critic_grid), 20)

        asyncio.run(send_both())
    assert receivers[0].receive() == policy_grid[policy_rank]
    received_value = receivers[1].receive()
    assert received_value == critic_grid[critic_rank]
    tensor_batches = [DataLoader._micro_batch_to_tensor(None, batch) for batch in received_value]
    model = TrainableDummyValueModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = SimpleNamespace(
        model=SimpleNamespace(cp_style="ring", lora=None, full_offload=None),
        optim=SimpleNamespace(max_norm=None),
        updates_per_step=1,
    )
    loss, _ = _train_batch(model, optimizer, scheduler, None, tensor_batches, dims, config, head_only=False)
    assert torch.isfinite(torch.tensor(loss))
    assert scheduler.last_epoch == 1
    dist.barrier()
    for receiver in receivers:
        receiver.close()
    for sender in senders:
        sender.close()


def _run_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    cp = int(os.environ["TEST_VALUE_CP"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    dims = ParallelDims(dp_replicate=1, dp_shard=world_size // cp, cp=cp, pp=1, ep=1, world_size=world_size)
    _check_training_traffic(world_size // cp, world_size, dims)
    batches = [
        [[1, 2, 3], list(range(11, 19)), [21], list(range(31, 39)), [40, 41]],
        [[2, 3, 4]],
        [list(range(100 + i * 10, 100 + i * 10 + (i % 8) + 1)) for i in range(40)],
    ]
    for sequences in batches:
        model = DummyValueModel()
        result = _score_batch(model, sequences, dims, 16, "ring", False)
        counts = [None] * world_size
        dist.all_gather_object(counts, model.calls)
        assert len(set(counts)) == 1
        if dist.get_rank() == 0:
            assert result == (
                [_expected(ids)[0] for ids in sequences],
                [_expected(ids)[1] for ids in sequences],
            )
        else:
            assert result is None
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_distributed()

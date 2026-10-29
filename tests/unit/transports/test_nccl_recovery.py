import multiprocessing as mp
import os
import signal
import socket
import time

import pytest

pytestmark = pytest.mark.gpu


def _connect(rank, port, pipe, fail_init=False):
    import torch
    from vllm.distributed.utils import StatelessProcessGroup

    from prime_rl.transports.weights.nccl_communicator import WeightBroadcastCommunicator

    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(0.08, rank)
    group = StatelessProcessGroup.create(host="127.0.0.1", port=port, rank=rank, world_size=2, store_timeout=15)
    if fail_init:
        from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

        group.broadcast_obj(NCCLLibrary().ncclGetUniqueId(), src=0)
        pipe.send("before_init")
        pipe.recv()
    return WeightBroadcastCommunicator(group, rank, 5), group


def _sender(pipe, port, failure):
    import torch

    comm, group = _connect(0, port, pipe, failure == "initialization")
    data = torch.full((1024 * 1024,), 23.0, device="cuda:0")
    comm.broadcast(data, 0)
    pipe.send("first_sent")
    if failure:
        pipe.recv()
    data.fill_(42)
    comm.broadcast(data, 0)
    pipe.send("second_sent")
    pipe.recv()
    comm.close()


def _receiver(pipe):
    import torch

    for _ in range(2):
        port, failure = pipe.recv()
        if failure == "initialization":
            try:
                _connect(1, port, pipe)
            except TimeoutError:
                pipe.send("failed")
            else:
                raise AssertionError("Missing sender joined NCCL initialization")
        else:
            comm, group = _connect(1, port, pipe)
            data = torch.zeros(1024 * 1024, device="cuda:1")
            comm.broadcast(data, 0)
            assert data.mean().item() == 23
            pipe.send(("first_received", os.getpid()))
            pipe.send("receiving_second")
            try:
                comm.broadcast(data, 0)
            except (RuntimeError, TimeoutError) as error:
                assert failure
                assert "NCCL" in str(error)
                pipe.send("failed")
                with pytest.raises(RuntimeError, match="closed"):
                    comm.broadcast(data, 0)
            else:
                assert not failure
                assert data.mean().item() == 42
                pipe.send(("recovered", os.getpid()))
            comm.close()
            comm.close()
        # A communication failure must not poison the surviving CUDA context.
        matrix = torch.ones((128, 128), device="cuda:1")
        assert (matrix @ matrix).mean().item() == 128
        pipe.send("cuda_healthy")


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _recv(pipe):
    assert pipe.poll(45), "NCCL recovery child did not respond"
    return pipe.recv()


@pytest.mark.parametrize("failure", ["killed", "stalled", "initialization"])
def test_nccl_reconnects_after_sender_failure(monkeypatch, failure):
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("NCCL recovery test requires two GPUs")
    # Exercise network disconnects even on hosts with peer-to-peer GPU access.
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("NCCL_SHM_DISABLE", "1")
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    receiver = ctx.Process(target=_receiver, args=(child,))
    processes = [receiver]
    receiver.start()
    try:
        for fault in (failure, None):
            port = _port()
            sender_pipe, sender_child = ctx.Pipe()
            sender = ctx.Process(target=_sender, args=(sender_child, port, fault))
            processes.append(sender)
            sender.start()
            parent.send((port, fault))
            if fault == "initialization":
                assert _recv(sender_pipe) == "before_init"
            else:
                assert _recv(sender_pipe) == "first_sent"
                assert _recv(parent) == ("first_received", receiver.pid)
                assert _recv(parent) == "receiving_second"
            if fault:
                time.sleep(0.2)
                if fault == "killed":
                    sender.kill()
                else:
                    os.kill(sender.pid, signal.SIGSTOP)
                assert _recv(parent) == "failed"
                sender.kill()
            else:
                assert _recv(sender_pipe) == "second_sent"
                assert _recv(parent) == ("recovered", receiver.pid)
                sender_pipe.send("close")
            assert _recv(parent) == "cuda_healthy"
            sender.join(10)
            if not fault:
                assert sender.exitcode == 0
        receiver.join(10)
        assert receiver.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(10)


def test_aborted_layerwise_reload_preserves_storage_and_discards_pending_weights():
    import torch
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    from prime_rl.inference.vllm.worker.weight_transfer import abort_layerwise_reload

    with torch.device("cuda:0"), torch.no_grad():
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        model[1].register_buffer("alias", model[1].weight.data, persistent=False)
        original = dict(model.named_parameters())
        alias = model[1].alias
        record_metadata_for_reloading(model)
        initialize_layerwise_reload(model)
        # One layer completes; another retains a deferred weight while awaiting its bias.
        for parameter in list(model[0].parameters()):
            parameter.weight_loader(parameter, torch.full(parameter.shape, 3.0))
        parameter = model[1].weight
        parameter.weight_loader(parameter, torch.full(parameter.shape, 5.0))
        assert get_layerwise_info(model[1]).loaded_weights
        abort_layerwise_reload(model)
        assert all(parameter is original[name] for name, parameter in model.named_parameters())
        assert model[1].alias is alias
        assert "alias" in model[1]._non_persistent_buffers_set
        assert all(not get_layerwise_info(layer).loaded_weights for layer in model.modules())
        assert all(not get_layerwise_info(layer).can_load() for layer in model.modules())
        # This is cleanup, not rollback: the completed layer's values have changed.
        assert torch.all(model[0].weight == 3)
        initialize_layerwise_reload(model)
        for layer in model:
            for parameter in list(layer.parameters()):
                parameter.weight_loader(parameter, torch.full(parameter.shape, 7.0))
        finalize_layerwise_reload(model, None)
        assert all(parameter is original[name] for name, parameter in model.named_parameters())
        assert all(torch.all(parameter == 7) for parameter in model.parameters())

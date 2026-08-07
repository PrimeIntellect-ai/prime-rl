from prime_rl.inference.vllm.worker import nccl


def test_receiver_preserves_state_dict_boundaries(monkeypatch):
    receiver = object.__new__(nccl.NCCLWeightBroadcastReceiver)
    receiver.communicator = object()
    streams = [iter([("layer.0", 0)]), iter([("layer.1", 1)])]

    monkeypatch.setattr(nccl, "receive_integer", lambda _communicator: len(streams))
    monkeypatch.setattr(nccl, "receive_state_dict", lambda _communicator: streams.pop(0))

    assert [list(stream) for stream in receiver.receive_state_dicts()] == [
        [("layer.0", 0)],
        [("layer.1", 1)],
    ]

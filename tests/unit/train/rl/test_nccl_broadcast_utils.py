import torch

from prime_rl.trainer.rl.broadcast.nccl import filter_state_dict_by_layers


def test_filter_state_dict_by_layers_includes_layer_zero():
    state_dict = {
        "model.embed_tokens.weight": torch.zeros(1),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "model.layers.1.self_attn.q_proj.weight": torch.zeros(1),
    }

    chunks = list(filter_state_dict_by_layers(state_dict, num_layers=2))

    assert chunks[0][0] == 0
    assert "model.embed_tokens.weight" in chunks[0][1]

    layer_chunks = {layer_id: layer_state for layer_id, layer_state in chunks[1:]}
    assert "model.layers.0.self_attn.q_proj.weight" in layer_chunks[0]
    assert "model.layers.1.self_attn.q_proj.weight" in layer_chunks[1]

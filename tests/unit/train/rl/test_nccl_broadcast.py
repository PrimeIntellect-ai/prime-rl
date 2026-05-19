import multiprocessing as mp

import pytest
import torch

from prime_rl.inference.vllm.worker.nccl import NCCLWeightBroadcastReceiver
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeConfig
from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeForCausalLM as PrimeRLQwen3_5MoeForCausalLM
from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcastSender, preprocess_layer_checkpoint
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def test_nccl_preprocess_converts_mtp_non_layer_chunk_to_hf_keys():
    config = Qwen3_5MoeConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        use_grouped_mm=False,
        mtp_num_hidden_layers=1,
    )
    config._attn_implementation = "sdpa"
    with default_dtype(torch.float32):
        model = PrimeRLQwen3_5MoeForCausalLM._from_config(config)
    inject_prime_lm_head(model, chunk_size=None)
    non_layer_state_dict = {
        key: value for key, value in model.state_dict().items() if not key.startswith("model.layers.")
    }

    converted = preprocess_layer_checkpoint(model, non_layer_state_dict, layer_idx=-1)

    assert "mtp.fc.weight" in converted
    assert "mtp.layers.0.self_attn.q_proj.weight" in converted
    assert not any(key.startswith("mtp_layers.") for key in converted)


@pytest.mark.skip(reason="Skipping NCCL broadcast as it fail only in ci")
def test_nccl_broadcast(free_port):
    host = "localhost"
    free_port = free_port()

    def send():
        device = torch.device(f"cuda:{0}")
        nccl_broadcast = NCCLWeightBroadcastSender(
            host=host, port=free_port, rank=0, world_size=2, device=device, timeout=10
        )

        class SubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.layers = torch.nn.ModuleList([torch.nn.Linear(10, 10) for _ in range(10)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SubModel()

            def forward(self, x):
                return self.model(x)

        model = Model().to(device)
        for param in model.parameters():
            param.data = torch.ones_like(param.data)

        nccl_broadcast.broadcast_weights(model, step=0)

    def receive():
        device = torch.device(f"cuda:{1}")
        nccl_broadcast = NCCLWeightBroadcastReceiver(
            host=host, port=free_port, rank=1, world_size=2, device=device, timeout=10
        )

        for key, value in nccl_broadcast.receive_state_dict():
            assert value.allclose(torch.ones_like(value))

    processes = [mp.Process(target=send), mp.Process(target=receive)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
        assert process.exitcode == 0, f"Process {process.name} exited with code {process.exitcode}"

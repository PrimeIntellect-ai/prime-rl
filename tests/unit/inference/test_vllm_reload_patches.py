import torch

from prime_rl.inference import patches


class AliasedKernelLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
        self.register_buffer("conv_weights", self.weight.view(-1))
        self.register_buffer("running", torch.zeros(2, dtype=torch.float32))


def _layer_reloading_info(layer: AliasedKernelLayer):
    from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo

    return LayerReloadingInfo(
        restore_metadata=({}, {}),
        restore_device=torch.device("cpu"),
        kernel_tensors=(
            {"weight": layer._parameters["weight"]},
            {
                "conv_weights": layer._buffers["conv_weights"],
                "running": layer._buffers["running"],
            },
        ),
    )


def _install_processed_tensors(layer: AliasedKernelLayer, aliased_buffer_value: float = 99.0):
    layer._parameters["weight"] = torch.nn.Parameter(torch.full_like(layer.weight, 7.0))
    layer._buffers["conv_weights"] = torch.full_like(layer.conv_weights, aliased_buffer_value)
    layer._buffers["running"] = torch.full_like(layer.running, 3.0)


def test_vllm_layerwise_reload_alias_buffer_patch_preserves_parameter_storage():
    from vllm.model_executor.model_loader.reload import layerwise

    layer = AliasedKernelLayer()
    info = _layer_reloading_info(layer)
    _install_processed_tensors(layer)

    patches.monkey_patch_vllm_layerwise_reload_alias_buffers()
    layerwise._copy_and_restore_kernel_tensors(layer, info)

    assert torch.equal(layer.weight, torch.full_like(layer.weight, 7.0))
    assert torch.equal(layer.conv_weights, layer.weight.view(-1))
    assert torch.equal(layer.running, torch.full_like(layer.running, 3.0))


def test_vllm_0_22_layerwise_reload_copy_order_reproduces_alias_corruption():
    layer = AliasedKernelLayer()
    info = _layer_reloading_info(layer)
    original_param = info.kernel_tensors[0]["weight"]
    original_buffer = info.kernel_tensors[1]["conv_weights"]

    _install_processed_tensors(layer, aliased_buffer_value=float("nan"))

    original_param.data.copy_(layer.weight)
    original_buffer.data.copy_(layer.conv_weights)

    assert torch.isnan(original_param).all()

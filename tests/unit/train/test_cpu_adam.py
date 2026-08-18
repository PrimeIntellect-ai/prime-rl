import torch

from prime_rl.trainer.optim.cpu_adam import (
    adamw_step,
    add_bfloat16_,
    copy_bfloat16_,
    copy_or_add_bfloat16_multi_,
    sign_sgd_step,
)
from prime_rl.trainer.optim.offload import _cast_full_offload_compute_parameters
from prime_rl.trainer.sign_sgd import SignSGD


def test_full_offload_preserves_per_parameter_compute_dtypes():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False))
    bfloat16_param, float32_param = model.parameters()
    float32_param.data.fill_(1.0001)
    original_float32 = float32_param.detach().clone()
    model.register_buffer("statistics", torch.tensor([1.0001], dtype=torch.float32))
    original_statistics = model.statistics.detach().clone()
    policy = {
        id(bfloat16_param): (torch.bfloat16, torch.bfloat16),
        id(float32_param): (torch.float32, torch.float32),
    }

    _cast_full_offload_compute_parameters(model, policy)

    assert bfloat16_param.dtype == torch.bfloat16
    assert float32_param.dtype == torch.float32
    torch.testing.assert_close(float32_param, original_float32, rtol=0, atol=0)
    torch.testing.assert_close(model.statistics, original_statistics, rtol=0, atol=0)


def test_native_cpu_adamw_matches_fused_torch_and_preserves_gradients():
    torch.manual_seed(0)
    accumulated = torch.randn(1025)
    contribution = torch.randn(1025).bfloat16()
    expected_accumulation = accumulated + contribution.float()
    add_bfloat16_(accumulated, contribution)
    torch.testing.assert_close(accumulated, expected_accumulation, rtol=0, atol=0)
    copy_bfloat16_(accumulated, contribution)
    torch.testing.assert_close(accumulated, contribution.float(), rtol=0, atol=0)
    destinations = [torch.randn(1025), torch.randn(513)]
    sources = [torch.randn(1025).bfloat16(), torch.randn(513).bfloat16()]
    expected = [destinations[0] + sources[0].float(), sources[1].float()]
    copy_or_add_bfloat16_multi_(destinations, sources, [True, False])
    for actual, expected_tensor in zip(destinations, expected):
        torch.testing.assert_close(actual, expected_tensor, rtol=0, atol=0)

    shapes = [(17,), (33, 65), (257, 129)]
    initial = [torch.randn(shape) for shape in shapes]
    native_params = [tensor.clone() for tensor in initial]
    float_gradient_params = [tensor.clone() for tensor in initial]
    torch_params = [torch.nn.Parameter(tensor.clone()) for tensor in initial]
    exp_avgs = [torch.zeros_like(tensor) for tensor in native_params]
    exp_avg_sqs = [torch.zeros_like(tensor) for tensor in native_params]
    state_steps = [torch.zeros((), dtype=torch.float32) for _ in native_params]
    float_gradient_exp_avgs = [torch.zeros_like(tensor) for tensor in native_params]
    float_gradient_exp_avg_sqs = [torch.zeros_like(tensor) for tensor in native_params]
    float_gradient_state_steps = [torch.zeros((), dtype=torch.float32) for _ in native_params]
    float_compute_params = [torch.empty_like(tensor) for tensor in native_params]
    compute_dtypes = [torch.bfloat16, torch.float32, torch.bfloat16]
    compute_params = [torch.empty_like(tensor, dtype=dtype) for tensor, dtype in zip(native_params, compute_dtypes)]
    torch_optimizer = torch.optim.AdamW(
        torch_params,
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )

    for iteration in range(1, 6):
        gradient_scale = 1.0 / (iteration * 7 + 3)
        gradients = [torch.randn_like(tensor).bfloat16() for tensor in native_params]
        native_gradients = [gradient.clone() for gradient in gradients]
        for param, gradient in zip(torch_params, gradients):
            param.grad = gradient.float()
        torch_optimizer.grad_scale = torch.tensor(1.0 / gradient_scale)
        torch_optimizer.found_inf = torch.zeros((), dtype=torch.float32)
        torch_optimizer.step()

        adamw_step(
            float_gradient_params,
            [gradient.float() for gradient in gradients],
            float_gradient_exp_avgs,
            float_gradient_exp_avg_sqs,
            float_gradient_state_steps,
            float_compute_params,
            lr=3e-4,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            eps=1e-8,
            gradient_scale=gradient_scale,
        )
        adamw_step(
            native_params,
            native_gradients,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
            compute_params,
            lr=3e-4,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            eps=1e-8,
            gradient_scale=gradient_scale,
        )

        for actual, expected in zip(native_gradients, gradients):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for compute_param, native_param in zip(compute_params, native_params):
            torch.testing.assert_close(compute_param, native_param.to(compute_param.dtype), rtol=0, atol=0)
        for compute_param, native_param in zip(float_compute_params, float_gradient_params):
            torch.testing.assert_close(compute_param, native_param, rtol=0, atol=0)

    for actual, expected in zip(native_params, float_gradient_params):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(exp_avgs, float_gradient_exp_avgs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(exp_avg_sqs, float_gradient_exp_avg_sqs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(state_steps, float_gradient_state_steps):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    for native_param, torch_param in zip(native_params, torch_params):
        torch.testing.assert_close(native_param, torch_param, rtol=1e-6, atol=1e-7)
    for index, torch_param in enumerate(torch_params):
        state = torch_optimizer.state[torch_param]
        torch.testing.assert_close(exp_avgs[index], state["exp_avg"], rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(exp_avg_sqs[index], state["exp_avg_sq"], rtol=1e-6, atol=1e-8)
        assert state_steps[index].item() == state["step"].item() == 5


def test_native_cpu_sign_sgd_matches_python_reference():
    torch.manual_seed(0)
    # Sizes deliberately off vector-width multiples to exercise the scalar tail.
    shapes = [(17,), (33, 65), (257, 129), (1000,)]
    gradient_dtypes = [torch.bfloat16, torch.float32, torch.bfloat16, torch.float32]
    compute_dtypes = [torch.bfloat16, torch.float32, torch.bfloat16, torch.bfloat16]
    for weight_decay in (0.0, 0.1):
        initial = [torch.randn(shape) for shape in shapes]
        native_params = [tensor.clone() for tensor in initial]
        reference_params = [torch.nn.Parameter(tensor.clone()) for tensor in initial]
        compute_params = [torch.empty_like(tensor, dtype=dtype) for tensor, dtype in zip(native_params, compute_dtypes)]
        reference = SignSGD(reference_params, lr=3e-4, weight_decay=weight_decay)
        for iteration in range(1, 6):
            gradient_scale = 1.0 / (iteration * 7 + 3)
            gradients = []
            for tensor, dtype in zip(native_params, gradient_dtypes):
                gradient = torch.randn_like(tensor).to(dtype)
                gradient.view(-1)[::7] = 0.0  # exercise sign(0) == 0
                gradients.append(gradient)
            # The reference sees scaled FP32 gradients; the native kernel consumes the raw
            # unscaled gradients because sign(scale * g) == sign(g) for scale > 0.
            for param, gradient in zip(reference_params, gradients):
                param.grad = gradient.float() * gradient_scale
            reference.step()
            native_gradients = [gradient.clone() for gradient in gradients]
            sign_sgd_step(
                native_params,
                native_gradients,
                compute_params,
                lr=3e-4,
                weight_decay=weight_decay,
            )
            for actual, expected in zip(native_gradients, gradients):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for native_param, reference_param in zip(native_params, reference_params):
                torch.testing.assert_close(native_param, reference_param, rtol=0, atol=0)
            for compute_param, native_param in zip(compute_params, native_params):
                torch.testing.assert_close(compute_param, native_param.to(compute_param.dtype), rtol=0, atol=0)

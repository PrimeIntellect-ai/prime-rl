import torch

from prime_rl.trainer.cpu_adam import adamw_step, add_bfloat16_


def test_native_cpu_adamw_matches_fused_torch_and_preserves_gradients():
    torch.manual_seed(0)
    accumulated = torch.randn(1025)
    contribution = torch.randn(1025).bfloat16()
    expected_accumulation = accumulated + contribution.float()
    add_bfloat16_(accumulated, contribution)
    torch.testing.assert_close(accumulated, expected_accumulation, rtol=0, atol=0)

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
    compute_params = [torch.empty_like(tensor, dtype=torch.bfloat16) for tensor in native_params]
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
            torch.testing.assert_close(compute_param, native_param.bfloat16(), rtol=0, atol=0)

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

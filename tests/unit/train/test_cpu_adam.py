import torch

from prime_rl.trainer.cpu_adam import adamw_step


def test_native_cpu_adamw_matches_fused_torch_and_preserves_gradients():
    torch.manual_seed(0)
    shapes = [(17,), (33, 65), (257, 129)]
    initial = [torch.randn(shape) for shape in shapes]
    native_params = [tensor.clone() for tensor in initial]
    torch_params = [torch.nn.Parameter(tensor.clone()) for tensor in initial]
    exp_avgs = [torch.zeros_like(tensor) for tensor in native_params]
    exp_avg_sqs = [torch.zeros_like(tensor) for tensor in native_params]
    state_steps = [torch.zeros((), dtype=torch.float32) for _ in native_params]
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
        gradients = [torch.randn_like(tensor) for tensor in native_params]
        native_gradients = [gradient.clone() for gradient in gradients]
        for param, gradient in zip(torch_params, gradients):
            param.grad = gradient.clone()
        torch_optimizer.grad_scale = torch.tensor(1.0 / gradient_scale)
        torch_optimizer.found_inf = torch.zeros((), dtype=torch.float32)
        torch_optimizer.step()

        adamw_step(
            native_params,
            native_gradients,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
            lr=3e-4,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            eps=1e-8,
            gradient_scale=gradient_scale,
        )

        for actual, expected in zip(native_gradients, gradients):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    for native_param, torch_param in zip(native_params, torch_params):
        torch.testing.assert_close(native_param, torch_param, rtol=1e-6, atol=1e-7)
    for index, torch_param in enumerate(torch_params):
        state = torch_optimizer.state[torch_param]
        torch.testing.assert_close(exp_avgs[index], state["exp_avg"], rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(exp_avg_sqs[index], state["exp_avg_sq"], rtol=1e-6, atol=1e-8)
        assert state_steps[index].item() == state["step"].item() == 5

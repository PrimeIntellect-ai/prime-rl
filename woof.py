import torch
from triton.testing import do_bench


def main():
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    N = 2**14
    A = torch.randn(N, N, dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(2, N, N, dtype=torch.bfloat16, requires_grad=True)

    def foo():
        # C = torch.matmul(A, B)
        C = torch._grouped_mm(A, B, offs=torch.tensor([N // 2, N], dtype=torch.int32))
        torch.autograd.grad(C, [A, B], grad_outputs=torch.ones_like(C), create_graph=False, retain_graph=False)

    def bar():
        # C = torch.matmul(A, B.T)
        C = torch._grouped_mm(A, B.transpose(-2, -1), offs=torch.tensor([N // 2, N], dtype=torch.int32))
        torch.autograd.grad(C, [A, B], grad_outputs=torch.ones_like(C), create_graph=False, retain_graph=False)

    # test_b_ms = do_bench(bar, warmup=10, rep=2000)
    test_b_ms = do_bench(bar, warmup=10, rep=2000)
    test_a_ms = do_bench(foo, warmup=10, rep=2000)
    print(f"test_a_ms: {test_a_ms}, test_b_ms: {test_b_ms}")


if __name__ == "__main__":
    main()

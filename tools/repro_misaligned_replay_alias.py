"""torch.compile returns a wrong alias of a misaligned input once the graph has dynamic shapes.

An int32 index tensor `base` of shape (1, S, L, K) with K = 6 is sliced per layer, `base[:, :, l, :]`.
Slice l starts 24 * l bytes into the storage, so odd l is not 16-byte aligned. The compiled function
returns a view of that slice. From the second S onward (automatic dynamic shapes), the returned view of
a misaligned slice should equal the eager one but instead aliases the start of `base` (layer 0).

Run on one CUDA device: TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 uv run python tools/repro_misaligned_replay_alias.py
"""

import torch

L, K, E = 4, 6, 16
SEQ_LENS = [57, 91, 120, 33, 64]


def router_like(scores, routed):
    rows = routed.reshape(-1, routed.shape[-1])
    return scores.gather(1, rows).sum(-1), rows


def run(label, compile_kwargs, make_contiguous=False):
    torch._dynamo.reset()
    compiled = torch.compile(router_like, **compile_kwargs)
    failures = []
    for seq_len in SEQ_LENS:
        base = torch.randint(0, E, (1, seq_len, L, K), dtype=torch.int32, device="cuda")
        scores = torch.randn(seq_len, E, device="cuda")
        for layer in range(L):
            routed = base[:, :, layer, :]
            if make_contiguous:
                routed = routed.contiguous()
            _, eager_rows = router_like(scores, routed)
            _, compiled_rows = compiled(scores, routed)
            if not torch.equal(eager_rows, compiled_rows):
                aliases_layer0 = torch.equal(compiled_rows, base[:, :, 0, :].reshape(-1, K))
                failures.append((seq_len, layer, routed.storage_offset() * 4 % 16, aliases_layer0))
    print(f"{label}: {len(failures)} mismatches (S, layer, byte offset mod 16, equals layer 0) {failures}")
    return failures


def main():
    print(f"torch {torch.__version__} git {torch.version.git_version} on {torch.cuda.get_device_name()}")
    bug = run("strided slice, dynamic shapes", {})
    contiguous = run("contiguous slice (control)", {}, make_contiguous=True)
    static = run("strided slice, dynamic=False (control)", {"dynamic": False})
    aot_eager = run("strided slice, backend=aot_eager (control)", {"backend": "aot_eager"})
    print("BUG REPRODUCED" if bug and not (contiguous or static or aot_eager) else "bug not reproduced cleanly")


if __name__ == "__main__":
    main()

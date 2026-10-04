import torch
from dion import Muon as DionMuon
from dion.muon import muon_update_batch_async, zeropower_via_newtonschulz5
from dion.opt_utils import AsyncTask
from torch import Tensor
from torch.distributed.tensor import DTensor

# The quintic coefficients of Dion's dense reference, zeropower_via_newtonschulz5.
NS_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def get_quack_gemm_symmetric():
    """QuACK's symmetric GEMM, or None when it is not installed or the GPU is unsupported (it needs SM80+)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        return None
    try:
        from quack.gemm_interface import gemm_symmetric
    except ImportError:
        return None
    return gemm_symmetric


@torch.no_grad()
def newton_schulz_symmetric(G: Tensor, epsilon: float | Tensor = 1e-7) -> Tensor:
    """Dion's Newton-Schulz, with the two symmetric products (X Xᵀ and A·A) on QuACK's symmetric GEMM.

    A symmetric GEMM computes one triangle of the output and mirrors it, so it does about half the
    work of a dense one. Falls back to the dense reference where QuACK cannot run.
    """
    gemm_symmetric = get_quack_gemm_symmetric() if G.is_cuda else None
    # QuACK needs 16-byte aligned rows, i.e. both matrix dimensions divisible by 8 in bf16.
    if gemm_symmetric is None or G.size(-1) % 8 or G.size(-2) % 8:
        return zeropower_via_newtonschulz5(G, epsilon=epsilon)

    X = G.to(dtype=torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)).contiguous()
    addmm = torch.baddbmm if X.ndim == 3 else torch.addmm
    for a, b, c in NS_COEFFICIENTS:
        A = gemm_symmetric(X, X.mT)
        B = gemm_symmetric(A, A, C=A, alpha=c, beta=b)
        X = addmm(X, B, X, beta=a)
    return X.mT if transposed else X


def holds_whole_matrices(param: Tensor, flatten: bool) -> bool:
    """Whether every shard of ``param`` is a stack of whole matrices: a 3D+ batch sharded only on dim 0."""
    return (
        isinstance(param, DTensor)
        and param.ndim >= 3
        and not flatten
        and all(placement.dim == 0 for placement in param.placements if placement.is_shard())
    )


class Muon(DionMuon):
    """Dion's Muon, except that sharded parameters whose shards are whole matrices update in place.

    Dion all-to-alls every sharded parameter so that one rank holds the whole tensor (and, under EP,
    zero-pads the rank's expert shard to the full expert count). An expert stack
    ``[experts, fan_out, fan_in]`` sharded only over the expert axis already holds whole matrices on
    every rank, and Newton-Schulz treats the leading axis as a batch, so it needs no communication.
    """

    def _create_muon_tasks(self, param_groups: list[dict], algo_name: str = "muon"):
        def split(group: dict, local: bool) -> dict:
            params = [p for p in group["params"] if holds_whole_matrices(p, group["flatten"]) == local]
            return {**group, "params": params}

        for group in param_groups:
            for param in split(group, local=True)["params"]:
                if param.grad is None:
                    continue
                state = self._get_or_initialize_state(param, algo_name)
                yield AsyncTask(
                    muon_update_batch_async(
                        X=[param],
                        G=[param.grad],
                        M=[state["momentum"]],
                        lr=torch.tensor(group["lr"]),
                        momentum=torch.tensor(group["mu"]),
                        weight_decay=torch.tensor(group["weight_decay"]),
                        epsilon=torch.tensor(group["epsilon"]),
                        nesterov=group["nesterov"],
                        flatten=group["flatten"],
                        adjust_lr=group["adjust_lr"],
                        device_rank=0,
                        world_size=1,
                        newton_schulz_func=self._newton_schulz_func,
                        partitions=self._matrix_partitions.get(param),
                    )
                )
        yield from super()._create_muon_tasks([split(group, local=False) for group in param_groups], algo_name)

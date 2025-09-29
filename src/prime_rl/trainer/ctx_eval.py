import torch
import torch.distributed as dist
import numpy as np

from collections import defaultdict
from contextlib import nullcontext
from jaxtyping import Float

from prime_rl.utils.logger import get_logger
from prime_rl.trainer.config import ContextWiseEvalConfig


def log_binning(data, num_bins=100):
    data = np.asarray(data)
    n = len(data)
    # Create log-spaced bin edges from index=1 up to index=n
    # (we'll floor them to int; add 1 if you prefer 1-based edges)
    bin_edges = np.logspace(
        start=np.log10(1),
        stop=np.log10(n),
        num=num_bins + 1
    )
    # Convert bin edges to integer indices; clamp to [0, n]
    bin_edges = np.floor(bin_edges).astype(int)
    bin_edges[0] = 0  # ensure the first edge is at index 0
    bin_edges[-1] = n # ensure the final edge is exactly the end
    results = []
    for i in range(num_bins):
        start = bin_edges[i]
        end   = bin_edges[i + 1]
        if start < end:
            avg = data[start:end].mean().item()
        else:
            # If edges coincide (can happen at very small bin sizes),
            # just take the single point or skip, depending on preference
            avg = data[start].item()
        results.append(avg)
    return results


# TODO: fix multi-gpu hangs
def setup_context_wise_eval(config: ContextWiseEvalConfig, micro_batch_size: int):

    assert config is None or (config.interval <= 0 or config.interval >= config.eval_steps), "interval must be greater than 0 and less than eval_steps"
    context_wise_loss = None

    @torch.no_grad()
    def context_wise_eval(loss: Float[torch.Tensor, "batch seq"], step: int, micro_batch: int):
        nonlocal context_wise_loss
        logger = get_logger()
        if config is None or (config.interval <= 0 or not (step % config.interval < config.eval_steps)):
            return None
        
        logger.info(f"Context-wise evaluation at step {step}")
        context_wise_loss = loss.mean(dim=0) if context_wise_loss is None else context_wise_loss + loss.mean(dim=0)

        if micro_batch == micro_batch_size - 1 and step % config.eval_steps == (config.eval_steps - 1):
            context_wise_loss = context_wise_loss / (micro_batch_size * config.eval_steps)
            logger.info(f"Averaging context-wise loss across ranks: {context_wise_loss}")
            dist.all_reduce(context_wise_loss, op=dist.ReduceOp.AVG)

            block_size = context_wise_loss.shape[0]
            if config.bins < 0:
                ctx_bins = np.arange(block_size).tolist()
                ctx_losses = context_wise_loss.detach().cpu().to(torch.float32).numpy().tolist()
            else:
                ctx_bins = log_binning(np.arange(block_size), config.bins)
                ctx_losses = log_binning(context_wise_loss.detach().cpu().to(torch.float32).numpy(), config.bins)
            tail_loss = ctx_losses[-1]
            best_loss = min(ctx_losses)
            context_wise_loss = None

            return {
                "iter": step,
                "train_contextwise_bins": ctx_bins,
                "train_contextwise_losses": ctx_losses,
                "tail_loss": tail_loss,
                "best_loss": best_loss,
            }

        return None
        
    return context_wise_eval
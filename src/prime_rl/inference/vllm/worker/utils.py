from typing import Iterator, Tuple

import torch
from prime_rl.trainer.weights import quantize_param, should_quantize_weight

def quantize_weights_iterator(iterator: Iterator[Tuple[str, torch.Tensor]]) -> Iterator[Tuple[str, torch.Tensor]]:
    """Iterate over weights and quantize them if needed."""
    for key, value in iterator:
        if should_quantize_weight(key):
             qweight, scale, scale_name = quantize_param(key, value)
             yield key, qweight
             yield scale_name, scale
        else:
             yield key, value


from collections import defaultdict

from vllm import RequestOutput


class MaskCache:
    def __init__(self):
        """Initializes the mask cache for output tokens"""
        # Stores a list of output token_ids that should be masked out for each sequence (identified by seq_id)
        self._mask_cache: dict[int, list[int]] = defaultdict(list)

    def mask_token(self, seq_id: int, token_id: int) -> None:
        """
        Adds a token to the mask cache for a given sequence.

        Args:
            seq_id: The sequence ID.
            token_id: The token ID to mask out.
        """
        self._mask_cache[seq_id].append(token_id)

    def construct_masks(self, request_outputs: list[RequestOutput]) -> list[list[bool]]:
        """
        Constructs the mask for all sequences in the cache given the request
        outputs as a list of list of booleans.

        Args:
            request_outputs: The request outputs.

        Returns:
            A list of masks, where each mask is a list of booleans indicating whether each token should be masked out.
        """
        output_masks = []
        for request_output in request_outputs:
            request_id = int(request_output.request_id)
            for completion_output in request_output.outputs:
                # We dynamically compute the sequence ID from the request ID and the completion index
                # This will only be true if all completions are returned (e.g. sampling.best_of = sampling.n)
                # If not, the assertion below will catch the error though because after each batch, the mask cache should be automatically cleared
                seq_id = request_id * len(request_output.outputs) + completion_output.index
                # By default, all tokens are included in the mask
                output_mask = [True] * len(completion_output.token_ids)
                # Mask out the tokens from the cache
                for token_id in self._mask_cache.pop(seq_id, []):
                    output_mask[token_id] = False
                output_masks.append(output_mask)

        assert len(self._mask_cache) == 0, (
            "Mask cache is not empty (remaining keys: "
            + ", ".join(str(seq_id) for seq_id in self._mask_cache.keys())
            + "). It is likely that there is a mismatch between the request ID and the sequence IDs used to populate the mask cache."
        )
        return output_masks


_MASK_CACHE: MaskCache | None = None


def get_mask_cache() -> MaskCache:
    global _MASK_CACHE
    if _MASK_CACHE is None:
        _MASK_CACHE = MaskCache()
    return _MASK_CACHE

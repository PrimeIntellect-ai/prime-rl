from typing import Optional, Dict, List
import torch
from toploc import build_proofs_bytes
from concurrent.futures import ThreadPoolExecutor, Future, wait


class TopLocCache:
    def __init__(self, max_seqs: int, max_len: int, hidden_size: int, device: Optional[torch.device] = None):
        self.max_seqs = max_seqs
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.device = device
        self._cache: Optional[torch.Tensor] = None
        self.proofs: Dict[int, List[bytes]] = {}
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._proof_futures: Dict[int, Future] = {}

    def _init_cache(self, device: torch.device, dtype: torch.dtype):
        self._cache = torch.empty(self.max_seqs, self.max_len, self.hidden_size, device=device, dtype=dtype)
        self._seq_id_2_cache_index = {}
        self._current_seq_len = [0 for k in range(self.max_seqs)]
        # Tracks which chunk to alloc next
        self._current_cache_index = 0

    def reset_cache(self):
        self.proofs = {}
        self._seq_id_2_cache_index = {}
        self._current_seq_len = [0 for k in range(self.max_seqs)]
        # Tracks which chunk to alloc next
        self._current_cache_index = 0

    def add(self, seq_ids: list[int], values: torch.Tensor):
        assert len(seq_ids) == values.shape[0]
        if self._cache is None:
            self._init_cache(self.device or values.device, values.dtype)

        for i, seq_id in enumerate(seq_ids):
            if seq_id not in self._seq_id_2_cache_index:
                self._seq_id_2_cache_index[seq_id] = self._current_cache_index
                self._current_cache_index += 1
                self.proofs[seq_id] = []
                self._proof_futures[seq_id] = None
            cache_index = self._seq_id_2_cache_index[seq_id]
            if self._proof_futures[seq_id] is not None:
                wait([self._proof_futures[seq_id]])
                self._proof_futures[seq_id] = None
            self._cache[cache_index, self._current_seq_len[cache_index]].copy_(values[i], non_blocking=True)
            self._current_seq_len[cache_index] += 1
        self.maybe_generate_proofs_in_background()

    def maybe_generate_proofs_in_background(self, force_generate: bool = False):
        for seq_id, cache_index in self._seq_id_2_cache_index.items():
            if force_generate or self._current_seq_len[cache_index] == self.max_len:
                self._proof_futures[seq_id] = self._executor.submit(
                    self._generate_proof, seq_id, cache_index, self._current_seq_len[cache_index]
                )
                self._current_seq_len[cache_index] = 0

    def _generate_proof(self, seq_id: int, cache_index: int, seq_len: int) -> None:
        """Helper method for proof generation"""
        proof = build_proofs_bytes(self._cache[cache_index, :seq_len], decode_batching_size=self.max_len, topk=128, skip_prefill=True)[0]
        self.proofs[seq_id].append(proof)

    def wait_for_proofs(self):
        wait(list(self._proof_futures.values()))

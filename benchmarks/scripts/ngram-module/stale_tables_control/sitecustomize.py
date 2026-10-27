"""Positive control: vLLM workers drop every n-gram table chunk from NCCL weight updates, so their tables go stale."""

import importlib.abc
import importlib.util
import sys

TARGET = "prime_rl.inference.vllm.worker.nccl"
TABLE_CHUNK_MARKER = ".ngram_embedding.shard_"


class StaleTablesFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def patched_exec_module(module):
            exec_module(module)
            receive_state_dict = module.receive_state_dict

            def receive_state_dict_without_tables(communicator):
                for key, value in receive_state_dict(communicator):
                    if TABLE_CHUNK_MARKER not in key:
                        yield key, value

            module.receive_state_dict = receive_state_dict_without_tables
            print(f"stale-tables control: patched {TARGET}.receive_state_dict", file=sys.stderr, flush=True)

        spec.loader.exec_module = patched_exec_module
        return spec


sys.meta_path.insert(0, StaleTablesFinder())

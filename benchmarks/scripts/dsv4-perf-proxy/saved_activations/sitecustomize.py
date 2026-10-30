"""Record the tensors AOTAutograd saves for backward in every compiled graph: put this directory on PYTHONPATH.

Enabled only when SAVED_ACTIVATIONS_OUT is set and the process command line contains "prime_rl.trainer". Wraps
Inductor's min-cut partitioner via torch._inductor.config.custom_partitioner_fn and appends one JSON line per
saved value to $SAVED_ACTIVATIONS_OUT/rank<RANK>.jsonl: the compile id, node name, op, checkpoint policy tag,
innermost Dynamo module source name, shape, dtype and bytes (symbolic sizes at their trace-time hints).
Summarize with saved_activations.py. Shadows the interpreter's own sitecustomize (Ubuntu's only installs apport).
"""

import os
import sys

if os.environ.get("SAVED_ACTIVATIONS_OUT") and "prime_rl.trainer" in " ".join(getattr(sys, "orig_argv", sys.argv)):
    import json
    from pathlib import Path

    import torch
    import torch._inductor.config
    from torch._functorch.partitioners import _size_of, min_cut_rematerialization_partition
    from torch._guards import CompileContext
    from torch._inductor.custom_graph_pass import CustomPartitionerFn, get_hash_for_files

    OUT = Path(os.environ["SAVED_ACTIVATIONS_OUT"]) / f"rank{os.environ.get('RANK', '0')}.jsonl"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    class RecordingPartitioner(CustomPartitionerFn):
        def __call__(self, gm, joint_inputs, **kwargs):
            forward, backward = min_cut_rematerialization_partition(gm, joint_inputs, **kwargs)
            num_fwd_outputs = kwargs["num_fwd_outputs"]
            compile_id = str(CompileContext.current_compile_id())
            saved = forward.graph.find_nodes(op="output")[0].args[0][num_fwd_outputs:]
            with OUT.open("a") as f:
                for index, node in enumerate(saved):
                    val = node.meta.get("val")
                    stack = list(node.meta.get("nn_module_stack", {}).values())
                    record = {
                        "compile_id": compile_id,
                        "index": index,
                        "name": node.name,
                        "op": node.op,
                        "target": str(node.target),
                        "recompute": str(node.meta.get("recompute")),
                        "module": stack[-1][0] if stack else "",
                        "shape": [str(size) for size in val.shape] if isinstance(val, torch.Tensor) else None,
                        "dtype": str(val.dtype) if isinstance(val, torch.Tensor) else type(val).__name__,
                        "bytes": _size_of(node),
                    }
                    f.write(json.dumps(record) + "\n")
            return forward, backward

        def uuid(self):
            return get_hash_for_files((__file__,))

    torch._inductor.config.custom_partitioner_fn = RecordingPartitioner()

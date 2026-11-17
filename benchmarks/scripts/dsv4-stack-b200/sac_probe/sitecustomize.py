import importlib.abc
import importlib.util
import os
import sys

TARGET_MODULE = "prime_rl.trainer.activation_checkpointing"
LOG_PATH = os.environ.get("SAC_PROBE_LOG", "$INVESTIGATION/inputs/sac_probe/probe.log")
RANK = os.environ.get("RANK")


def _describe(value):
    import torch

    if isinstance(value, torch.Tensor):
        return f"{type(value).__name__}{tuple(value.shape)}:{value.dtype}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_describe(v) for v in value) + "]"
    return type(value).__name__


def _patch(module):
    original = module._selective_checkpoint_policy
    seen = {}

    def logged_policy(context, operation, *args, **kwargs):
        decision = original(context, operation, *args, **kwargs)
        import torch.fx.traceback as fx_traceback

        if fx_traceback.has_preserved_node_meta():
            paths = tuple(module._traced_block_module_paths())
        else:
            paths = ("<no preserved node meta>",)
        innermost = paths[-1] if paths else "<block>"
        key = (operation.name(), innermost, getattr(context, "is_recompute", None))
        if key not in seen:
            seen[key] = 1
            arg_desc = ",".join(_describe(a) for a in args)
            out_kwargs = {k: _describe(v) for k, v in kwargs.items() if k not in ("targets", "module_targets", "matched_module_targets")}
            with open(LOG_PATH, "a") as f:
                f.write(
                    f"SAC_PROBE op={operation.name()} path={innermost} is_recompute={key[2]} "
                    f"decision={decision.name} paths={list(paths)} args=({arg_desc}) kwargs={out_kwargs}\n"
                )
        return decision

    module._selective_checkpoint_policy = logged_policy
    print(f"SAC_PROBE patched {module.__name__}._selective_checkpoint_policy rank={RANK} pid={os.getpid()}", file=sys.stderr, flush=True)


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _PatchingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET_MODULE:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader)
        return spec


if RANK == "0":
    sys.meta_path.insert(0, _PatchingFinder())
    print(f"SAC_PROBE sitecustomize loaded rank={RANK} pid={os.getpid()} log={LOG_PATH}", file=sys.stderr, flush=True)

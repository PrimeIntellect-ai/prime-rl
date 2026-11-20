import runpy
import sys

import triton.backends.nvidia.compiler as nvc

orig = nvc.llvm.optimize_module


def optimize_module_noslp(*a, **k):
    k["disable_slp_vectorizer"] = True
    return orig(*a, **k)


nvc.llvm.optimize_module = optimize_module_noslp
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

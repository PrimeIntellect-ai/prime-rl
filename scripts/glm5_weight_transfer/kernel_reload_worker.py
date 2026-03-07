"""Worker extension for kernel-format weight reloading."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
    Worker = Worker
else:
    Worker = object


class KernelReloadWorker(Worker):
    def reload_kernel_weights_from_path(self, path: str) -> None:
        """Load kernel-format weights from safetensors and copy_ in-place (CUDA-graph safe)."""
        from safetensors import safe_open
        import torch

        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model

        params = dict(model.named_parameters())
        loaded = set()

        with safe_open(path, framework="pt", device="cuda") as f:
            for name in f.keys():
                if name in params:
                    with torch.no_grad():
                        params[name].copy_(f.get_tensor(name))
                    loaded.add(name)
                else:
                    print(f"  SKIP {name} (not in model params)")

        print(f"Reloaded {len(loaded)} kernel-format weights in-place")

        # Re-run process_weights_after_loading (needed for MLA KV absorption etc.)
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)
        print("process_weights_after_loading completed")

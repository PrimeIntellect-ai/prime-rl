"""Offline probe RPCs: importable methods avoid serializing Python callbacks."""

from collections import Counter
from pathlib import Path

import torch


class FP8ProbeWorker:
    def fp8_receive_kernel(self, port):
        from vllm.distributed import get_tensor_model_parallel_world_size

        from prime_rl.inference.vllm.worker.nccl import NCCLWeightBroadcastReceiver
        from prime_rl.inference.vllm.worker.weight_transfer import load_weights_kernel

        assert get_tensor_model_parallel_world_size() == 1
        self._fp8_probe_receiver = NCCLWeightBroadcastReceiver(
            host="127.0.0.1", port=port, rank=1, world_size=2, device=self.device, timeout=600
        )
        model = getattr(self.model_runner.model, "runnable", self.model_runner.model)
        load_weights_kernel(model, self._fp8_probe_receiver.receive_state_dict())
        torch.cuda.synchronize()
        return self.fp8_audit()

    def fp8_audit(self):
        model = getattr(self.model_runner.model, "runnable", self.model_runner.model)
        counts = Counter()
        padded = []
        for name, layer in model.named_modules():
            method = getattr(layer, "quant_method", None)
            if method is not None:
                counts[type(method).__name__] += 1
            padding = getattr(layer, "_prime_rl_fp8_input_padding", 0)
            if padding:
                assert layer.weight.dtype == torch.float8_e4m3fn
                assert layer.weight.shape[-1] % 128 == 0
                padded.append(
                    {
                        "name": name,
                        "padding": padding,
                        "dtype": str(layer.weight.dtype),
                        "shape": list(layer.weight.shape),
                    }
                )
        return {"methods": dict(counts), "padded": padded}

    def fp8_reload_checkpoint(self, path):
        from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator

        from prime_rl.inference.vllm.worker.weight_transfer import load_weights_checkpoint_layerwise

        files = sorted(str(file) for file in Path(path).glob("*.safetensors"))
        assert files, path
        weights = safetensors_weights_iterator(files, use_tqdm_on_load=False)
        model = getattr(self.model_runner.model, "runnable", self.model_runner.model)
        load_weights_checkpoint_layerwise(model, weights, self.model_runner.model_config, self.vllm_config)
        return self.fp8_audit()

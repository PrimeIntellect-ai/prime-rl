import os
from typing import TYPE_CHECKING, cast

import torch
from torch.nn import Module

from prime_rl.transports.weights.mx_phases import timed_refit
from prime_rl.utils.mx_compat import require_mx_refit

mx = require_mx_refit()


def _step_of(version_uid: str) -> int:
    """Parse the step suffix from a ModelExpress version ID."""
    _, _, suffix = version_uid.rpartition(":")
    if not suffix.isdigit():
        raise ValueError(f"Malformed ModelExpress version id: {version_uid!r}")
    return int(suffix)


if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class MXRefitUpdateWorker(Worker):
    """vLLM worker extension for ModelExpress weight updates."""

    def init_broadcaster(self, mx_server_host: str, mx_server_port: int, *args) -> None:
        del args  # unused extras from the shared init_broadcaster route
        require_mx_refit(streaming=os.environ.get("MX_REFIT_STAGING_BYTES") is not None)
        model = cast(Module, self.model_runner.get_model())
        # TODO: Move PrimeRL-to-HF conversion to the trainer.
        from prime_rl.trainer.models import get_custom_causal_lm_cls
        from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf

        hf_config = self.model_runner.model_config.hf_config
        conversion_chain = get_custom_causal_lm_cls(hf_config).conversion_chain(hf_config)
        convert_native_to_hf = (lambda sd: apply_prime_to_hf(sd, conversion_chain)) if conversion_chain else None
        self._generator = mx.ModelExpressGeneratorClient.initialize(
            mx.ModelExpressGeneratorConfig(
                engine_context=mx.VllmGeneratorContext(
                    model=model,
                    vllm_config=self.vllm_config,
                    convert_native_to_hf=convert_native_to_hf,
                ),
                model_name=self.model_runner.model_config.model,
                server_url=f"{mx_server_host}:{mx_server_port}",
                source_order=(mx.WeightSource.TRAINER,),
            )
        )

    def liveness_probe(self) -> None:
        return None

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None, version_uid: str | None = None) -> None:
        del weight_dir  # mx_refit pulls by version, not a path
        if version_uid is None:
            raise ValueError("mx_refit update_weights requires version_uid")
        with timed_refit("generator", _step_of(version_uid), version_uid) as timer:
            timer.mark("rank", self.rank)
            timer.mark("replica", int(os.environ.get("MX_REFIT_REPLICA_ID", "0")))
            budget = os.environ.get("MX_REFIT_STAGING_BYTES")
            if budget is not None:
                max_staging_bytes = int(budget)
                if max_staging_bytes <= 0:
                    raise ValueError("MX_REFIT_STAGING_BYTES must be positive")
                metrics = self._generator.apply_weight_streaming(
                    version=mx.WeightVersionRef(version_uid), max_staging_bytes=max_staging_bytes
                )
                for name, value in metrics.items():
                    timer.mark(name, float(value))
            else:
                with timer.phase("wire"):
                    staged = self._generator.stage_weight(version=mx.WeightVersionRef(version_uid))
                try:
                    with timer.phase("install"):
                        self._generator.apply_weight(staged)
                finally:
                    with timer.phase("release"):
                        staged.release()
        return None

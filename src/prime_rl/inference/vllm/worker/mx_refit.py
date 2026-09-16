import os
from typing import TYPE_CHECKING, cast

import torch
from modelexpress_rl import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    VllmGeneratorContext,
    WeightSource,
    WeightVersionRef,
)
from torch.nn import Module

from prime_rl.transports.weights.mx_phases import timed_refit
from prime_rl.transports.weights.mx_rdma import apply_rdma_defaults
from prime_rl.utils.mx_verification import perturb_weights, snapshot_weights, verify_weights


def _step_of(version_uid: str) -> int:
    """Parse the step suffix from a ModelExpress version ID."""
    _, _, suffix = version_uid.rpartition(":")
    if not suffix.isdigit():
        # We construct these ids ourselves, so an unparseable one is a bug
        # rather than input to tolerate. Returning -1 let it reach the timing
        # record as though it were a real step.
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
        model = cast(Module, self.model_runner.get_model())
        # TODO: Move PrimeRL-to-HF conversion to the trainer.
        from prime_rl.trainer.models import get_custom_causal_lm_cls
        from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf

        hf_config = self.model_runner.model_config.hf_config
        conversion_chain = get_custom_causal_lm_cls(hf_config).conversion_chain(hf_config)
        convert_native_to_hf = (lambda sd: apply_prime_to_hf(sd, conversion_chain)) if conversion_chain else None
        # The NIXL agent reads the rail configuration during initialization.
        apply_rdma_defaults()
        self._generator = ModelExpressGeneratorClient.initialize(
            ModelExpressGeneratorConfig(
                engine_context=VllmGeneratorContext(
                    model=model,
                    vllm_config=self.vllm_config,
                    convert_native_to_hf=convert_native_to_hf,
                ),
                model_name=self.model_runner.model_config.model,
                server_url=f"{mx_server_host}:{mx_server_port}",
                source_order=(WeightSource.TRAINER,),
            )
        )

    def liveness_probe(self) -> None:
        return None

    def prepare_initial_verification(self) -> dict:
        if os.environ.get("MX_VERIFY_INITIAL_REFIT") != "1":
            raise RuntimeError("Initial refit verification is not enabled")
        if getattr(self, "_initial_verification_started", False):
            raise RuntimeError("Initial refit verification cannot be repeated in the same engine")
        self._initial_verification_started = True
        model = self.model_runner.get_model()
        budget = int(os.environ.get("MX_VERIFY_CPU_BYTES", str(64 * 1024**3)))
        self._initial_snapshot = snapshot_weights(model, max_bytes=budget)
        perturb_weights(model, self._initial_snapshot)
        return {"rank": self.rank, "changed_tensors": self._initial_snapshot.changed_tensors}

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None, version_uid: str | None = None) -> dict | None:
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
                    version=WeightVersionRef(version_uid), max_staging_bytes=max_staging_bytes
                )
                for name, value in metrics.items():
                    timer.mark(name, float(value))
            else:
                with timer.phase("wire"):
                    staged = self._generator.stage_weight(version=WeightVersionRef(version_uid))
                try:
                    with timer.phase("install"):
                        self._generator.apply_weight(staged)
                finally:
                    with timer.phase("release"):
                        staged.release()
            snapshot = getattr(self, "_initial_snapshot", None)
            if snapshot is not None:
                if _step_of(version_uid) != 0:
                    raise RuntimeError("Initial restoration requires startup version :0")
                with timer.phase("initial_verification"):
                    record = verify_weights(self.model_runner.get_model(), snapshot)
                record.update(
                    version_uid=version_uid, rank=self.rank, replica=int(os.environ.get("MX_REFIT_REPLICA_ID", "0"))
                )
                # Drop the snapshot either way. The server turns a failed record
                # into a 500 naming the real cause; keeping it on failure only
                # means the next refit fails instead on the unrelated startup
                # version check, which reports step numbering for what was
                # actually a verification mismatch.
                self._initial_snapshot = None
                return record
        return None

import uuid
import warnings
from pathlib import Path
from typing import Annotated, Literal, TypeAlias
from urllib.parse import urlparse

from pydantic import Field, model_validator

from prime_rl.configs.eval import SFTOnlineEvalConfig
from prime_rl.configs.inference import InferenceConfig, setup_dummy_load
from prime_rl.configs.inference import WeightBroadcastConfig as InferenceWeightBroadcastConfig
from prime_rl.configs.monitors import MonitorsConfig
from prime_rl.configs.shared import (
    BaseMultiNodeDeploymentConfig,
    BaseSingleNodeDeploymentConfig,
    FileSystemWeightBroadcastConfig,
    NCCLWeightBroadcastConfig,
    ResumeConfig,
    RunConfig,
    SlurmConfig,
    WeightBroadcastConfig,
)
from prime_rl.configs.trainer import BaseTrainerConfig, mega_moe_max_tokens_per_rank
from prime_rl.utils.config import BaseConfig, default_output_dir, find_package_resource


class BaseDataConfig(BaseConfig):
    batch_size: int = Field(128, ge=1)
    """Global batch size."""

    seq_len: int = Field(128, ge=1)
    """Sequence length."""

    micro_batch_size: int = Field(1, ge=1)
    """Per-step micro batch size. ``batch_size`` must be divisible by this."""

    num_workers: int = Field(1, ge=1)
    """Number of dataloader worker processes. Batches are prepared and pinned in the background so tokenization/packing overlaps with training."""

    pad_to_seq_len: bool = False
    """Pad every packed row to ``seq_len * micro_batch_size`` instead of to a multiple of ``model.cp``. Rows then have
    one static shape, which avoids recompilation on dynamic sequence lengths (torch.compile, DeepGEMM JIT kernels, ...)
    at the cost of extra padding."""

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")
        return self


class BaseSourceConfig(BaseConfig):
    """One named source of the training data (``[[data.source]]``), in pretraining and SFT."""

    name: str = Field(pattern=r"^[^/]+$")
    """Names the source in the ``progress/<name>/*`` metrics."""

    weight: float | None = Field(None, ge=0)
    """The source's relative weight: its share of the training data (tokens in pretraining, samples in SFT) is its
    weight over the sum of all weights. If None on every source, the share is the source's share of all sources' data
    (tokens in pretraining, rows in SFT)."""


def check_sources(sources: list[BaseSourceConfig]) -> None:
    """Raise if two sources share a name, or the weights are set on only some sources or sum to 0."""
    names = [source.name for source in sources]
    if len(set(names)) < len(names):
        raise ValueError(f"data.source: the names must be unique, got {names}")
    weights = [source.weight for source in sources]
    if None in weights and any(weight is not None for weight in weights):
        raise ValueError("data.source: set weight on every source or on none (natural shares)")
    if None not in weights and sum(weights) <= 0:
        raise ValueError("data.source: the weights must sum to more than 0")


class BaseValConfig(BaseConfig):
    interval: int = Field(50, ge=1)
    """Run validation every N training steps."""

    eval_on_start: bool = False
    """Run validation before the first training step."""

    data: BaseDataConfig


class DocumentsValSetConfig(BaseConfig):
    """Pre-tokenized documents. Every document is scored on its own, from its first token, in non-overlapping windows
    of ``seq_len`` scored tokens: a longer document continues in the next window, which does not see the tokens
    before it."""

    type: Literal["documents"] = "documents"

    name: str = Field(pattern=r"^[^/]+$")
    """Names the set's metrics, ``val/<name>/<metric>@<seq_len>``, e.g. ``val/fineweb-edu/bpb@2k``."""

    path: Path
    """Folder of datatrove tokenized shards, as for a pretraining source's ``path``. The tokens are scored as stored."""

    seq_len: int | None = Field(None, ge=1)
    """Context length: the most input tokens a scored sequence has. Defaults to the training ``data.seq_len``."""


class ValConfig(BaseConfig):
    interval: int = Field(50, ge=1)
    """Run validation every N training steps."""

    eval_on_start: bool = False
    """Run validation before the first training step."""

    set: list[DocumentsValSetConfig] = Field(min_length=1)
    """The validation sets (``[[val.set]]``). Each logs ``val/<name>/loss@<seq_len>``, the mean loss per scored
    token, and with a byte-level BPE tokenizer ``val/<name>/bpb@<seq_len>``, bits per UTF-8 byte of the scored
    tokens."""

    num_samples: int | None = Field(None, ge=1)
    """Items per set and validation: the first ``num_samples`` documents of each set, the same ones every time. If
    None, every item."""

    micro_batch_tokens: int | None = Field(None, ge=1)
    """Tokens per packed row of the sets (one forward pass). Defaults to the training micro-batch
    (``data.seq_len * data.micro_batch_size``) or the longest set ``seq_len``, whichever is larger, rounded up to a
    multiple of ``model.cp``; must be at least that longest ``seq_len``."""

    save_arrays: bool = False
    """Write the per-item arrays (npz) and summary (json) of every set to ``<run_dir>/val/step_N/``."""

    @model_validator(mode="after")
    def validate_unique_names(self):
        names = [val_set.name for val_set in self.set]
        if len(set(names)) < len(names):
            raise ValueError(f"val.set names must be unique: {names}")
        return self


class SingleNodeDeploymentConfig(BaseSingleNodeDeploymentConfig):
    @model_validator(mode="after")
    def validate_gpu_count(self):
        if self.num_train_gpus > self.gpus_per_node:
            raise ValueError(f"num_train_gpus ({self.num_train_gpus}) exceeds gpus_per_node ({self.gpus_per_node}).")
        return self


class MultiNodeDeploymentConfig(BaseMultiNodeDeploymentConfig):
    num_infer_nodes: int = Field(0, ge=0)
    """Inference nodes for online evals. These nodes share one SLURM allocation with the
    trainer nodes."""


SupervisedDeploymentConfig: TypeAlias = Annotated[
    SingleNodeDeploymentConfig | MultiNodeDeploymentConfig, Field(discriminator="type")
]


class SupervisedTrainerConfig(BaseTrainerConfig):
    """SFT and pretraining: next-token loss on a fixed dataset, as opposed to RL."""

    auto_freeze_moe_router = False
    # Standard L2 regularization for supervised training.
    auto_weight_decay = 0.01
    data: BaseDataConfig

    val: BaseValConfig | ValConfig | None = None
    """Validation configuration. If None, no validation runs."""

    eval: SFTOnlineEvalConfig | None = None
    """Online evaluation configuration: rollout-based evals against a live inference
    server that receives the trainer's weight broadcasts. If None, no online evals run."""

    inference: InferenceConfig | None = None
    """Inference server for online evals. If None (with ``eval`` set), the launcher
    does not start a server and evals connect to ``eval.client.base_url``."""

    weight_broadcast: WeightBroadcastConfig | None = None
    """Trainer-to-inference weight transport for online evals. Defaults to NCCL.
    LoRA and external inference use filesystem broadcast."""

    resume: ResumeConfig | None = None
    """Resume the run from a checkpoint (point at it with the previous run's ``run.name``). Without ``[ckpt]`` the run loads the checkpoint but saves no new ones. The checkpoint supplies the training state; the hyperparameters and the learning rate schedule come from this config. If None, does not resume."""

    monitors: MonitorsConfig = MonitorsConfig()
    """Metric monitors (``monitors.wandb``, ``monitors.file``, ``monitors.prime``)."""

    run: RunConfig = Field(default_factory=RunConfig)
    """Run metadata. ``run.name`` names the run directory under ``output_dir``."""

    output_dir: Path = Field(default_factory=default_output_dir)
    """Directory that groups related runs. Each run writes its artifacts (checkpoints, logs, ...) to ``output_dir / run.name``. Should be a persistent directory with enough disk space. Defaults to ``$PRL_OUTPUT_DIR`` if set, else ``outputs``."""

    clean: bool = False
    """Delete the run directory (``output_dir / run.name``) before starting training. Required to overwrite a run directory that contains artifacts from a previous run when not resuming."""

    @property
    def run_dir(self) -> Path:
        assert self.run.dir is not None  # resolved at construction
        return self.output_dir / self.run.dir

    @model_validator(mode="after")
    def auto_setup_run_identity(self):
        """Auto-generate the run name (``<source>--<model>--<short-id>``, with the source only if it is the only
        one) when unset and default the run directory, W&B run name and platform run name to it when not set
        explicitly."""
        if self.run.name is None:
            sources = getattr(self.data, "source", [])
            dataset = "fake" if getattr(self.data, "fake", None) else sources[0].name if len(sources) == 1 else ""
            model = self.model.name.split("/")[-1]
            parts = [part for part in (dataset, model) if part]
            self.run.name = "--".join([*parts, uuid.uuid4().hex[:8]]).lower()
        if self.run.dir is None:
            self.run.dir = self.run.name
        if self.monitors.wandb is not None and self.monitors.wandb.name is None:
            self.monitors.wandb.name = self.run.name
        if self.monitors.prime is not None and self.monitors.prime.name is None:
            self.monitors.prime.name = self.run.name
        return self

    deployment: SupervisedDeploymentConfig = SingleNodeDeploymentConfig()

    slurm: SlurmConfig | None = None
    """SLURM configuration. When set, the run is submitted as a SLURM job instead of running locally."""

    dashboard: bool = True
    """Make sure a local dashboard daemon serves this run's output dir (started on
    demand in interactive sessions; an already-running daemon's URL is logged)."""

    dry_run: bool = False
    """Only validate and dump resolved configs, then exit early."""

    ### Pre-validation normalization

    @model_validator(mode="before")
    @classmethod
    def propagate_model_to_inference(cls, data):
        """Fill ``inference.vllm.model`` and ``tokenizer`` from ``model.name`` and ``tokenizer.name`` before
        InferenceConfig construction, so its parser auto-resolution sees the right names."""
        if not isinstance(data, dict):
            return data
        inference = data.get("inference")
        if not isinstance(inference, dict):
            return data
        vllm = inference.setdefault("vllm", {})
        if not isinstance(vllm, dict):
            return data
        for section, key in (("model", "model"), ("tokenizer", "tokenizer")):
            config = data.get(section)
            name = config.get("name") if isinstance(config, dict) else None
            if name:
                vllm.setdefault(key, name)
        return data

    ### Validate configs (e.g. raise for unsupported (combinations of) configs)

    @model_validator(mode="after")
    def validate_deployment(self):
        if self.deployment.type == "multi_node" and self.slurm is None:
            raise ValueError("Must use SLURM for multi-node deployment.")
        return self

    @model_validator(mode="after")
    def auto_setup_online_eval(self):
        """Wire online evals and the trainer-to-inference weight transport."""
        if self.eval is None:
            if self.inference is not None:
                raise ValueError("[inference] is only used for online evals — add an [eval] block or remove it.")
            return self

        # LoRA runs broadcast the raw adapter, which evals reload via /load_lora_adapter
        if self.model.lora is not None:
            if self.inference is not None:
                self.inference.vllm.enable_lora = True
                self.inference.vllm.max_lora_rank = self.model.lora.rank
            else:
                warnings.warn(
                    "LoRA is enabled, but inference is not configured. When manually starting the inference server, "
                    "make sure to set --enable_lora and --max-lora-rank.",
                    stacklevel=2,
                )

        # A scratch init reaches the server with the trainer's startup broadcast.
        if self.inference is not None:
            setup_dummy_load(self.inference, self.model.init.type == "scratch")

        if self.weight_broadcast is None:
            if self.model.lora is not None or self.inference is None:
                self.weight_broadcast = FileSystemWeightBroadcastConfig()
            else:
                self.weight_broadcast = NCCLWeightBroadcastConfig()
        if self.weight_broadcast.interval != 1:
            raise ValueError("weight_broadcast.interval is RL only: online evals broadcast at every eval step.")
        if self.weight_broadcast.type != "filesystem":
            if self.weight_broadcast.type in ("nixl", "modelexpress"):
                raise ValueError("NIXL and ModelExpress weight broadcast are not supported for SFT online evals.")
            if self.model.lora is not None:
                raise ValueError(
                    "LoRA training is not yet supported with in-memory weight broadcast. "
                    "Set weight_broadcast.type = 'filesystem'."
                )

        if self.deployment.type == "multi_node":
            # Dedicated nodes in the SFT allocation run the inference pool, router,
            # env worker pools, and evals process.
            if self.inference is None:
                raise ValueError(
                    "Multi-node online evals require an [inference] block - dedicated nodes in the "
                    "SFT allocation run the inference pool and evals process."
                )
            if self.deployment.num_infer_nodes < 1:
                raise ValueError("Online evals on a multi-node deployment require deployment.num_infer_nodes >= 1.")
            if self.inference.router is None:
                raise ValueError(
                    "Multi-node online evals require an inference router - the launcher starts one "
                    "router in front of the per-rank engines. Remove inference.router = 'None'."
                )
            if self.inference.vllm.model != self.model.name:
                raise ValueError(
                    f"inference.vllm.model ({self.inference.vllm.model}) does not match model.name "
                    f"({self.model.name}). Remove inference.vllm.model to inherit it."
                )
            if self.deployment.gpus_per_node % self.inference.vllm.tensor_parallel_size != 0:
                raise ValueError(
                    f"deployment.gpus_per_node ({self.deployment.gpus_per_node}) must be divisible by "
                    f"inference.vllm.tensor_parallel_size ({self.inference.vllm.tensor_parallel_size})."
                )
            if self.weight_broadcast.type == "nccl":
                self.weight_broadcast.inference_world_size = (
                    self.deployment.num_infer_nodes * self.deployment.gpus_per_node
                )
            self.inference.weight_broadcast = InferenceWeightBroadcastConfig(type=self.weight_broadcast.type)
            if self.max_steps is None:
                warnings.warn(
                    "Online evals without max_steps: the evals process never sees a final checkpoint, "
                    "so the SFT SLURM job holds its allocation until its walltime.",
                    stacklevel=2,
                )
            # The client is wired at runtime by the SFT sbatch script (the router and
            # per-rank admin URLs are only known once SLURM assigns hosts).
            return self

        if self.inference is None:
            warnings.warn(
                "Online evals are configured without an [inference] block - the launcher will not "
                f"start an inference server. Make sure one is running at eval.client.base_url "
                f"({self.eval.client.base_url}) with weight_broadcast.type = 'filesystem', "
                "otherwise the online-eval process will hang waiting for it. If a router fronts the "
                "deployment, set eval.client.admin_base_url to the engine URLs - admin ops "
                "(pause/update_weights/resume) must bypass the router.",
                stacklevel=2,
            )
            return self

        total_gpus = self.deployment.num_train_gpus + self.deployment.num_infer_gpus
        if total_gpus > self.deployment.gpus_per_node:
            raise ValueError(
                f"Total GPU count ({total_gpus} = {self.deployment.num_train_gpus} train + "
                f"{self.deployment.num_infer_gpus} infer) exceeds gpus_per_node "
                f"({self.deployment.gpus_per_node})."
            )

        if self.inference.vllm.model != self.model.name:
            raise ValueError(
                f"inference.vllm.model ({self.inference.vllm.model}) does not match model.name "
                f"({self.model.name}). Remove inference.vllm.model to inherit it."
            )

        # Fill inference capacity with DP ranks (mirrors RLConfig.auto_setup_deployment).
        num_infer_gpus = self.deployment.num_infer_gpus
        vllm = self.inference.vllm
        if num_infer_gpus != vllm.data_parallel_size * vllm.tensor_parallel_size:
            if num_infer_gpus % vllm.tensor_parallel_size != 0:
                raise ValueError(
                    f"deployment.num_infer_gpus ({num_infer_gpus}) must be divisible by "
                    f"inference.vllm.tensor_parallel_size ({vllm.tensor_parallel_size})."
                )
            vllm.data_parallel_size = num_infer_gpus // vllm.tensor_parallel_size
        if vllm.api_server_count < vllm.data_parallel_size and not vllm.enable_lora:
            vllm.api_server_count = vllm.data_parallel_size
        if self.weight_broadcast.type == "nccl":
            self.weight_broadcast.inference_world_size = vllm.data_parallel_size * vllm.tensor_parallel_size
        self.inference.weight_broadcast = InferenceWeightBroadcastConfig(type=self.weight_broadcast.type)

        host = self.inference.server.host or "localhost"
        client = self.eval.client
        if "base_url" not in client.model_fields_set:
            client.base_url = f"http://{host}:{self.inference.server.port}/v1"
        elif urlparse(client.base_url).port != self.inference.server.port:
            raise ValueError(
                f"eval.client.base_url port ({urlparse(client.base_url).port}) does not match "
                f"inference.server.port ({self.inference.server.port})."
            )
        if self.inference.router is not None and "admin_base_url" not in client.model_fields_set:
            # Admin ops (pause/update_weights/resume) must bypass the router and hit
            # the engine directly.
            client.admin_base_url = [f"http://{host}:{self.inference.backend_port}/v1"]
        return self

    @model_validator(mode="after")
    def auto_setup_val(self):
        if not isinstance(self.val, ValConfig):
            return self
        for val_set in self.val.set:
            if val_set.seq_len is None:
                val_set.seq_len = self.data.seq_len
        longest = max(val_set.seq_len for val_set in self.val.set)
        if self.val.micro_batch_tokens is None:
            row_len = max(self.data.seq_len * self.data.micro_batch_size, longest)
            self.val.micro_batch_tokens = -(-row_len // self.model.cp) * self.model.cp
        elif self.val.micro_batch_tokens < longest:
            raise ValueError(f"val.micro_batch_tokens must be at least the longest set seq_len ({longest})")
        return self

    @model_validator(mode="after")
    def validate_cp_seq_len(self):
        if self.model.cp > 1:
            if self.data.seq_len % self.model.cp != 0:
                raise ValueError("Sequence length must be divisible by CP degree")
            if isinstance(self.val, ValConfig) and self.val.micro_batch_tokens % self.model.cp != 0:
                raise ValueError("val.micro_batch_tokens must be divisible by CP degree")
            if isinstance(self.val, BaseValConfig) and self.val.data.seq_len % self.model.cp != 0:
                raise ValueError("Validation sequence length must be divisible by CP degree")
        return self

    @model_validator(mode="after")
    def resolve_mega_moe_max_tokens_per_rank(self):
        dispatch = self.model.moe.dispatch
        if dispatch.type == "mega_moe":
            row_lens = [self.data.micro_batch_size * self.data.seq_len]
            if isinstance(self.val, ValConfig):
                row_lens.append(self.val.micro_batch_tokens)
            elif self.val is not None:
                row_lens.append(self.val.data.micro_batch_size * self.val.data.seq_len)
            dispatch.max_tokens_per_rank = mega_moe_max_tokens_per_rank(
                dispatch.max_tokens_per_rank,
                max(row_lens),
                self.model.cp,
            )
        return self

    @model_validator(mode="after")
    def validate_cp_micro_batch_size(self):
        if self.model.cp > 1:
            if self.data.micro_batch_size != 1:
                raise ValueError("Micro batch size must be 1 when CP is enabled")
            if isinstance(self.val, BaseValConfig) and self.val.data.micro_batch_size != 1:
                raise ValueError("Validation micro batch size must be 1 when CP is enabled")
        return self

    ### Auto-setup and validate shared configs

    @model_validator(mode="after")
    def auto_setup_slurm_template(self):
        if self.slurm is not None and self.slurm.job_name is None:
            self.slurm.job_name = self.run.name
        if self.slurm is not None and self.slurm.template_path is None:
            templates_dir = find_package_resource("templates")
            if templates_dir is not None:
                if self.deployment.type == "single_node":
                    self.slurm.template_path = templates_dir / "single_node_sft.sbatch.j2"
                else:
                    self.slurm.template_path = templates_dir / "multi_node_sft.sbatch.j2"
        return self

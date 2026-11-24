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
from prime_rl.configs.val import HFDatasetConfig, LossMaskConfig, SFTColumnsConfig, ValConfig
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
    """One named source of the training data (``[[data.source]]``)."""

    name: str = Field(pattern=r"^[^/]+$")
    """Names the source in the ``progress/<name>/*`` metrics."""

    weight: float | None = Field(None, ge=0)
    """The source's relative weight: its share of the training data (tokens of ``documents``, samples of ``chat``
    data) is its weight over the sum of all weights. If None on every source, the share is the source's share of all
    sources' data (tokens of ``documents``, rows of ``chat`` data)."""


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


class TokenSourceConfig(BaseSourceConfig):
    """A folder of datatrove tokenized shards, one source of ``documents`` data."""

    path: Path
    """Folder of ``*.ds`` token files, each with its ``.ds.index`` and ``.ds.metadata``."""

    repetitions: float | None = Field(None, ge=1)
    """How often every token of a fixed subset of the source is seen over the run (needs ``max_steps``). If None, the
    whole source is read."""


class DocumentsDataConfig(BaseDataConfig):
    """Tokenized documents, sampled from one or more sources into rows of ``seq_len`` tokens."""

    type: Literal["documents"] = "documents"

    source: list[TokenSourceConfig] = Field(min_length=1)
    """The folders to sample documents from (``[[data.source]]``)."""

    seed: int = Field(0, ge=0)
    """Seed for the order of the documents, within rows too, and for the subsets."""

    document_masking: bool = True
    """Attention and positions restart at every document and piece. If False, a row is one sequence."""

    eod_mask_loss: bool = True
    """Don't train the model to predict the first token of the next, unrelated document: no loss where the input is
    the document separator. The separator is the special token the training shards' documents end with, detected at
    startup; startup fails if they don't all end with one, unless they start with BOS (then there is nothing to
    mask)."""

    @model_validator(mode="after")
    def validate_sources(self):
        check_sources(self.source)
        return self


class HFDatasetSourceConfig(HFDatasetConfig, BaseSourceConfig):
    """One ``(dataset, subset, split)`` of a Hugging Face dataset, one source of ``chat`` data."""


class FakeDataConfig(BaseConfig):
    """Synthetic token sequences of up to ``data.seq_len`` tokens, for throughput and plumbing tests."""

    length: Literal["fixed", "variable"] = "fixed"
    """Use fixed-length samples or variable-length samples."""

    input_ids: Literal["increasing", "random"] = "increasing"
    """Token id generator: ``increasing`` for deterministic sequences, ``random`` for random ids."""


class ChatDataConfig(BaseDataConfig):
    """Conversations rendered with ``renderer``: samples drawn from one or more sources into packed rows of
    ``seq_len`` tokens."""

    type: Literal["chat"] = "chat"

    source: list[HFDatasetSourceConfig] = Field(
        default_factory=lambda: [HFDatasetSourceConfig(name="reverse-text", dataset="PrimeIntellect/Reverse-Text-SFT")],
        min_length=1,
    )
    """The sources to draw samples from (``[[data.source]]``)."""

    fake: FakeDataConfig | None = None
    """Train on synthetic token sequences instead of the sources. Needs no renderer."""

    shuffle: bool = True
    """Shuffle every source at the start of each of its passes."""

    seed: int = 0
    """Seed for the order of every source, re-shuffled per pass by adding the pass count, and for fake tokens."""

    columns: SFTColumnsConfig = SFTColumnsConfig()
    """Columns of every source. A source's ``columns`` override single fields."""

    loss_mask: LossMaskConfig = LossMaskConfig()
    """Which message types contribute to the loss."""

    @model_validator(mode="after")
    def validate_sources(self):
        check_sources(self.source)
        for source in self.source:
            source.resolve_columns(self.columns)
        return self


DataConfig: TypeAlias = Annotated[DocumentsDataConfig | ChatDataConfig, Field(discriminator="type")]


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
    """SFT and pretraining: next-token loss on a fixed dataset, as opposed to RL. ``uv run sft`` (``SFTConfig``) and
    ``uv run pretrain`` (``PretrainConfig``) differ only in their defaults."""

    auto_freeze_moe_router = False
    # Standard L2 regularization for supervised training.
    auto_weight_decay = 0.01

    data: DataConfig
    """Training data: tokenized ``documents`` or ``chat`` conversations rendered with ``renderer``."""

    val: ValConfig | None = None
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
    """Resume the run from a checkpoint (point at it with the previous run's ``run.name``). Without ``[ckpt]`` the run loads the checkpoint but saves no new ones. If None, does not resume."""

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
            sources = self.data.source
            fake = self.data.type == "chat" and self.data.fake is not None
            dataset = "fake" if fake else sources[0].name if len(sources) == 1 else ""
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
        if self.val is not None:
            chat = self.data.type == "chat"
            columns = self.data.columns if chat else SFTColumnsConfig()
            loss_mask = self.data.loss_mask if chat else LossMaskConfig()
            row_len = self.data.seq_len * self.data.micro_batch_size
            self.val.resolve(self.data.seq_len, row_len, self.model.cp, columns, loss_mask)
        return self

    @model_validator(mode="after")
    def validate_cp_seq_len(self):
        if self.model.cp > 1 and self.data.seq_len % self.model.cp != 0:
            raise ValueError("Sequence length must be divisible by CP degree")
        return self

    @model_validator(mode="after")
    def resolve_mega_moe_max_tokens_per_rank(self):
        dispatch = self.model.moe.dispatch
        if dispatch.type == "mega_moe":
            row_lens = [self.data.micro_batch_size * self.data.seq_len]
            if self.val is not None:
                row_lens.append(self.val.micro_batch_tokens)
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
        return self

    @model_validator(mode="after")
    def validate_vlm_val(self):
        if self.model.vlm is not None and self.val is not None:
            raise ValueError("Validation scores text only: [val] does not support VLM models ([model.vlm]).")
        return self

    @model_validator(mode="after")
    def validate_repetitions(self):
        if self.data.type != "documents" or self.max_steps is not None:
            return self
        if any(source.repetitions is not None for source in self.data.source):
            raise ValueError(
                "data.source: repetitions sizes a source's subset by the run's tokens, so it needs max_steps"
            )
        return self

    @model_validator(mode="after")
    def validate_vlm_constraints(self):
        if self.model.vlm is None:
            return self
        if self.data.type == "documents":
            raise ValueError("[model.vlm] trains on chat data, not documents: set data.type = 'chat'.")
        if self.data.micro_batch_size != 1:
            raise ValueError("VLM training requires data.micro_batch_size = 1.")
        return self

    def renders_conversations(self) -> bool:
        """Whether the trainer renders conversations with ``renderer``: ``chat`` data without ``fake``, or a ``chat``
        validation source."""
        if self.data.type == "chat" and self.data.fake is None:
            return True
        return self.val is not None and any(val_source.type == "chat" for val_source in self.val.source)

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

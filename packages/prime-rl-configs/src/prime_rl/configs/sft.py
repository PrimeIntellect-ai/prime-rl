import uuid
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, model_validator
from renderers import AutoRendererConfig, DefaultRendererConfig, RendererConfig
from renderers.base import MODEL_RENDERER_MAP

from prime_rl.configs.eval import SFTOnlineEvalConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.inference import WeightBroadcastConfig as InferenceWeightBroadcastConfig
from prime_rl.configs.monitors import TrainMonitorsConfig
from prime_rl.configs.shared import (
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


class FakeDataConfig(BaseDataConfig):
    type: Literal["fake"] = "fake"

    length: Literal["fixed", "variable"] = "fixed"
    """Use fixed-length samples or variable-length samples."""

    input_ids: Literal["increasing", "random"] = "increasing"
    """Token id generator: ``increasing`` for deterministic sequences, ``random`` for random ids."""

    seed: int = 0
    """Seed for the per-rank packing/token generator, combined with the data rank."""


class LossMaskConfig(BaseConfig):
    system: bool = False
    """System messages contribute to the loss."""

    user: bool = False
    """User messages contribute to the loss."""

    assistant: bool = True
    """Assistant messages contribute to the loss."""

    tool: bool = False
    """Tool messages contribute to the loss."""


class SFTColumnsConfig(BaseConfig):
    """Dataset columns the trainer reads. Set a field to read a column under another name."""

    messages: str = "messages"
    """Column with the whole chat as a list of OpenAI chat messages."""

    prompt: str = "prompt"
    """Column with the prompt as a list of OpenAI chat messages, read with ``completion`` when a row has no messages."""

    completion: str = "completion"
    """Column with the completion as a list of OpenAI chat messages."""

    tools: str = "tools"
    """Column with the tool schemas in OpenAI function-calling format."""

    message_loss_mask: str = "message_loss_mask"
    """Optional column of booleans or integer 0/1 flags, one per resolved message (prompt + completion for split rows). Zero excludes a message from loss; one preserves its normal supervision. Missing or null masks leave supervision unchanged."""

    renderer: dict[str, str] = {"reasoning_effort": "reasoning_effort"}
    """Per-sample renderer arguments as ``renderer field = dataset column``, e.g. ``reasoning_effort = "effort"``. A row's non-null value overrides the ``[renderer]`` setting; rows and datasets without the column use it unchanged."""


class SFTDataConfig(BaseDataConfig):
    type: Literal["sft"] = "sft"

    name: str = "PrimeIntellect/Reverse-Text-SFT"
    """HF dataset name or path."""

    revision: str | None = None
    """HF dataset revision to load. Ignored for a local path."""

    subsets: list[str] | None = None
    """Subsets to load from the HF dataset."""

    splits: list[str] | None = None
    """Splits to load from the HF dataset."""

    probabilities: list[float] | None = None
    """Sampling probabilities for each subset/split."""

    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "all_exhausted"
    """Stopping strategy when interleaving multiple subsets/splits."""

    shuffle: bool = True
    """Shuffle the dataset at the start of each epoch."""

    seed: int = 0
    """Random seed for shuffling. Re-shuffled per epoch by adding the epoch count to the seed."""

    columns: SFTColumnsConfig = SFTColumnsConfig()
    """Columns that carry per-sample renderer arguments."""

    # Configuring
    loss_mask: LossMaskConfig = LossMaskConfig()
    """Which message types contribute to the loss."""

    @model_validator(mode="after")
    def validate_subsets_and_splits(self):
        if self.subsets is not None or self.splits is not None:
            if self.subsets is not None and self.splits is not None:
                if len(self.subsets) != len(self.splits):
                    raise ValueError(
                        "Number of subsets must be equal to number of splits. Please specify which split to load for each subset."
                    )
            if self.subsets is not None and self.probabilities is not None:
                if len(self.probabilities) != len(self.subsets):
                    raise ValueError(
                        "Number of probabilities must be equal to number of subsets. Please specify a probability for each subset."
                    )
            if self.splits is not None and self.probabilities is not None:
                if len(self.probabilities) != len(self.splits):
                    raise ValueError(
                        "Number of probabilities must be equal to number of splits. Please specify a probability for each split."
                    )
        return self


class SFTValConfig(BaseConfig):
    interval: int = Field(50, ge=1)
    """Run validation every N training steps."""

    eval_on_start: bool = False
    """Run validation before the first training step."""

    after_optimizer_step: bool = False
    """Run the interval validations after the step's optimizer update instead of before it. ``val/loss`` at
    step N is then the loss of the weights a checkpoint of step N holds, and the final step is validated as well
    (also when ``max_steps`` is not a multiple of ``interval``). ``eval_on_start`` still runs before the first
    update, except on a resume whose first step validates after its update anyway. Off: ``val/loss`` at step N
    is the loss of the weights after N - 1 updates."""

    keep_best: bool = False
    """Keep a weights-only snapshot (model parameters and buffers, no optimizer state) of the lowest validation
    loss so far under ``<checkpoint dir>/best-weights``, with ``best.json`` recording its step and loss. Each rank
    writes and reads only its own shard file, so the snapshot only reloads into the same world size and sharding.
    The ``eval_on_start`` validation is not a candidate. A new best replaces the previous snapshot."""

    keep_best_max_step: int | None = Field(None, ge=1)
    """Only validations after at most this many steps are ``keep_best`` candidates (later losses are still
    recorded). For smoke tests: forces 'best' to be an earlier step than 'last', so its export loads the snapshot.
    None: every validation is a candidate."""

    data: SFTDataConfig


class SFTExportCleanupConfig(BaseConfig):
    """Keep run data that is safe elsewhere off the volume.

    Deleting it from within the run is the cheap way to keep the volume small (deleting folders on the volume
    afterwards needs a read-write session that stages the whole volume first). The run dir behaves as shared by all
    nodes (a delete on one node removes the file for every node), so each deletion waits until the export that
    replaces the data is verified on Hugging Face."""

    best_snapshot: bool = False
    """Delete the ``[val] keep_best`` snapshot (``best-weights/``, fp32 parameters, ~0.48 TB for Nemotron 3.5 Super)
    once the 'best' export is on Hugging Face and its sha256s were verified there. A restart of the finished run then
    reuses that export as it is (it could not export 'best' again)."""

    final_checkpoint: bool = False
    """Delete the final full checkpoint (``checkpoints/step_<max_steps>``, parameters + optimizer state, ~1.45 TB)
    once the 'last' export is on Hugging Face and its sha256s were verified there. Gives up resuming the run (e.g. to
    train further from it)."""

    pushed_exports: bool = False
    """Delete the exports (``hf-export/<name>``) whose Hugging Face upload was verified, after the in-run eval (if
    any) is done with them."""


class SFTHFExportConfig(BaseConfig):
    """Export checkpoints as Hugging Face models before the trainer exits.

    The export rewrites the safetensors files of ``base_dir`` (the HF checkpoint the run started from) with the
    fine-tuned tensors, cast to the base file's dtypes. Every other tensor (e.g. an untrained MTP head) and the
    shard layout, index, config and tokenizer stay the base's. The result goes to
    ``<checkpoint dir>/hf-export/<name>``, is checked against the base (keys, dtypes, shapes, relative change of
    each tensor) and, if a repo is set and ``HF_TOKEN`` is in the environment, pushed. A failed export or push
    is logged and never fails the run."""

    checkpoints: list[Literal["best", "last"]] = ["best", "last"]
    """Which checkpoints to export: ``last`` is the final weights, ``best`` the ``[val] keep_best`` snapshot."""

    best_repo: str | None = None
    """Hugging Face repo id (``org/name``) for the best checkpoint. None keeps it on the volume only."""

    last_repo: str | None = None
    """Hugging Face repo id (``org/name``) for the last checkpoint. None keeps it on the volume only."""

    private: bool = True
    """Create the repos as private."""

    base_dir: Path | None = None
    """Local HF checkpoint whose files are rewritten. Defaults to ``model.name``, which must then be a local
    directory."""

    keep_base_prefixes: list[str] = ["mtp."]
    """Base tensors under these name prefixes are not held by the trainer (prime-rl's NemotronH drops the MTP
    head) and are copied from the base unchanged. Every other base tensor must come from the trained model."""

    max_relative_change: float = Field(0.2, gt=0)
    """Refuse to push when any fine-tuned tensor moved further than this (relative L2) from the base: that
    points to a wrong key mapping, not training."""

    model_card: str = ""
    """Free text added to the generated README.md (what was trained on, how to run it)."""

    cleanup: SFTExportCleanupConfig = SFTExportCleanupConfig()
    """What to delete once the exports made it safe (default: nothing)."""

    @model_validator(mode="after")
    def validate_checkpoints(self):
        if not self.checkpoints or len(set(self.checkpoints)) != len(self.checkpoints):
            raise ValueError("hf_export.checkpoints must be a non-empty list without duplicates")
        for repo in (self.best_repo, self.last_repo):
            if repo is not None and repo.count("/") != 1:
                raise ValueError(f"hf_export repo {repo!r} must be 'org/name'")
        return self


TB4_SOURCE_LABELS = ("tb-eng", "tb-eng-extra9", "tb-sci")


class SFTInRunEvalServingConfig(BaseConfig):
    """vLLM on every node of the run, serving the checkpoint under evaluation. The defaults are the TB4 eval spec
    (EVAL-HARNESS-REPLICA-SPEC.md): vLLM 0.32.0 from the image, the GA chat template, thinking on with the history's
    reasoning kept, temperature 1.0 / top_p 0.95 as server defaults, no max_tokens, no speculative decoding."""

    tensor_parallel_size: int = Field(2, ge=1)
    """GPUs per vLLM engine."""

    data_parallel_size: int = Field(4, ge=1)
    """Engines per node (one vLLM data-parallel group per node, each engine its own API server). ``tensor_parallel_size
    * data_parallel_size`` must be the node's GPU count."""

    enable_expert_parallel: bool = True
    """Shard the experts over the node's engines (EP = TP x DP)."""

    max_model_len: int = Field(1048576, ge=1)
    """Context window (the model's 1M, as trained)."""

    served_model_name: str = "nemotron-3.5-super-tb4-sft"
    """The model name the harness sends and the agent can see (mini's args, its trajectory file): the same for every
    evaluated checkpoint, so what the agent sees does not depend on which one it is."""

    chat_template: str = "chat_template.jinja"
    """Chat template file, relative to the served model directory (or absolute)."""

    chat_template_sha256: str | None = "1369e90b19976f99ce2070eb21ade8fcb4af640f008e14e82af5c5e89a9cc816"
    """Refuse to serve unless the template has this sha256 (the GA template). None skips the check."""

    default_chat_template_kwargs: dict[str, Any] = {"enable_thinking": True, "truncate_history_thinking": False}
    """Server-side template defaults (the training rendering keeps past turns' reasoning)."""

    override_generation_config: dict[str, Any] = {"temperature": 1.0, "top_p": 0.95}
    """Sampling defaults for requests that do not set them (the harness sends temperature 1.0 and no top_p)."""

    reasoning_parser: str = "nemotron_v3"
    tool_call_parser: str = "qwen3_coder"

    mamba_cache_dtype: str = "float32"
    enable_prefix_caching: bool = True
    dtype: str = "bfloat16"
    seed: int = 0
    gpu_memory_utilization: float = Field(0.85, gt=0, le=1)
    max_num_seqs: int = Field(256, ge=1)
    max_num_batched_tokens: int = Field(8192, ge=1)

    vllm_extra: dict[str, Any] = {
        "all2all_backend": "allgather_reducescatter",
        "safetensors_load_strategy": "prefetch",
    }
    """Further vLLM engine arguments (vLLM's names), passed through as they are."""

    env_vars: dict[str, str] = {}
    """Extra environment for the engines (on top of prime-rl's inference defaults)."""

    engine_port: int = Field(8100, ge=1, le=65535)
    """Engine ``d`` of a node listens on ``engine_port + d``."""

    router_port: int = Field(8000, ge=1, le=65535)
    """The router of the i-th checkpoint served at the same time listens on node 0 at ``router_port + i``."""

    router_policy: str = "sticky_least_loaded"
    """vllm-router policy: keeps a rollout (its ``X-Session-ID``) on one engine for its prefix cache."""

    router_request_timeout_secs: int = Field(14400, ge=1)

    rpc_port: int = Field(13345, ge=1, le=65535)
    """vLLM data-parallel RPC port (node-local)."""

    startup_timeout_seconds: float = Field(7200, gt=0)
    """How long the engines of a node may take to load and compile before the node reports failure."""

    @model_validator(mode="after")
    def validate_spec(self):
        if "speculative_config" in self.vllm_extra:
            raise ValueError("inrun_eval.serving: no speculative decoding (the exports' MTP head is untrained)")
        for key in ("max_tokens", "max_new_tokens"):
            if key in self.override_generation_config:
                raise ValueError(f"inrun_eval.serving: no {key} (the harness runs without an output cap)")
        clash = {"model", "served_model_name", "tensor_parallel_size", "data_parallel_size"} & set(self.vllm_extra)
        if clash:
            raise ValueError(f"inrun_eval.serving.vllm_extra must not set {sorted(clash)} (set by the eval)")
        return self


class SFTInRunEvalHarnessConfig(BaseConfig):
    """The collection harness (sft-gen ``flow run`` of the task-QC pipeline, measure preset: k rollouts per task, no QC
    seats) at its collection pins, run on node 0 in its own virtualenv. Everything private (the sft-gen tree, the flow
    configs of the collection runs, the Harbor task caches, wheels) comes from ``bundle_dir`` on the volume; see
    PREREQS.md."""

    bundle_dir: Path = Path("/volume/datasets/tb4-eval")
    """The staged eval bundle (``manifest.json`` lists every file with its sha256)."""

    sources: list[Literal["tb-eng", "tb-eng-extra9", "tb-sci"]] = list(TB4_SOURCE_LABELS)
    """The collection runs to replay, one ``flow run`` each, all three at the same time."""

    source_config_sha256: dict[str, str] = {
        "tb-eng": "d0e5b6c7bd031fcffaf6438210dbab60f78dcdd26ac26f887a32df719858ae5a",
        "tb-eng-extra9": "86a3fba822506c754a192edb4cb14926767ed33fff47a277cb51b8c2227e6512",
        "tb-sci": "06d60eb7f1eb1dbc8def8124e9d9d0097c8d6a4d82c0b3a492aa4d99b2084a92",
    }
    """sha256 of each run's ``configs/attempt_2/flow.toml`` (the collection config, verifiers 24107a4ea)."""

    tb_sci_dataset: str | None = "tb-science-vendor@71124a84e6f3"
    """Dataset of the tb-sci run instead of its collection pin (85df9700aaf4): 71124a84e6f3 is the same 290 tasks at
    the same commits plus convolution_filters_english, the 500th task. None keeps the collection pin."""

    k: int | None = Field(None, ge=1)
    """Rollouts per task. None keeps the collection configs' k (8)."""

    tasks_file: Path | None = None
    """Restrict the eval to the task names in this file (one per line; e.g. the smoke's three). None: every task of
    every source."""

    expected_tasks: int | None = 500
    """Refuse to start when the selected tasks are not this many (None: no check)."""

    expected_val_tasks: int | None = 50
    """The bundle's task table must label exactly this many tasks ``val`` (the summaries' val / train split): the v2
    dataset's validation split, ``data-prep/tb4-val-tasks-v2.txt`` (the 44 frozen tasks plus 6 zero-pass ones, 10%).
    None: no check."""

    rollouts_per_engine: int = Field(24, ge=1)
    """Rollouts in flight per vLLM engine serving the checkpoint (the rollout gate is engines x this)."""

    max_concurrent_rollouts: int | None = Field(None, ge=1)
    """Rollouts in flight per checkpoint, over all sources, instead of engines x ``rollouts_per_engine``."""

    api_key_var: str = "TB4_EVAL_VLLM_KEY"
    """Environment variable the harness client reads the (unchecked) vLLM key from; set to a random value."""

    prime_team_id: str | None = None
    """The Prime team that owns the private task images. None: ``PRIME_TEAM_ID`` from the environment (the run
    config's ``[env_vars]``). The Prime API key is always the platform's per-run ``PRIME_API_KEY`` (``prime train -e``
    forwards only HF_TOKEN / WANDB_API_KEY): without it the eval records ``no_credentials`` and serves nothing."""

    credential_probe: bool = True
    """Before anything is served, list one of the team's sandboxes and one tunnel with the run's ``PRIME_API_KEY``
    (cheap GETs to the Prime API): a 401 ends the eval phase as ``no_credentials``, a 402 / 403 as ``forbidden``
    (``eval/status.json``), nothing is loaded; a network error or a 5xx that persists is recorded and the eval goes on.
    The key is never logged."""

    python: str = "3.13"
    """Python of the harness virtualenv (sft-gen runs on 3.13; the image only has 3.12)."""

    verifiers_version: str = "0.4.1.dev340"
    """Installed verifiers version (24107a4ea)."""

    prime_flow_version: str | None = "0.1.1.dev354"
    """Installed prime-flow version (ed23e23). None skips the check."""

    pinned_files_sha256: dict[str, str] = {
        "verifiers/v1/harnesses/mini_swe_agent/harness.py": "f7ac569d865c4049752b0b09cd920d90689f38c2297ec49e76409b533edc316d",
        "verifiers/v1/harnesses/mini_swe_agent/program.py": "1467b440577f1a2fee1f628abdb3bd60b8375a2116b2e193ab97c752c14d0297",
        "verifiers/v1/runtimes/runtime.py": "4842d302967a4149d73f1f2d5756f9c3274f5568c4d36cbe1dc117dbab5c689d",
        "verifiers/v1/runtimes/base.py": "6af2b4776bdc718c2b24c83ce094e2e3bfe0314b69bac1bedf4ac16a4d799fca",
        "verifiers/v1/runtimes/host_uv.py": "8f3ab8a846ace1dcb09d4e600074fe4dc9c70520b0f5be4c7718c993cf4c8be9",
    }
    """sha256 of the installed agent-facing verifiers files (relative to site-packages), as at 24107a4ea (program.py
    sets the leaked venv path ``.../2cc22997...-fd57668a0b15dabf``: it is never modified)."""

    d5_exclude_newer: str | None = "2026-10-24T07:23:00Z"
    """Spec decision D5: the in-sandbox ``uv sync --script`` that builds mini-swe-agent's venv resolves its (unlocked)
    dependencies as of this time, the attempt_2 launch, so the agent sees the packages of collection time instead of
    today's PyPI. Applied through the ``tb4-mini-swe-agent-d5`` harness (``inrun_eval/plugins``): it passes
    ``UV_EXCLUDE_NEWER`` only to the preparation step; the agent's command environment, ``program.py`` and the venv
    path are unchanged. None: run the collection's ``mini_swe_agent`` harness as is (dependencies float)."""

    in_box_uv_version: str | None = "0.13.5"
    """With ``d5_exclude_newer``: when a task image has no script-capable uv, install this uv version (collection
    boxes reported 0.13.5) the way verifiers' bootstrap would (``pip install --user``, else the installer script)
    instead of the latest. None: the bootstrap's own unpinned install."""

    flow_command: list[str] = ["{venv}/bin/flow", "run", "@", "{config}"]
    """How one source's flow run is started (``{venv}`` and ``{config}`` are filled in)."""

    landlock_bundle: str = "/data/sft-gen/kernels/landlock-6.12.113-deb13.tgz"
    """Where the collection configs expect the landlock kernel bundle (2 tb-eng tasks); copied there from the bundle."""

    @model_validator(mode="after")
    def validate_sources(self):
        if not self.sources or len(set(self.sources)) != len(self.sources):
            raise ValueError("inrun_eval.harness.sources must be a non-empty list without duplicates")
        missing = [label for label in self.sources if label not in self.source_config_sha256]
        if missing:
            raise ValueError(f"inrun_eval.harness.source_config_sha256 lacks {missing}")
        if self.d5_exclude_newer is not None:
            from datetime import datetime

            try:
                stamp = datetime.fromisoformat(self.d5_exclude_newer.replace("Z", "+00:00"))
            except ValueError:
                stamp = None
            if stamp is None or stamp.tzinfo is None:
                raise ValueError("inrun_eval.harness.d5_exclude_newer must be an RFC 3339 time with a zone (…Z)")
        return self


class SFTInRunEvalConfig(BaseConfig):
    """After the HF export, turn the run's nodes from trainer into inference and evaluate the exported checkpoints with
    the collection harness: every node serves vLLM, node 0 runs the harness, scores pass@k and writes the results to
    ``<run dir>/eval/<checkpoint>/`` (node 0's files only). An eval failure never fails the run: the exports are
    written (and pushed) before it starts."""

    checkpoints: list[Literal["best", "last"]] = ["best", "last"]
    """Which exports to evaluate, in this order. When 'best' is the final step it is 'last': evaluated once."""

    model_dir: Path | None = None
    """Evaluate this HF checkpoint (same path on every node) instead of the run's exports, as ``model_name``."""

    model_name: str = "base"
    """Name of ``model_dir``'s results."""

    eval_only: bool = False
    """Skip training altogether (no model, no data loaded) and only evaluate ``model_dir``: the smoke test."""

    schedule: Literal["sequential", "parallel"] = "sequential"
    """``sequential``: every node serves one checkpoint after the other (a vLLM restart in between). ``parallel``: the
    nodes are split in two and both checkpoints are evaluated at the same time (one load, the same sandbox
    conditions); needs at least 2 nodes."""

    budget_hours: float | None = Field(None, gt=0)
    """Optional safety cap on the eval phase, from the hand-off. None (the default): no cap beyond the collection
    configs' own timeouts (rollout 86400 s, scoring up to 21600 s per task, stage 1209600 s), so every draw runs to
    its end as in collection. When set, each round gets an equal share of what is left (minus ``reserve_minutes``);
    at its end the harness is stopped, and the draws it cut are reported as ``cut``, excluded from pass@k (the
    summary states the coverage), never scored as failures."""

    reserve_minutes: float = Field(60.0, ge=0)
    """With ``budget_hours``: kept from the cap for scoring, archiving and the shutdown."""

    evaluate_aliases: bool = False
    """Evaluate an export that is a copy of another ('best' at the final step) in a round of its own instead of
    once. Only for the smoke test (it exercises the vLLM restart between rounds)."""

    hf_fallback: bool = True
    """When a node's local copy of an export is incomplete, download it from its Hugging Face repo (needs HF_TOKEN)."""

    work_dir: Path | None = None
    """Node-local scratch (harness virtualenv, flow stores, vLLM configs). Defaults to ``<run dir>/.inrun-eval-work``
    (the node's staging disk), deleted before the exit unless ``keep_work_dir``."""

    keep_work_dir: bool = False

    archive_stores: bool = True
    """Copy each flow store (full traces) to the results as a compressed tar."""

    pass_threshold: float = 1.0
    """A rollout passes when its reward is at least this."""

    coord_port: int = Field(29650, ge=1, le=65535)
    """Port of the eval's coordination store on node 0 (``MASTER_ADDR``)."""

    handoff: Literal["exec", "subprocess"] = "exec"
    """How each node's local rank 0 becomes the node's eval process: ``exec`` replaces the trainer process (its GPU
    and host memory are released), ``subprocess`` keeps it waiting for a child."""

    serving: SFTInRunEvalServingConfig = SFTInRunEvalServingConfig()
    harness: SFTInRunEvalHarnessConfig = SFTInRunEvalHarnessConfig()

    @model_validator(mode="after")
    def validate_selection(self):
        if not self.checkpoints or len(set(self.checkpoints)) != len(self.checkpoints):
            raise ValueError("inrun_eval.checkpoints must be a non-empty list without duplicates")
        if self.eval_only and self.model_dir is None:
            raise ValueError("inrun_eval.eval_only needs inrun_eval.model_dir (there are no exports to evaluate)")
        if self.budget_hours is not None and self.reserve_minutes >= self.budget_hours * 60:
            raise ValueError("inrun_eval.reserve_minutes must be less than the budget")
        return self


DataConfig: TypeAlias = Annotated[FakeDataConfig | SFTDataConfig, Field(discriminator="type")]


class BaseDeploymentConfig(BaseConfig):
    gpus_per_node: int = 8
    """GPUs per node."""


class SingleNodeDeploymentConfig(BaseDeploymentConfig):
    type: Literal["single_node"] = "single_node"

    num_train_gpus: int = 1
    """GPUs allocated to the trainer."""

    num_infer_gpus: int = Field(1, validation_alias=AliasChoices("num_infer_gpus", "num_eval_gpus"))
    """GPUs allocated to inference for online evals (alias: ``num_eval_gpus``). Only used when an ``[inference]`` block is configured."""

    @model_validator(mode="after")
    def validate_gpu_count(self):
        if self.num_train_gpus > self.gpus_per_node:
            raise ValueError(f"num_train_gpus ({self.num_train_gpus}) exceeds gpus_per_node ({self.gpus_per_node}).")
        return self


class MultiNodeDeploymentConfig(BaseDeploymentConfig):
    type: Literal["multi_node"] = "multi_node"

    num_train_nodes: int = Field(2, ge=1)
    """Training nodes."""

    num_infer_nodes: int = Field(0, ge=0, validation_alias=AliasChoices("num_infer_nodes", "num_eval_nodes"))
    """Inference nodes for online evals (alias: ``num_eval_nodes``). These nodes share
    one SLURM allocation with the trainer nodes."""

    nodes_per_fsdp_group: int | None = None
    """Nodes per FSDP island. Auto-sets ``model.dp_replicate = num_train_nodes / nodes_per_fsdp_group``."""


SFTDeploymentConfig: TypeAlias = Annotated[
    SingleNodeDeploymentConfig | MultiNodeDeploymentConfig, Field(discriminator="type")
]


class SFTConfig(BaseTrainerConfig):
    auto_freeze_moe_router = False
    # Standard L2 regularization for supervised training.
    auto_weight_decay = 0.01

    renderer: RendererConfig = AutoRendererConfig()
    """Renderer config. Defaults to auto-selecting from the tokenizer model name."""

    data: DataConfig = SFTDataConfig()

    val: SFTValConfig | None = None
    """Validation configuration. If None, no validation runs."""

    hf_export: SFTHFExportConfig | None = None
    """Export the best and/or last checkpoint as Hugging Face models at the end of the run. If None, no export."""

    inrun_eval: SFTInRunEvalConfig | None = None
    """After the export, evaluate the exported checkpoints on the run's own nodes with the TB4 collection harness. If
    None (the default), the run ends after the export."""

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

    monitors: TrainMonitorsConfig = TrainMonitorsConfig()
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
        """Auto-generate the run name (``<dataset>--<model>--<short-id>``) when unset and
        default the run directory, W&B run name and platform run name to it when not
        set explicitly."""
        if self.run.name is None:
            dataset = str(getattr(self.data, "name", "")).split("/")[-1]
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

    deployment: SFTDeploymentConfig = SingleNodeDeploymentConfig()

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
    def normalize_deployment(cls, data):
        if not isinstance(data, dict):
            return data
        deployment = data.get("deployment")
        if isinstance(deployment, dict) and deployment.get("type") == "multi_node":
            for key in ("num_train_gpus", "num_infer_gpus"):
                deployment.pop(key, None)
        return data

    @model_validator(mode="before")
    @classmethod
    def propagate_model_to_inference(cls, data):
        """Fill ``inference.vllm.model`` from ``model.name`` before InferenceConfig
        construction, so its parser auto-resolution sees the right model name."""
        if not isinstance(data, dict):
            return data
        inference = data.get("inference")
        model = data.get("model")
        name = model.get("name") if isinstance(model, dict) else None
        if isinstance(inference, dict) and name:
            vllm = inference.setdefault("vllm", {})
            if isinstance(vllm, dict):
                vllm.setdefault("model", name)
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

        if self.weight_broadcast is None:
            if self.model.lora is not None or self.inference is None:
                self.weight_broadcast = FileSystemWeightBroadcastConfig()
            else:
                self.weight_broadcast = NCCLWeightBroadcastConfig()
        if self.weight_broadcast.type != "filesystem":
            if self.weight_broadcast.type == "nixl":
                raise ValueError("NIXL weight broadcast is not supported for SFT online evals.")
            if self.model.lora is not None:
                raise ValueError(
                    "LoRA training is not yet supported with in-memory weight broadcast. "
                    "Set weight_broadcast.type = 'filesystem'."
                )

        if self.deployment.type == "multi_node":
            # Dedicated nodes in the SFT allocation run the inference pool, router,
            # env servers, and evals process.
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
    def validate_typed_renderer(self):
        """Require a typed renderer whenever SFT renders real samples."""
        if self.data.type == "fake" and self.val is None:
            return self

        model_id = self.tokenizer.name or self.model.name
        if isinstance(self.renderer, AutoRendererConfig):
            if model_id in MODEL_RENDERER_MAP:
                return self
            reason = f"no typed renderer is registered for {model_id!r}"
        elif isinstance(self.renderer, DefaultRendererConfig):
            reason = "renderer.name='default' selects DefaultRenderer"
        else:
            return self

        raise ValueError(
            f"SFT requires a typed renderer with sampled-token and content attribution, but {reason}. "
            "Implement and register the renderer in the renderers package, or explicitly select an existing "
            "typed renderer only when its template is verified to match."
        )

    @model_validator(mode="after")
    def validate_cp_seq_len(self):
        if self.model.cp > 1:
            if self.data.seq_len % self.model.cp != 0:
                raise ValueError("Sequence length must be divisible by CP degree")
            if self.val is not None and self.val.data.seq_len % self.model.cp != 0:
                raise ValueError("Validation sequence length must be divisible by CP degree")
        return self

    @model_validator(mode="after")
    def resolve_mega_moe_max_tokens_per_rank(self):
        dispatch = self.model.moe.dispatch
        if dispatch.type == "mega_moe":
            data_configs = [self.data] + ([self.val.data] if self.val is not None else [])
            dispatch.max_tokens_per_rank = mega_moe_max_tokens_per_rank(
                dispatch.max_tokens_per_rank, max(data.micro_batch_size * data.seq_len for data in data_configs)
            )
        return self

    @model_validator(mode="after")
    def validate_cp_micro_batch_size(self):
        if self.model.cp > 1:
            if self.data.micro_batch_size != 1:
                raise ValueError("Micro batch size must be 1 when CP is enabled")
            if self.val is not None and self.val.data.micro_batch_size != 1:
                raise ValueError("Validation micro batch size must be 1 when CP is enabled")
        return self

    @model_validator(mode="after")
    def validate_best_and_hf_export(self):
        if self.hf_export is None:
            return self
        if "best" in self.hf_export.checkpoints and (self.val is None or not self.val.keep_best):
            raise ValueError("hf_export of the 'best' checkpoint needs [val] keep_best = true")
        if self.model.lora is not None:
            raise ValueError("hf_export writes full weights and does not support LoRA")
        if self.model.vlm is not None:
            raise ValueError("hf_export does not support VLM training")
        return self

    @model_validator(mode="after")
    def validate_inrun_eval(self):
        cfg = self.inrun_eval
        if cfg is None:
            return self
        serving = cfg.serving
        gpus = self.deployment.gpus_per_node
        if serving.tensor_parallel_size * serving.data_parallel_size != gpus:
            raise ValueError(
                f"inrun_eval.serving: tensor_parallel_size x data_parallel_size must be gpus_per_node ({gpus})"
            )
        if cfg.model_dir is None:
            if self.hf_export is None:
                raise ValueError("inrun_eval evaluates the [hf_export] exports: configure [hf_export] or model_dir")
            missing = [name for name in cfg.checkpoints if name not in self.hf_export.checkpoints]
            if missing:
                raise ValueError(f"inrun_eval.checkpoints {missing} are not in hf_export.checkpoints")
        if self.eval is not None:
            raise ValueError("inrun_eval and the online [eval] both need the GPUs: use one of them")
        return self

    @model_validator(mode="after")
    def validate_vlm_constraints(self):
        if self.model.vlm is None:
            return self
        if self.model.optimization_dtype != "bfloat16" or self.model.reduce_dtype != "bfloat16":
            raise ValueError(
                "VLM models must use optimization_dtype='bfloat16' and reduce_dtype='bfloat16' to match vLLM inference."
            )
        if self.data.micro_batch_size != 1:
            raise ValueError("VLM SFT requires data.micro_batch_size = 1.")
        if self.val is not None and self.val.data.micro_batch_size != 1:
            raise ValueError("VLM SFT requires val.data.micro_batch_size = 1.")
        return self

    ### Auto-setup and validate shared configs

    @model_validator(mode="after")
    def auto_setup_deployment(self):
        if self.deployment.type == "multi_node":
            if self.deployment.nodes_per_fsdp_group is not None:
                if self.deployment.num_train_nodes % self.deployment.nodes_per_fsdp_group != 0:
                    raise ValueError(
                        f"deployment.num_train_nodes ({self.deployment.num_train_nodes}) must be divisible by "
                        f"deployment.nodes_per_fsdp_group ({self.deployment.nodes_per_fsdp_group})"
                    )
                self.model.dp_replicate = self.deployment.num_train_nodes // self.deployment.nodes_per_fsdp_group
        return self

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

"""CPU tests of the in-run eval's offline pieces: task selection, pass@k scoring of a flow store's trace index, the
flow-config derivation, the plan, the vLLM / router commands, the harness install and environment, the coordination
store and the staging cleanup. Synthetic task names only (the real ones are private)."""

import hashlib
import itertools
import json
import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomli_w

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.sft import SFTInRunEvalConfig
from prime_rl.trainer.sft.inrun_eval import flowcfg, harness, node, plan, scoring, serving, tasks
from prime_rl.trainer.sft.inrun_eval.coord import Coordinator, MemoryStore

# Tasks


TABLE = """# name\tsource\tsplit
terminal-bench/alpha_one\ttb-sci\tval
terminal-bench/beta-two\ttb-sci\ttrain
turing/gamma-three\ttb-eng\ttrain
turing/delta\ttb-eng-extra9\tother
"""


def test_task_table_and_selection():
    table = tasks.parse_task_table(TABLE)
    assert [row.source for row in table] == ["tb-sci", "tb-sci", "tb-eng", "tb-eng-extra9"]
    assert tasks.select_tasks(table, ["tb-eng", "tb-eng-extra9", "tb-sci"]) == {
        "tb-eng": ["turing/gamma-three"],
        "tb-eng-extra9": ["turing/delta"],
        "tb-sci": ["terminal-bench/alpha_one", "terminal-bench/beta-two"],
    }
    # A restriction (the smoke) drops sources left without tasks: an empty include list would select everything.
    assert tasks.select_tasks(table, ["tb-eng", "tb-sci"], ["terminal-bench/beta-two"]) == {
        "tb-sci": ["terminal-bench/beta-two"]
    }
    with pytest.raises(ValueError, match="not in the task table"):
        tasks.select_tasks(table, ["tb-sci"], ["nope"])
    with pytest.raises(ValueError, match="unknown source"):
        tasks.parse_task_table("a\ttb-x\tval\n")
    with pytest.raises(ValueError, match="listed twice"):
        tasks.parse_task_table("a\ttb-sci\tval\na\ttb-eng\tval\n")


def test_job_slugs_match_prime_flow():
    # prime_flow qc.solving.job_id at ed23e23: slug(name)[:48] + "-" + key[:12] + "-" + draw.
    assert tasks.slug("terminal-bench/Chapter_Loudness") == "terminal-bench-chapter-loudness"
    assert tasks.job_slug("terminal-bench-chapter-loudness-697db0896cda-qc") == "terminal-bench-chapter-loudness"
    assert tasks.job_slug("697db0896cda-qc") == ""
    assert tasks.job_slug("no-key-here-qc") is None
    long = "x" * 60
    assert tasks.job_slug(f"{tasks.slug(long)}-0123456789ab-qc") == "x" * 48
    unique, ambiguous = tasks.slug_index(["a/b-c", "a/b_c", "d/e"])
    assert unique == {"d-e": "d/e"} and ambiguous == {"a-b-c"}


def test_harbor_cache_names():
    # The cache directories verifiers 24107a4ea computes for the collection's datasets (checked against its source).
    repo = "PrimeIntellect-ai/nvidia-delivery@main"
    assert {
        tasks.harbor_cache_name("tb-engineering-vendor@1eb123da72ed", repo, "registries/tb-engineering.json"),
        tasks.harbor_cache_name("tb-engineering-vendor@37107d21ec3d", repo, "registries/tb-engineering.json"),
        tasks.harbor_cache_name("tb-science-vendor@85df9700aaf4", repo, "registries/tb-science.json"),
        tasks.harbor_cache_name("tb-science-vendor@71124a84e6f3", repo, "registries/tb-science.json"),
    } == {
        "tb-engineering-vendor_1eb123da72ed_8cfd9e1a5499",
        "tb-engineering-vendor_37107d21ec3d_4a76686edb11",
        "tb-science-vendor_85df9700aaf4_afefdd67189a",
        "tb-science-vendor_71124a84e6f3_d4e4ed50a94a",
    }


# Scoring


def test_pass_at_k_matches_enumeration():
    for n, c in itertools.product(range(1, 9), range(0, 9)):
        if c > n:
            continue
        outcomes = [1] * c + [0] * (n - c)
        for k in range(1, n + 1):
            subsets = list(itertools.combinations(range(n), k))
            brute = sum(any(outcomes[i] for i in s) for s in subsets) / len(subsets)
            assert math.isclose(scoring.pass_at_k(n, c, k), brute), (n, c, k)
    assert scoring.pass_at_k(8, 3, 1) == 3 / 8
    assert scoring.report_ks(8) == [1, 2, 4, 8] and scoring.report_ks(1) == [1] and scoring.report_ks(6) == [1, 2, 4, 6]


def row(job, draw, *, ok=True, reward=1.0, stop="agent_completed", error=None, seat="solver", tid=None):
    return {
        "id": tid or f"{job}-{draw}-{reward}-{error}",
        "labels": {"flow": {"job": job, "call": f"solver/draw{draw}-1c9ed8c54399", "seat": seat, "stage": "x"}},
        "agent": seat,
        "ok": ok,
        "error": error,
        "stop_condition": stop,
        "reward": reward,
    }


def test_score_rows():
    table = tasks.parse_task_table(TABLE)
    sci = [r for r in table if r.source == "tb-sci"]
    a = f"{tasks.slug(sci[0].name)}-0123456789ab-qc"
    b = f"{tasks.slug(sci[1].name)}-ba9876543210-qc"
    rows = [
        row(a, 0, reward=1.0),
        row(a, 1, reward=0.0),
        # A HarnessError run, then its rerun that finished: the finished one counts.
        row(a, 2, ok=False, reward=None, stop="error", error="HarnessError"),
        row(a, 2, reward=1.0),
        # Fractional reward below the bar.
        row(a, 3, reward=0.95),
        # Only an errored row: an error (a failure), reported.
        row(a, 4, ok=True, reward=None, stop="provider_error", error="ProviderError"),
        # Draw 5 never landed: missing. Draws 6, 7 pass.
        row(a, 6),
        row(a, 7),
        # Not solver draws: ignored.
        row(a, 0, seat="investigator"),
        {"labels": {"flow": {"job": a, "call": "tester/run0", "seat": "tester"}}, "ok": True, "reward": 1.0},
        # Draws beyond k are counted apart.
        row(b, 8),
        row(b, 0, reward=None),  # ok but ungraded: a failure
    ]
    scores, diag = scoring.score_rows(rows, sci, "tb-sci", k=8, pass_threshold=1.0)
    by_task = {s.task: s for s in scores}
    first = by_task[sci[0].name]
    assert first.c == 4 and first.split == "val"
    assert [d.status for d in first.draws] == [
        "passed",
        "failed",
        "passed",
        "failed",
        "error",
        "missing",
        "passed",
        "passed",
    ]
    assert first.draws[2].rows == 2 and first.draws[4].error == "ProviderError"
    second = by_task[sci[1].name]
    assert second.c == 0 and [d.status for d in second.draws][:2] == ["failed", "missing"]
    assert diag["draw_rows_beyond_k"] == 1 and diag["unmapped_jobs"] == []

    summary = scoring.summary_by_group(scores)
    overall = summary["overall"]
    assert overall["tasks"] == 2 and overall["rollouts"] == 16
    assert overall["passed"] == 4 and overall["errored"] == 1 and overall["missing"] == 1 + 7
    assert math.isclose(overall["pass@1"], (4 / 8 + 0) / 2)
    assert overall["pass@8"] == 0.5 and overall["tasks_solved"] == 1
    assert math.isclose(overall["pass@2"], (scoring.pass_at_k(8, 4, 2) + 0) / 2)
    assert set(summary["split"]) == {"val", "train"} and summary["split"]["val"]["pass@1"] == 0.5
    assert summary["source"]["tb-sci"]["tasks"] == 2
    assert overall["cut"] == 0 and overall["coverage"]["fraction"] == 1.0 and "pass@1_tasks" not in overall


def test_cut_draws_are_excluded_from_pass_at_k():
    """A source the safety cap stopped: its unfinished draws are 'cut', not failures; pass@k is over what is left."""
    table = tasks.parse_task_table(TABLE)
    sci = [r for r in table if r.source == "tb-sci"]
    a = f"{tasks.slug(sci[0].name)}-0123456789ab-qc"
    b = f"{tasks.slug(sci[1].name)}-ba9876543210-qc"
    rows = [row(a, 0, reward=1.0), row(a, 1, reward=0.0), row(a, 2, ok=False, reward=None, stop="error", error="H")]
    rows += [row(b, n, reward=1.0) for n in range(8)]
    scores, diag = scoring.score_rows(rows, sci, "tb-sci", k=8, pass_threshold=1.0, cut=True)
    by_task = {s.task: s for s in scores}
    first = by_task[sci[0].name]
    assert [d.status for d in first.draws] == ["passed", "failed"] + ["cut"] * 6
    assert first.cut == 6 and first.graded == 2 and first.c == 1 and diag["cut_by_cap"] is True
    overall = scoring.summarize(scores)
    assert overall["cut"] == 6 and overall["errored"] == 0 and overall["missing"] == 0 and overall["failed"] == 1
    # pass@1 over both tasks (each has >= 1 draw): (1/2 + 8/8) / 2; pass@2 too; pass@4 / pass@8 only the full task.
    assert math.isclose(overall["pass@1"], (0.5 + 1.0) / 2) and overall["pass@1_tasks"] == 2
    assert math.isclose(overall["pass@2"], (scoring.pass_at_k(2, 1, 2) + 1.0) / 2) and overall["pass@2_tasks"] == 2
    assert overall["pass@8"] == 1.0 and overall["pass@8_tasks"] == 1
    cov = overall["coverage"]
    assert cov["draws_counted"] == 10 and cov["draws_planned"] == 16 and cov["tasks_with_all_draws"] == 1
    assert "not comparable" in cov["note"]
    # The same rows without the cap: the unfinished draws are failures (error / missing).
    scores, _ = scoring.score_rows(rows, sci, "tb-sci", k=8, pass_threshold=1.0)
    assert scoring.summarize(scores)["pass@1"] == (1 / 8 + 1.0) / 2


def test_unmapped_jobs_resolve_from_traces(tmp_path):
    # Two names with the same slug: their jobs are resolved from the store's episodes.
    names = [tasks.TaskRow("x/a-b", "tb-eng", "train"), tasks.TaskRow("x/a_b", "tb-eng", "val")]
    job1, job2 = "x-a-b-0123456789ab-qc", "x-a-b-ba9876543210-qc"
    rows = [row(job1, 0, tid="t1"), row(job2, 0, reward=0.0, tid="t2"), row("weird-job", 0, tid="t3")]
    store = tmp_path / "store"
    (store / "traces").mkdir(parents=True)
    with open(store / "traces" / "index.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write('{"torn": ')  # a killed writer's last line
    with open(store / "traces.jsonl", "w") as f:
        f.write(json.dumps({"traces": [{"id": "t1", "task": {"data": {"name": "x/a-b"}}}]}) + "\n")
        f.write(json.dumps({"traces": [{"id": "t2", "task": {"data": {"name": "x/a_b"}}}]}) + "\n")
    scores, diag = scoring.score_store(store, names, "tb-eng", k=1, pass_threshold=1.0)
    assert {s.task: s.c for s in scores} == {"x/a-b": 1, "x/a_b": 0}
    assert diag["unmapped_jobs"] == ["weird-job"] and diag["rows"] == 3


# Flow configs


SOURCE = """dashboard = false

[run]
name = "tb-science-glm-measure-20261023"
output_dir = "/data/sft-gen/vendor-draws/glm"

[flow]
id = "sft-gen-task-qc"
store = "/data/sft-gen/vendor-draws/glm/tb-science-glm-measure-20261023/store"
model = "internal/glm-5.3-high-throughput"
concurrency = 250

[flow.sampling]
temperature = 1.0
reasoning_effort = "high"

[flow.taskset]
id = "sft-gen-kernel-harbor"
dataset = "tb-science-vendor@85df9700aaf4"
repo = "PrimeIntellect-ai/nvidia-delivery@main"
registry_path = "registries/tb-science.json"

[flow.taskset.task.prime_kernels."turing/k"]
bundle = "/data/sft-gen/kernels/landlock.tgz"
minimum_landlock_abi = 4

[flow.select.include]
names = []

[flow.pieces.rollouts]
on = true
k = 8

[flow.solver]
model = "internal/glm-5.3-high-throughput"
network_notice = true

[flow.solver.harness]
id = "mini_swe_agent"
command_timeout = 1800

[flow.solver.client]
base_url = "https://api.pinference.ai/api/v1"
api_key_var = "PRIME_API_KEY"
type = "eval"

[flow.solver.client.timeout]
connect = 30.0
read = 3600.0

[flow.solver.timeout]
rollout = 86400.0
scoring = 21600.0

[flow.rollout_gate]
name = "glm"
max_concurrent = 2000
dir = "/data/sft-gen/rollout-gates"

[flow.rollout_gate.priorities]
"run:tb-science-glm-measure-20261023" = 9000
"""


def derive(**overrides):
    source = SOURCE.encode()
    kwargs = dict(
        label="tb-sci",
        checkpoint="best",
        model="nemotron-3.5-super-tb4-sft",
        base_url="http://127.0.0.1:8000/v1",
        api_key_var="TB4_EVAL_VLLM_KEY",
        root="/work/runs/best",
        gate_name="tb4-eval-best",
        gate_dir="/work/gates",
        gate_max=384,
    )
    kwargs.update(overrides)
    return flowcfg.derive(source, hashlib.sha256(source).hexdigest(), **kwargs)


def test_flow_config_changes_only_model_endpoint_and_bookkeeping():
    config, changed = derive()
    before = tomllib.loads(SOURCE)
    assert config["flow"]["solver"]["harness"] == before["flow"]["solver"]["harness"]
    assert config["flow"]["sampling"] == before["flow"]["sampling"]
    assert config["flow"]["solver"]["timeout"] == before["flow"]["solver"]["timeout"]
    assert config["flow"]["solver"]["client"]["timeout"] == {"connect": 30.0, "read": 3600.0}
    assert config["flow"]["solver"]["client"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert config["flow"]["store"] == "/work/runs/best/tb4-eval-best-tb-sci/store"
    assert config["flow"]["rollout_gate"] == {"name": "tb4-eval-best", "max_concurrent": 384, "dir": "/work/gates"}
    assert config["flow"]["pieces"]["rollouts"]["k"] == 8 and config["flow"]["select"]["include"]["names"] == []
    assert config["flow"]["taskset"]["dataset"] == "tb-science-vendor@85df9700aaf4"
    assert all(key.startswith(flowcfg.ALLOWED_CHANGES) for key in changed)
    assert "flow.solver.client.base_url" in changed and "flow.solver.harness.command_timeout" not in changed
    # Round-trips through TOML (what flow reads).
    assert tomllib.loads(tomli_w.dumps(config)) == config


def test_flow_config_options_and_guards():
    config, changed = derive(
        dataset="tb-science-vendor@71124a84e6f3",
        k=1,
        include_names=["terminal-bench/b", "terminal-bench/a"],
        path_rewrites={"/data/sft-gen/kernels/landlock.tgz": "/work/kernels/landlock.tgz"},
    )
    assert config["flow"]["taskset"]["dataset"] == "tb-science-vendor@71124a84e6f3"
    assert config["flow"]["pieces"]["rollouts"]["k"] == 1
    assert config["flow"]["select"]["include"]["names"] == ["terminal-bench/a", "terminal-bench/b"]
    assert config["flow"]["taskset"]["task"]["prime_kernels"]["turing/k"]["bundle"] == "/work/kernels/landlock.tgz"
    assert all(key.startswith(flowcfg.ALLOWED_CHANGES) for key in changed)
    with pytest.raises(ValueError, match="empty include"):
        derive(include_names=[])
    with pytest.raises(ValueError, match="sha256"):
        flowcfg.derive(SOURCE.encode(), "0" * 64, **{k: v for k, v in derive_kwargs().items()})
    # D5: the solver's harness becomes the plugin (same config otherwise), recorded as changed keys.
    config, changed = derive(d5_exclude_newer="2026-10-24T07:23:00Z", in_box_uv_version="0.13.5")
    assert config["flow"]["solver"]["harness"] == {
        "id": "tb4-mini-swe-agent-d5",
        "command_timeout": 1800,
        "prepare_exclude_newer": "2026-10-24T07:23:00Z",
        "prepare_uv_version": "0.13.5",
    }
    assert {"flow.solver.harness.id", "flow.solver.harness.prepare_exclude_newer"} <= set(changed)
    assert all(key.startswith(flowcfg.ALLOWED_CHANGES) for key in changed)
    with pytest.raises(ValueError, match="expected 'mini_swe_agent'"):
        flowcfg.derive(
            SOURCE.replace('id = "mini_swe_agent"', 'id = "bash"').encode(),
            hashlib.sha256(SOURCE.replace('id = "mini_swe_agent"', 'id = "bash"').encode()).hexdigest(),
            d5_exclude_newer="2026-10-24T07:23:00Z",
            **derive_kwargs(),
        )
    assert flowcfg.rollouts_k(SOURCE.encode()) == 8
    assert flowcfg.kernel_bundles(SOURCE.encode()) == ["/data/sft-gen/kernels/landlock.tgz"]
    assert flowcfg.taskset_fields(SOURCE.encode())["registry_path"] == "registries/tb-science.json"


def derive_kwargs():
    return dict(
        label="tb-sci",
        checkpoint="best",
        model="m",
        base_url="u",
        api_key_var="K",
        root="/r",
        gate_name="g",
        gate_dir="/g",
        gate_max=1,
    )


# Plan


def test_checkpoints_and_rounds():
    exports = {
        "last": {"ok": True, "dir": "/run/hf-export/last", "step": 280, "verified": True, "repo_id": "o/last"},
        "best": {"ok": True, "dir": "/run/hf-export/best", "step": 80, "verified": False, "repo_id": "o/best"},
    }
    ckpts, aliases, skipped = plan.checkpoints_to_eval(["best", "last"], exports)
    assert [c.name for c in ckpts] == ["best", "last"] and aliases == {} and skipped == {}
    assert ckpts[0].repo_id is None and ckpts[1].repo_id == "o/last"  # only a verified upload is a fallback
    # 'best' is the final step: evaluated once, as 'last'.
    exports["best"].update(step=280, same_as="last")
    ckpts, aliases, _ = plan.checkpoints_to_eval(["best", "last"], exports)
    assert [c.name for c in ckpts] == ["last"] and aliases == {"best": "last"}
    # ... unless 'last' is not evaluated.
    ckpts, aliases, _ = plan.checkpoints_to_eval(["best"], exports)
    assert [c.name for c in ckpts] == ["best"] and aliases == {}
    exports["last"]["ok"] = False
    ckpts, aliases, skipped = plan.checkpoints_to_eval(["best", "last"], exports)
    assert [c.name for c in ckpts] == ["best"] and skipped == {"last": "the export failed its checks"}

    assert plan.make_rounds(["best", "last"], 4, "sequential") == [{"best": [0, 1, 2, 3]}, {"last": [0, 1, 2, 3]}]
    assert plan.make_rounds(["best", "last"], 4, "parallel") == [{"best": [0, 1], "last": [2, 3]}]
    assert plan.make_rounds(["best", "last"], 3, "parallel") == [{"best": [0, 1], "last": [2]}]
    assert plan.make_rounds(["best", "last"], 1, "parallel") == [{"best": [0]}, {"last": [0]}]
    assert plan.make_rounds(["last"], 4, "parallel") == [{"last": [0, 1, 2, 3]}]
    assert plan.make_rounds([], 4, "sequential") == []
    assert plan.round_deadline(0, 100, 10, 2) == 45 and plan.round_deadline(50, 100, 10, 1) == 90
    assert plan.round_deadline(95, 100, 10, 1) == 95
    assert plan.round_deadline(95, None, 10, 1) is None
    # The smoke's evaluate_aliases: 'best' at the final step gets a round of its own.
    exports["last"]["ok"] = True
    ckpts, aliases, _ = plan.checkpoints_to_eval(["best", "last"], exports, evaluate_aliases=True)
    assert [c.name for c in ckpts] == ["best", "last"] and aliases == {}


def test_build_plan_round_trips(tmp_path):
    cfg = SFTInRunEvalConfig(budget_hours=2)
    exports = {"last": {"ok": True, "dir": "/run/hf-export/last", "step": 7}, "best": {"ok": False}}
    p = plan.build_plan(
        cfg,
        run_dir=tmp_path,
        exports=exports,
        node=1,
        num_nodes=2,
        gpus_per_node=8,
        master_addr="node0",
        node_files=["b.safetensors"],
        master_files=["config.json"],
        now=100.0,
    )
    assert p.deadline == 100.0 + 7200 and p.work_dir == str(tmp_path / ".inrun-eval-work" / "node1")
    assert (
        plan.build_plan(
            SFTInRunEvalConfig(),
            run_dir=tmp_path,
            exports=exports,
            node=0,
            num_nodes=1,
            gpus_per_node=8,
            master_addr="x",
        ).deadline
        is None
    )  # no cap by default
    assert p.eval_dir == str(tmp_path / "eval") and p.dedupe_exports
    assert [c.name for c in p.checkpoints] == ["last"] and p.skipped == {"best": "the export failed its checks"}
    again = plan.Plan.from_json(p.to_json())
    assert again == p
    assert SFTInRunEvalConfig.model_validate(again.config) == cfg
    smoke = SFTInRunEvalConfig(eval_only=True, model_dir="/volume/models/base")
    p = plan.build_plan(smoke, run_dir=tmp_path, exports={}, node=0, num_nodes=1, gpus_per_node=8, master_addr="x")
    assert [(c.name, c.model_dir) for c in p.checkpoints] == [("base", "/volume/models/base")]
    assert p.rounds == [{"base": [0]}] and not p.dedupe_exports


# Serving


def test_inference_config_is_valid_and_matches_the_spec(tmp_path):
    serving_cfg = SFTInRunEvalConfig().serving
    raw = serving.inference_config(serving_cfg, Path("/run/hf-export/best"), tmp_path)
    config = InferenceConfig.model_validate(raw)
    assert config.router is None
    namespace = config.to_namespace()
    assert namespace.model == "/run/hf-export/best" and namespace.max_model_len == 1048576
    assert namespace.served_model_name == "nemotron-3.5-super-tb4-sft"
    assert namespace.chat_template == "/run/hf-export/best/chat_template.jinja"
    assert namespace.default_chat_template_kwargs == {"enable_thinking": True, "truncate_history_thinking": False}
    assert namespace.override_generation_config == {"temperature": 1.0, "top_p": 0.95}
    assert namespace.reasoning_parser == "nemotron_v3" and namespace.tool_call_parser == "qwen3_coder"
    assert namespace.enable_auto_tool_choice is True and namespace.mamba_cache_dtype == "float32"
    assert namespace.tensor_parallel_size == 2 and namespace.enable_expert_parallel is True
    assert namespace.enable_prefix_caching is True and namespace.seed == 0 and namespace.dtype == "bfloat16"
    assert not hasattr(namespace, "speculative_config") and not hasattr(namespace, "max_tokens")
    with pytest.raises(ValueError, match="speculative"):
        SFTInRunEvalConfig(serving={"vllm_extra": {"speculative_config": {"method": "mtp"}}})
    with pytest.raises(ValueError, match="max_new_tokens"):
        SFTInRunEvalConfig(serving={"override_generation_config": {"max_new_tokens": 131072}})


def test_engine_and_router_commands(tmp_path):
    serving_cfg = SFTInRunEvalConfig().serving
    specs = serving.engine_specs(serving_cfg, tmp_path / "inference.json", "10.0.0.5", "/app/.venv/bin/python")
    assert [s.port for s in specs] == [8100, 8101, 8102, 8103]
    assert [s.env["CUDA_VISIBLE_DEVICES"] for s in specs] == ["0,1", "2,3", "4,5", "6,7"]
    third = specs[2]
    assert third.cmd[:5] == [
        "/app/.venv/bin/python",
        "-m",
        "prime_rl.entrypoints.inference",
        "@",
        str(tmp_path / "inference.json"),
    ]
    cmd = " ".join(third.cmd)
    assert "--server.port 8102" in cmd and "--vllm.data-parallel-size 4" in cmd
    assert "--vllm.data-parallel-size-local 1" in cmd and "--vllm.api-server-count 1" in cmd
    assert f"--vllm @ {tmp_path / 'vllm-overrides-8102.json'}" in cmd
    assert third.overrides == {"data_parallel_rank": 2, "data_parallel_address": "10.0.0.5"}
    # Without expert parallelism every engine is its own replica.
    no_ep = serving_cfg.model_copy(update={"enable_expert_parallel": False})
    for spec in serving.engine_specs(no_ep, tmp_path / "i.json", "10.0.0.5", "py"):
        assert "--vllm.data-parallel-size 1" in " ".join(spec.cmd) and spec.overrides_path is None

    # The engine config with a per-engine CLI override still parses the way the inference entrypoint does.
    from pydantic_config import cli

    path = tmp_path / "inference.json"
    path.write_text(json.dumps(serving.inference_config(serving_cfg, Path("/m"), tmp_path)))
    (tmp_path / "vllm-overrides-8102.json").write_text(json.dumps(third.overrides))
    parsed = cli(InferenceConfig, args=third.cmd[3:])
    assert parsed.server.port == 8102 and parsed.vllm.data_parallel_size == 4
    assert parsed.vllm.api_server_count == 1 and parsed.vllm.data_parallel_size_local == 1
    assert parsed.vllm.model_extra["data_parallel_rank"] == 2
    assert parsed.vllm.model_extra["data_parallel_address"] == "10.0.0.5"

    router = serving.router_cmd(serving_cfg, "vllm-router", 8001, ["http://a:8100", "http://b:8100"])
    assert router[:3] == ["vllm-router", "--policy", "sticky_least_loaded"]
    joined = " ".join(router)
    assert "--port 8001" in joined and "--worker-urls http://a:8100 http://b:8100 --request-id-headers" in joined
    assert "--request-id-headers x-session-id" in joined and "--request-timeout-secs 14400" in joined
    with pytest.raises(ValueError):
        serving.router_cmd(serving_cfg, "vllm-router", 8000, [])

    env = serving.engine_env(serving_cfg, {"RANK": "3", "MASTER_ADDR": "x", "TORCHELASTIC_RUN_ID": "y", "PATH": "/b"})
    assert "RANK" not in env and "MASTER_ADDR" not in env and "TORCHELASTIC_RUN_ID" not in env
    assert env["PATH"] == "/b" and env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"


def test_check_model_dir(tmp_path):
    serving_cfg = SFTInRunEvalConfig().serving
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "s1.safetensors"}}))
    (model / "s1.safetensors").write_bytes(b"x")
    (model / "chat_template.jinja").write_text("not the GA template")
    problems = serving.check_model_dir(serving_cfg, model)
    assert len(problems) == 1 and "sha256" in problems[0]
    ok = serving_cfg.model_copy(update={"chat_template_sha256": hashlib.sha256(b"not the GA template").hexdigest()})
    assert serving.check_model_dir(ok, model) == []
    (model / "s1.safetensors").unlink()
    assert any("s1.safetensors" in p for p in serving.check_model_dir(ok, model))


# Harness


def test_harness_install_env_and_pins(tmp_path):
    h = SFTInRunEvalConfig().harness
    cmds = harness.install_commands("uv", "/py/bin/python3.13", tmp_path / "venv", tmp_path / "bundle", "offline")
    assert cmds[0] == ["uv", "venv", str(tmp_path / "venv"), "--python", "/py/bin/python3.13"]
    assert cmds[1][:6] == ["uv", "pip", "install", "--python", str(tmp_path / "venv" / "bin" / "python"), "--no-deps"]
    assert cmds[1][-3:] == ["--find-links", str(tmp_path / "bundle" / "harness" / "wheels"), "--no-index"]
    mixed = harness.install_commands("uv", "3.13", tmp_path, tmp_path, "mixed")[1]
    assert "--find-links" in mixed and "--no-index" not in mixed
    assert "--find-links" not in harness.install_commands("uv", "3.13", tmp_path, tmp_path, "online")[1]
    wheels = tmp_path / "bundle" / "harness" / "wheels"
    assert harness.wheel_mode(tmp_path / "bundle") == "online"
    wheels.mkdir(parents=True)
    (wheels / "prime_flow-0.1.1.dev354-py3-none-any.whl").write_bytes(b"")
    assert harness.wheel_mode(tmp_path / "bundle") == "mixed"
    (wheels / ".complete").write_text("")
    assert harness.wheel_mode(tmp_path / "bundle") == "offline"

    info = {
        "packages": {"verifiers": "0.4.1.dev340", "prime-flow": "0.1.1.dev354"},
        "files": dict(h.pinned_files_sha256),
    }
    assert harness.check_versions(h, info) == []
    info["files"]["verifiers/v1/harnesses/mini_swe_agent/program.py"] = "0" * 64
    info["packages"]["verifiers"] = "0.4.1.dev338"
    assert len(harness.check_versions(h, info)) == 2

    base = {"PATH": "/usr/bin", "RANK": "0", "HF_TOKEN": "secret", "UV_PROJECT_ENVIRONMENT": "/app/.venv"}
    env = harness.uv_env(base, tmp_path)
    assert "HF_TOKEN" not in env and "RANK" not in env and "UV_PROJECT_ENVIRONMENT" not in env
    assert env["HOME"] == str(tmp_path / "home")
    henv = harness.HarnessEnv(
        venv=tmp_path / "venv", cwd=tmp_path / "sft-gen", env=env, info={"plugins_dir": str(tmp_path / "plugins")}
    )
    # Only the platform's PRIME_API_KEY counts (a custom name is not forwarded by `prime train -e` anyway).
    with pytest.raises(RuntimeError, match="PRIME_API_KEY is not set"):
        harness.run_env(h, henv, {**base, "TB4_EVAL_PRIME_API_KEY": "k", "PRIME_TEAM_ID": "team"})
    with pytest.raises(RuntimeError, match="PRIME_TEAM_ID"):
        harness.run_env(h, henv, {**base, "PRIME_API_KEY": "k"})
    run = harness.run_env(h, henv, {**base, "PRIME_API_KEY": "k"}, "team-from-config")
    assert run["PRIME_TEAM_ID"] == "team-from-config"
    run = harness.run_env(h, henv, {**base, "PRIME_API_KEY": "k", "PRIME_TEAM_ID": "team"})
    assert run["PRIME_API_KEY"] == "k" and run["PRIME_TEAM_ID"] == "team" and len(run["TB4_EVAL_VLLM_KEY"]) == 32
    assert run["PATH"].startswith(str(tmp_path / "venv" / "bin")) and "HF_TOKEN" not in run
    assert run["PYTHONPATH"] == f"{tmp_path / 'sft-gen'}:{tmp_path / 'sft-gen' / 'plugins'}:{tmp_path / 'plugins'}"
    creds = harness.credentials(h, {"PRIME_API_KEY": "secret-value"}, "t")
    assert creds == {
        "prime_api_key_present": True,
        "prime_team_id": "t",
        "prime_team_id_source": "run config [env_vars]",
        "problem": None,
    }
    assert "secret-value" not in json.dumps(creds)
    assert (harness.PLUGINS / f"{harness.D5_MODULE}.py").is_file()
    bundle = tmp_path / "hb"
    (bundle / harness.HOST_UV / "0.13.1" / "x86_64-unknown-linux-musl").mkdir(parents=True)
    (bundle / harness.HOST_UV / "0.13.1" / "x86_64-unknown-linux-musl" / "uv").write_bytes(b"uv")
    assert harness.stage_host_uv(tmp_path / "none", env) is None
    cache = harness.stage_host_uv(bundle, env)
    assert (cache / "0.13.1" / "x86_64-unknown-linux-musl" / "uv").read_bytes() == b"uv"
    assert cache == tmp_path / "home" / ".cache" / "verifiers" / "uv"
    assert harness.flow_cmd(h, Path("/v"), Path("/c.toml")) == ["/v/bin/flow", "run", "@", "/c.toml"]


def test_bundle_checks(tmp_path):
    h = SFTInRunEvalConfig().harness.model_copy(update={"sources": ["tb-sci"]})
    bundle = tmp_path / "bundle"
    files = {
        "tasks.tsv": TABLE,
        "harness/requirements.txt": "verifiers==0.4.1.dev340\n",
        "sft-gen/plugins/sft_gen_task_qc.py": "",
        "sft-gen/sft_gen/task_qc.py": "",
        "flow-configs/tb-sci.toml": SOURCE,
    }
    for rel, text in files.items():
        (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
        (bundle / rel).write_text(text)
    manifest = {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in files.items()}
    (bundle / "manifest.json").write_text(json.dumps({"files": manifest}))
    problems = harness.check_bundle(h, bundle)
    assert problems == [
        "flow-configs/tb-sci.toml sha256 "
        + manifest["flow-configs/tb-sci.toml"]
        + " is not the pinned collection config"
    ]
    h = h.model_copy(update={"source_config_sha256": {"tb-sci": manifest["flow-configs/tb-sci.toml"]}})
    assert harness.check_bundle(h, bundle) == []
    (bundle / "tasks.tsv").write_text("changed")
    assert harness.check_bundle(h, bundle) == ["tasks.tsv sha256 differs from the manifest"]

    # The Harbor caches and the kernel bundle (whose collection path is not creatable here).
    cache = bundle / "harbor-cache" / "ds_1"
    cache.mkdir(parents=True)
    (cache / "task.toml").write_text("x")
    home = tmp_path / "home"
    assert harness.stage_harbor_caches(bundle, home, ["ds_1", "ds_2"]) == ["harbor-cache/ds_2 missing"]
    assert (home / ".cache" / "harbor" / "ds_1" / "task.toml").read_text() == "x"
    (bundle / "kernels").mkdir()
    (bundle / "kernels" / "landlock.tgz").write_bytes(b"k")
    target = tmp_path / "data" / "kernels" / "landlock.tgz"
    assert harness.stage_kernel_bundle(bundle, str(target), tmp_path / "work") == {}
    assert target.read_bytes() == b"k"
    blocked = tmp_path / "ro"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        rewrite = harness.stage_kernel_bundle(
            bundle,
            str(blocked / "x" / "landlock.tgz"),
            tmp_path / "work",
            run=lambda *a, **k: SimpleNamespace(returncode=1),
        )
    finally:
        blocked.chmod(0o700)
    assert rewrite == {str(blocked / "x" / "landlock.tgz"): str(tmp_path / "work" / "kernels" / "landlock.tgz")}


# Coordination and cleanup


def test_coordinator_waits():
    coord = Coordinator(MemoryStore(), poll=0)
    clock = iter(range(100))
    assert coord.wait_any(["a", "b"], deadline=3, clock=lambda: next(clock)) is None
    coord.set("b", {"x": 1})
    assert coord.wait_any(["a", "b"], deadline=10**9) == ("b", {"x": 1})
    coord.set("a", {})
    assert coord.wait_any(["a", "b"], deadline=10**9) == ("a", {})
    assert coord.wait_all(["a", "b", "c"], deadline=0) == {"a": {}, "b": {"x": 1}}
    assert coord.wait_any(["c"], deadline=10**9, abort=lambda: True) is None


def test_cleanup_keeps_one_copy_per_file(tmp_path):
    def export(name):
        directory = tmp_path / "hf-export" / name
        directory.mkdir(parents=True)
        for file in ("a.safetensors", "b.safetensors", "c.safetensors", "config.json", "export_report.json"):
            (directory / file).write_text(file)
        return str(directory)

    exports = {
        "last": {"ok": True, "dir": export("last"), "verified": False},
        "best": {"ok": True, "dir": export("best"), "verified": True},
    }
    cfg = SFTInRunEvalConfig()
    base = dict(
        run_dir=str(tmp_path),
        eval_dir=str(tmp_path / "eval"),
        num_nodes=2,
        master_addr="x",
        coord_port=1,
        started=0,
        deadline=1,
        checkpoints=[],
        rounds=[],
        exports=exports,
        node_files=["b.safetensors"],
        master_files=["config.json", "export_report.json"],
        dedupe_exports=True,
        config=cfg.model_dump(mode="json"),
    )
    work = tmp_path / "work1"
    work.mkdir()
    p = plan.Plan(node=1, work_dir=str(work), delete_pushed_exports=True, **base)
    node.cleanup(p, cfg, log=lambda m: None)
    # Node 1 keeps its upload share; the master's small files go; the verified export goes entirely.
    assert sorted(x.name for x in (tmp_path / "hf-export" / "last").iterdir()) == ["b.safetensors"]
    assert not (tmp_path / "hf-export" / "best").exists() and not work.exists()

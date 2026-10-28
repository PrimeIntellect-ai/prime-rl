# In-run TB4 eval: prerequisites

`[inrun_eval]` (`src/prime_rl/trainer/sft/inrun_eval/`) makes the TB4 SFT run evaluate its own exports. After
the HF export and push, every rank releases its GPU. Each node's `local_rank 0` then `exec`s into the node's eval
process. Every node serves vLLM 0.32.0 from the image (TP2 x DP4 with EP, 4 engines per node). Node 0 replays the
collection harness: sft-gen `flow run`, verifiers 24107a4ea, mini-swe-agent 2.4.6, the three collection runs'
`configs/attempt_2/flow.toml`. It runs 500 tasks x 8 rollouts per checkpoint, 'best' then 'last'. Results are written
to `runs/<runId>/eval/`.

Configs:
- `configs/advanced/nemotron-3-super/sft-hosted-tb4-v2.toml`: the run.
- `configs/advanced/nemotron-3-super/tb4-inrun-eval-smoke.toml`: the hardware smoke test, eval-only. It serves the
  base model on 1 node and runs 3 tasks x 1 rollout.

Nothing private goes into this repository: the branch is pushed to the public prime-rl repo. The sft-gen and flow
code, the task trees and the task names all come from the **eval bundle** on the volume.

## 1. Volume `nemotron-sft-tb4-v2`

| Path | What | Status |
|---|---|---|
| `models/nemotron-3.5-super-text` | GA text backbone (fb0ee0bc), with `chat_template.jinja` sha256 `1369e90b…c816` | copy in progress (scratchpad `tb4/copy/v2-model.log`) |
| `datasets/nemotron-distill-sft-tb4-1m-v2` | the v2 training data | to build |
| `datasets/tb4-eval/` | the eval bundle (section 2) | to stage |

The eval checks the chat template's sha256 before serving. The exports copy it from the base, so the base copy
must be the GA template.

## 2. The eval bundle: `/volume/datasets/tb4-eval`

Assemble it with `scripts/tb4_inrun_eval/stage_bundle.py`. The script checks every input and writes
`manifest.json`. At run time the eval re-checks every sha256 before using anything. Layout:

| Path | Source | Needs the QC box? |
|---|---|---|
| `flow-configs/{tb-eng,tb-eng-extra9,tb-sci}.toml` | `<run>/configs/attempt_2/flow.toml` of the three `*-glm-measure-20261023` runs (sha256 d0e5b6c7… / 86a3fba8… / 06d60eb7…, pinned in the config) | no: local copies in scratchpad `replica-inv/collection-config/box/` |
| `tasks.tsv` | `name<TAB>source<TAB>split` for all 500 tasks. The script builds it from the collection trace extract (`replica-inv/collection-config/traces.jsonl`: run per task), `data-prep/tb4-val-tasks.txt` (44 val), the v2 dataset's train split (`--train-dataset`, `task` column) and `--extra-task terminal-bench/convolution_filters_english=tb-sci` | no, once the v2 dataset is built |
| `smoke-tasks.txt` | 3 task names for the smoke. Suggested (fast and mixed outcomes under GLM, one per source): scratchpad `tb4/inrun-eval-build/smoke-tasks.suggested.txt` | no |
| `sft-gen/` | **sft-gen @ bca802f** tree: `sft_gen/`, `plugins/`, `prompts/`, `configs/`, `pyproject.toml`, `uv.lock`. The flow runs with a copy of it as cwd: the configs name `prompts/*.md` relative to it | **yes**: bca802f exists only in the box's `/data/repos/sft-gen` (`git archive bca802f`; box git needs `HOME=<dir with [safe] directory=*>`). The scratchpad `replica-inv/eval-path/box-src/sft-gen-measure` has the same `sft_gen/` and `plugins/` but no `prompts/` |
| `harness/requirements.txt` | the flow host's closure at the collection versions (below) | **yes** (bca802f's `uv.lock`) |
| `harness/wheels/` | at least the **prime-flow @ ed23e23** wheel: the flow repo is private, so the pod cannot install it from git. With `--wheels-complete`, every requirement is included and the install runs offline | build where GitHub access works |
| `harness/python/*.tar.gz` | optional: python-build-standalone 3.13 `install_only` (x86_64 linux). Without it, uv downloads Python from GitHub | no |
| `harness/frpc/` | optional: `frpc` 0.66.0 and `.frpc_version`, as prime_tunnel lays out `~/.prime/bin`. Without them, the first tunnel downloads frpc from GitHub | no |
| `harbor-cache/<name>/` | Harbor exports of `tb-engineering-vendor@1eb123da72ed`, `@37107d21ec3d` and `tb-science-vendor@71124a84e6f3` (= the 290 tasks @85df9700aaf4 plus convolution_filters_english), named as verifiers caches them (`…_8cfd9e1a5499`, `…_4a76686edb11`, `…_d4e4ed50a94a`). Build with harbor 0.24.0: `harbor download <ds> --export -o <dir> --repo PrimeIntellect-ai/nvidia-delivery@main --registry-path registries/<tb-engineering\|tb-science>.json` | no box needed, but needs read access to the private nvidia-delivery repo. The box's `~/.cache/harbor/*` allows an optional byte-for-byte check |
| `kernels/landlock-6.12.113-deb13.tgz` | the landlock bundle that the 2 tb-eng kernel tasks' configs name (`/data/sft-gen/kernels/…`). The eval copies it there (with `sudo -n` if needed), or else rewrites the path. The path is host-side only | **yes** (box `/data/sft-gen/kernels/`). Collection ran with ABI 7 and `rebooted=false`, so the bundle is never booted, but it must exist (inferred, not verified) |

`harness/requirements.txt` must be the complete closure of `verifiers[harbor]` @ 24107a4ea, `prime-flow[qc,dashboard]`
@ ed23e23 and the `prime-sandboxes @ git+…prime.git@49a99bc#subdirectory=packages/prime-sandboxes` override, at the
versions of sft-gen bca802f's `uv.lock`. It is installed with `--no-deps`, so nothing is resolved on the pod. On the
box:

```
cd <tmp> && git -C /data/repos/sft-gen archive bca802f | tar -x -C sft-gen-bca802f
uv export --project sft-gen-bca802f --frozen --no-hashes --no-emit-project --no-dev > all-pins.txt
# A tiny project with just the flow host's dependencies, constrained to those pins:
#   [project] dependencies = ["verifiers[harbor]", "prime-flow[qc,dashboard]"]
#   [tool.uv] constraint-dependencies = <all-pins.txt lines>, override-dependencies = <sft-gen's two overrides>,
#   exclude-newer / exclude-newer-package and the verifiers / prime-flow git sources = sft-gen bca802f's
uv lock && uv export --frozen --no-hashes --no-emit-project > requirements.txt
```

Then rewrite the git lines (verifiers, prime-flow, prime-sandboxes) to `name==version`. Their wheels are in
`harness/wheels/`; build them with `uv build --wheel` or `pip wheel --no-deps <git url>`. Check: the venv the eval
builds must report verifiers `0.4.1.dev340` and prime-flow `0.1.1.dev354`, plus the four agent-facing verifiers files
at their 24107a4ea sha256s (`[inrun_eval.harness] pinned_files_sha256`). Otherwise the harness setup fails and the
eval records why.

Upload the bundle directory to `datasets/tb4-eval` on `nemotron-sft-tb4-v2` (volume-to-volume copy tooling, or
`prime volumes` put), and only after you approve the write.

## 3. Secrets: `-e` env files

| File | Variables | Notes |
|---|---|---|
| `~/.config/prime-export/hf.env` (exists) | `HF_TOKEN` | the export push; also the eval's fallback download of an incomplete node copy |
| `~/.config/prime-export/tb4-eval.env` (new) | `TB4_EVAL_PRIME_API_KEY=<the Prime API key the collection used>`, `PRIME_TEAM_ID=<the team that owns the task images>` | sandboxes, tunnels and the private task images (499/499 + the 500th's are COMPLETED, PRIVATE in that team). `PRIME_API_KEY` itself is used if the platform passes it; it may be reserved, hence the second name. It reaches only the harness, as `PRIME_API_KEY`. HF/W&B/GitHub tokens are removed from the harness env |

If `prime train` takes only one `-e`, put all three variables in one file. The vLLM key the harness sends is
generated per run; the engines do not check it, and the pods have no ingress.

## 4. Smoke test (before the run depends on it)

`prime train configs/advanced/nemotron-3-super/tb4-inrun-eval-smoke.toml --volume nemotron-sft-tb4-v2 -e ~/.config/prime-export/tb4-eval.env -y`

Announce the launch and its cost before launching. Check:

1. The log shows `in-run eval: node 0 hands over to its eval process`, then `started (pid …)`: the pod outlives
   the trainer's ranks.
2. `harness ready: {...}`: the venv built with the pins. This is where egress to PyPI/GitHub and the Python download
   fail if they fail.
3. `engines ready after …s` (vLLM fits 1M context at TP2 with the float32 Mamba cache; note the start-up time),
   then `router … probe ok`.
4. `flow runs […] started`, then `pass@…` per checkpoint: Prime sandboxes, frps tunnels and the task images work
   from the pod.
5. After the sync: `runs/<id>/eval/{status.json,summary.json,base/summary.json,base/per_task.jsonl,base/configs,base/stores}`
   exist, and `runs/<id>/.inrun-eval-work/` does **not**. That confirms that files deleted before the exit are not
   uploaded.

Set `num_train_nodes = 2` to also test the cross-node part: the coordination store on `MASTER_ADDR:29650`, node
1's engines behind node 0's router on ports 8100–8103, and node 0 exiting last.

What only the real run shows:
- the trainer → exec hand-off with a trained model in memory (GPU memory actually freed: the eval waits up to 15 min
  for < 4 GiB per GPU);
- the per-node full export copy: host RAM (each rank writes a file as soon as its tensors are in, so about one
  shard per rank is held at a time) and staging disk (+~0.25 TB per node per export);
- the rate of LiteLLM 600 s timeouts (HarnessError reruns) at 24 rollouts per engine (`rollouts_per_engine`).

## 5. Cleaning up the volume: the effective way

Deleting folders on a volume afterwards does not work in practice. A read-write session first stages the whole
volume and hits the platform's 60-minute cap (3.9 TB timed out), and putting files under `runs/` is refused. The
effective lever is to keep the data off the volume in the first place. Hosted runs stage the volume per node and sync
it after the run, so a file deleted from the staging before the run ends is never uploaded. Unknown 4 in the design:
the smoke's step 5 checks this.

| Item | Size | What happens | Switch |
|---|---|---|---|
| best snapshot `best-weights/` | ~0.48 TB | deleted on every node once the 'best' export passed | `[hf_export.cleanup] best_snapshot = true` (set) |
| duplicate export copies (each node writes a full copy to serve it) | ~0.75 TB per export | after the eval each node keeps only the files it uploads; the master keeps the small files | always, with the eval |
| eval work dir (venv, flow stores, vLLM configs) | ~0.1–0.2 TB | deleted on every node before the exit; the stores go to `eval/<ckpt>/stores/*.tar.zst` | `[inrun_eval] keep_work_dir = false` (default) |
| final checkpoint `checkpoints/step_<N>` | ~1.45 TB | **kept** | `[hf_export.cleanup] final_checkpoint = true` deletes it once 'last' is verified on HF. **Your decision**: deleting it means the run cannot be resumed or trained further |
| the exports themselves `hf-export/{best,last}` | 2 x 0.25 TB | kept | `pushed_exports = true` deletes the HF-verified ones after the eval |

The old volume **`nemotron-sft-tb4`** (3.66 TB: eivaw DCP and urip VL exports) cannot be cleaned folder by folder
for the same reason. The effective way is to delete the whole volume: `prime volumes delete nemotron-sft-tb4`. Do
this once the new volume has the model and the v2 dataset; everything else on it is reproducible (the models are
copies of nemotron-training, the datasets are in the local build). **It needs your confirmation**, and nothing has
been deleted.

## 6. Defaults worth knowing

- **Budget:** `budget_hours = 48` from the hand-off. Each checkpoint gets an equal share of what is left, minus a
  60-minute reserve. At its end the flow runs get SIGTERM ("stop now") and what landed is scored. Draws that did not
  land are reported as `missing` and count as failures. `summary.json` says `complete: false` in that case.
- **Scoring:** per task and draw, the last finished row counts. pass@k is unbiased, with n = 8; errored and missing
  draws are failures, counted separately. Results are given overall, for val (44), train and other, and per source;
  diagnostics include unmapped jobs, row errors and stop conditions.
- **The only flow-config changes** (checked against an allow-list and recorded per source in `summary.json`): run
  name and dir, store, `flow.model` / `flow.solver.model`, the solver client's `base_url` / `api_key_var`, the
  rollout gate, tb-sci's dataset (@71124a84e6f3 for the 500th task), and only for the smoke `k` and the task subset.
- **Served model name** `nemotron-3.5-super-tb4-sft`: the same for both checkpoints. Mini's args and trajectory show
  it to the agent.
- **prime-rl inference defaults** stay on: fp32 lm_head and fp32 router logits (`enable_fp32_lm_head` /
  `enable_fp32_router_logits`), as in the swe-marathon evals.
- **Failure isolation:** the node processes always exit 0. A failed checkpoint is recorded in `eval/status.json`
  and `eval/summary.json`. The exports are already written, checked and pushed before the hand-off.

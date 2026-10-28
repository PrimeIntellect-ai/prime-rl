# In-run TB4 eval: prerequisites

`[inrun_eval]` (`src/prime_rl/trainer/sft/inrun_eval/`) makes the TB4 SFT run evaluate its own exports. After
the HF export and push, every rank releases its GPU. Each node's `local_rank 0` then `exec`s into the node's eval
process. Every node serves vLLM 0.32.0 from the image (TP2 x DP4 with EP, 4 engines per node). Node 0 replays the
collection harness: sft-gen `flow run`, verifiers 24107a4ea, mini-swe-agent 2.4.6, the three collection runs'
`configs/attempt_2/flow.toml`, with spec decision D5 (below). It runs 500 tasks x 8 rollouts per checkpoint, 'best'
then 'last', with no wall-clock cap. Results are written to `runs/<runId>/eval/`.

Configs:
- `configs/advanced/nemotron-3-super/sft-hosted-tb4-v2.toml`: the run.
- `configs/advanced/nemotron-3-super/tb4-inrun-eval-smoke.toml`: **the smoke of the production path**. 2 nodes,
  3 real training steps at 64k on a tiny dataset with val + keep_best (best capped at step 2), then the run's
  post-training path unchanged: the full_per_node export of best and last (no push), check_node_copies,
  cleanup_staging, the hand-off, vLLM on both nodes, the router, two rounds (best, then last) of 3 tasks x 1 rollout
  with the pinned harness, the dedupe and the cleanup.
- `configs/advanced/nemotron-3-super/tb4-inrun-eval-smoke-evalonly.toml`: the quick eval-only check (1 node, serves
  the base model, no training, no export).

Nothing private goes into this repository: the branch is pushed to the public prime-rl repo. The sft-gen and flow
code, the task trees and the task names all come from the **eval bundle** on the volume.

## 1. Volume `nemotron-sft-tb4-v2`

| Path | What | Status |
|---|---|---|
| `models/nemotron-3.5-super-text` | GA text backbone (fb0ee0bc), with `chat_template.jinja` sha256 `1369e90b…c816` | copy in progress (scratchpad `tb4/copy/v2-model.log`) |
| `datasets/nemotron-distill-sft-tb4-1m-v2` | the v2 training data | to upload |
| `datasets/nemotron-distill-sft-tb4-smoke` | the smoke's tiny dataset (section 4) | to make and upload |
| `datasets/tb4-eval/` | the eval bundle (section 2) | built locally, to upload |

The eval checks the chat template's sha256 before serving. The exports copy it from the base, so the base copy
must be the GA template.

## 2. The eval bundle: `/volume/datasets/tb4-eval`

Assemble it with `scripts/tb4_inrun_eval/stage_bundle.py`. The script checks every input and writes
`manifest.json` (version 2): every file's sha256 **and mode**. At run time the eval re-checks every sha256 before
using anything. A Prime volume keeps no file modes (its objects carry only `md5chksum` / `mtime` metadata), so the
eval restores each file's recorded mode on its node-local copies before anything runs: sft-gen, frpc (also made
executable explicitly; prime_tunnel runs it as is), the host uv, and every Harbor task file (1,771 executable in the
three caches the eval uses: `tests/test.sh`, helper scripts under `tests/`, ...). A mode that cannot be restored, or a
manifest without modes, fails the harness setup (`eval/status.json` `harness.ok = false`, `eval.status =
"harness_failed"`) before any rollout. The Harbor caches' modes and blobs equal the git trees at the registry-pinned
commits (22,263 files over the four caches, 0 mismatches). Layout and where each input comes from (all fetched from
the QC box without writing outside `/tmp/azh_*`):

| Path | Source |
|---|---|
| `flow-configs/{tb-eng,tb-eng-extra9,tb-sci}.toml` | `<run>/configs/attempt_2/flow.toml` of the three `*-glm-measure-20261023` runs (sha256 d0e5b6c7… / 86a3fba8… / 06d60eb7…, pinned in the config) |
| `tasks.tsv` | `name<TAB>source<TAB>split` for all 500 tasks, from the collection trace extract (run per task), `data-prep/tb4-val-tasks-v2.txt` (50 val: the 44 frozen tasks plus 6 zero-pass ones, the dataset's 10% validation split; sha256 `3d7fc49f…a76`), the v2 dataset's train split (`--train-dataset`, 450 tasks) and `--extra-task terminal-bench/convolution_filters_english=tb-sci`. The eval refuses a table that does not label exactly `[inrun_eval.harness] expected_val_tasks` (50) tasks val |
| `smoke-tasks.txt` | 3 task names for the smokes, one per source |
| `sft-gen/` | sft-gen **@ bca802f**: `git -C /data/repos/sft-gen archive bca802f` (box git needs `HOME=<dir with [safe] directory=*>`). Includes `prompts/` (the configs name `prompts/*.md` relative to the flow's cwd) and `uv.lock` |
| `harness/requirements.txt` | the flow host's closure: `verifiers[harbor]` + `prime-flow[qc,dashboard]` in bca802f's `uv.lock` (with its `prime-sandboxes` 49a99bc override), evaluated for linux x86_64 / CPython 3.13, exact versions, 137 packages, git sources as `name==version` |
| `harness/wheels/` + `.complete` | every requirement as a wheel, so the pod installs offline (`--no-index`). verifiers 0.4.1.dev340, prime-flow 0.1.1.dev354 and prime-sandboxes 0.4.3 are built with `uv build --wheel` from `git clone --shared` checkouts of the box's `/data/repos/{verifiers-runtime-fixes,flow-wait-sources,prime-sdk-fixes}` at 24107a4ea / ed23e23 / 49a99bc (hatch-vcs gives the lock's versions); the other 134 are `pip download --no-deps --only-binary=:all:` for cp313 manylinux x86_64 |
| `harness/python/*.tar.gz` | CPython 3.13.16 (python-build-standalone, the build uv 0.13.1 installs for `3.13`, the uv the collection venv was made with), as `python/…` |
| `harness/frpc/` | `frpc` 0.66.0 and `.frpc_version`, downloaded and checksum-verified by prime_tunnel 0.1.14's own `get_frpc_path()` |
| `harness/host-uv/0.13.1/x86_64-unknown-linux-musl/uv` | verifiers' host uv (uploaded to boxes that cannot install uv), fetched and sha-checked by 24107a4ea's own `host_uv.host_uv_binary()`; the eval puts it in that function's cache dir |
| `harbor-cache/<name>/` | Harbor 0.24.0 `download <ds> --export --repo PrimeIntellect-ai/nvidia-delivery@main --registry-path registries/<x>.json` of `tb-engineering-vendor@1eb123da72ed`, `@37107d21ec3d`, `tb-science-vendor@71124a84e6f3` (and `@85df9700aaf4`, the collection pin, for `tb_sci_dataset = None`), named as verifiers caches them. Run on the box with git's `url.insteadOf` pointing the repo at the box's checkout `/data/benchmarks/nvidia-delivery` (same objects); every exported file is checked against the git blob (sha1) and mode of the registry's pinned commit and path |
| `kernels/landlock-6.12.113-deb13.tgz` | `/data/sft-gen/kernels/` on the box, sha256 `f8be727f…ee155` |

The pod's install is `uv venv` with the staged Python, then `uv pip install --no-deps --no-index --find-links
harness/wheels -r harness/requirements.txt`; `uv pip check` reports one known conflict, verifiers' `prime-sandboxes
>=0.5.0` against the 0.4.3 override, which is exactly the collection's environment. The eval refuses to start unless
it reports verifiers `0.4.1.dev340`, prime-flow `0.1.1.dev354`, and the five agent-facing verifiers files
(`mini_swe_agent/{harness,program}.py`, `runtimes/{runtime,base,host_uv}.py`) at their 24107a4ea sha256s
(`[inrun_eval.harness] pinned_files_sha256`), and unless verifiers resolves the D5 harness plugin.

Upload the bundle directory to `datasets/tb4-eval` on `nemotron-sft-tb4-v2`, only after you approve the write.

## 3. Credentials

`prime train -e` forwards only `HF_TOKEN` and `WANDB_API_KEY` for SFT runs and exits 1 on any other key, so the
eval cannot be given its own Prime key. It uses:

| Variable | From | Notes |
|---|---|---|
| `HF_TOKEN` | `-e ~/.config/prime-export/hf.env` (the run only) | the export push; also the eval's fallback download of an incomplete node copy |
| `PRIME_API_KEY` | the platform: it creates a per-run token and wires it into the run's pods as `PRIME_API_KEY` (k8s `secretKeyRef`; confirmed for the RL orchestrator, **unconfirmed for SFT trainer pods**) | sandboxes, tunnels, the team's private task images. Only its presence is logged (`in-run eval: node 0: PRIME_API_KEY present: yes/NO`) and recorded (`eval/status.json` `credentials`). Without it the eval phase ends at once with `eval/status.json` `eval.status = "no_credentials"` and every checkpoint `failed` / `reason: no_credentials`; nothing is served. With it, before vLLM loads, a probe lists one of the team's sandboxes and one tunnel with the key (`GET /api/v1/sandbox?team_id=…`, `GET /api/v1/tunnel?teamId=…`): 401 ends the eval phase as `no_credentials`, 402 / 403 (e.g. "You are not a member of this team") as `forbidden`; a network error or 5xx that persists after 6 tries is recorded as `inconclusive` and the eval goes on. The result is in `eval/status.json` `credentials.probe`; the key is never logged (`[inrun_eval.harness] credential_probe = false` skips it) |
| `PRIME_TEAM_ID` | `[env_vars] PRIME_TEAM_ID = "cmlr3u2er002zhr01tj8f48ts"` in the config (not a secret); the hand-off also reads it from the config if the pods' environment lacks it | the team owning the 499/499 + the 500th task's images (COMPLETED, PRIVATE) |

The vLLM key the harness sends is generated per run; the engines do not check it, and the pods have no ingress. The
vLLM engines and the router get none of the pod's credentials (`PRIME_API_KEY`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
`WANDB_API_KEY`, `GITHUB_TOKEN`, `GH_TOKEN` are stripped from their environment); the harness gets only
`PRIME_API_KEY`.

Launch commands: the run `prime train configs/advanced/nemotron-3-super/sft-hosted-tb4-v2.toml --volume
nemotron-sft-tb4-v2 -e ~/.config/prime-export/hf.env -y`; the smokes without `-e` (nothing is pushed).

The probe shows that the per-run token is accepted for the team's sandboxes and tunnels. Whether it may pull the
team's private images shows only in the first rollouts.

## 4. Smoke test (before the run depends on it)

Make and upload its dataset first (whole rows of at most 65536 tokens, so packing never truncates; private content,
volume only):

```
python scripts/tb4_inrun_eval/make_smoke_dataset.py --src <local nemotron-distill-sft-tb4-1m-v2> \
    --out <dir>/nemotron-distill-sft-tb4-smoke          # 24 train / 4 val rows, manifest.json lists them
# upload <dir>/nemotron-distill-sft-tb4-smoke to datasets/nemotron-distill-sft-tb4-smoke (after approval)
```

`prime train configs/advanced/nemotron-3-super/tb4-inrun-eval-smoke.toml --volume nemotron-sft-tb4-v2 -y`

Announce the launch and its cost before launching (16 H200, roughly 3-5 h). Check, in the run's log and then on the
volume after the sync:

1. Training: 3 steps, `val/loss` after each, `Best checkpoint` at step 1 or 2 and step 3 recorded as "not a
   candidate". The best snapshot is written by every rank.
2. Export: `[last]` from memory, `[best]` after reloading the snapshot; no `export check FAILED` (that check
   includes `check_node_copies`: every node's copy of every file has the same sha256), `export_report.json` with
   `full_copy_per_node: true` and no `problems`; `Deleted …/best-weights from this node's staging` on both nodes.
3. Hand-off: `PRIME_API_KEY present: yes`, `PRIME_TEAM_ID: cmlr3u2er002zhr01tj8f48ts (environment)`, then
   `node <n> hands over to its eval process` on both nodes and `started (pid …), safety cap …`.
4. Eval: `harness ready` (offline install, pins, D5 plugin resolved), `engines ready after …s` on both nodes (vLLM
   fits 1M context at TP2; note the start-up time), the router over 8 engines and `probe ok`, `flow runs […]
   started` and `pass@…` for `best`, then the engines stop and restart for `last` and its `pass@…`.
5. After the sync: `runs/<id>/eval/{status.json,summary.json,best/,last/}` with `summary.json`, `per_task.jsonl`,
   `configs/`, `stores/`; `runs/<id>/hf-export/{best,last}` hold each file once (the dedupe); `runs/<id>/best-weights/`
   and `runs/<id>/.inrun-eval-work/` do **not** exist (files deleted before the exit are not uploaded).

What the smoke leaves on the volume: the two exports (2 x ~0.25 TB) and the eval results; no full checkpoint (no
`[ckpt]`). With the run's peak (~3.9 TB) and the model that still fits the 5 TiB volume.

What the smoke cannot show: host RAM and staging disk at the run's 1M sequence length and 4 nodes, the rate of
LiteLLM 600 s timeouts at the full gate (24 rollouts per engine), and a 500-task round's length. The eval-only smoke
(`tb4-inrun-eval-smoke-evalonly.toml`, 1 node, no `-e`) checks serving, credentials and the harness in less time.

## 5. Spec decision D5 (exclude-newer) and the in-box uv

mini-swe-agent runs from a PEP 723 uv script (`program.py`) that each sandbox bootstraps with `uv sync --script`,
without a lockfile: its ~126 packages would resolve against PyPI at eval time. D5 resolves them as of
`2026-10-24T07:23:00Z` (the attempt_2 launch):

- The derived flow configs set `[flow.solver.harness] id = "tb4-mini-swe-agent-d5"` with `prepare_exclude_newer` and
  `prepare_uv_version` (allow-listed and recorded as changed keys). The plugin (`inrun_eval/plugins/
  tb4_mini_swe_agent_d5.py`, put on the flow runs' PYTHONPATH) subclasses verifiers 24107a4ea's
  `MiniSWEAgentHarness` and adds `UV_EXCLUDE_NEWER` only to the environment passed to `prepare_uv_script`, i.e. to
  the bootstrap command. The program runs with the collection's environment (`resolved_env` + `MSWEA_*`);
  `UV_EXCLUDE_NEWER` never reaches the agent's commands.
- `program.py` and the four other pinned verifiers files are unmodified (sha-checked), so the script is still
  `2cc22997….py` and the venv the agent sees keeps its path: uv names a script environment `<file stem>-<digest of
  the script path>`. Checked locally with uv 0.13.5: the same script synced with and without `UV_EXCLUDE_NEWER`
  lands in the same `environments-v2/2cc22997…-<digest>` directory.
- uv in the box: images with a script-capable uv use it, as in collection. Otherwise verifiers' bootstrap installs
  the latest uv (`pip install -U --user uv`, else `_INSTALL_CURL`, i.e. apt-get / apk a curl when the image has no
  downloader, then astral's installer); collection boxes reported uv 0.13.5 (94 rows; 0.13.6 came out 2026-10-27).
  The plugin installs `uv==0.13.5` the same way first, with the same steps in the same order and for any user (all
  500 boxes run as root): pip, else `_INSTALL_CURL` inline and `https://astral.sh/uv/0.13.5/install.sh`; a non-root
  user also gets the root-run `_INSTALL_CURL` first, as `prepare_uv_script` does. The bootstrap then finds it. Checked
  in amd64 containers as root: debian:bookworm-slim with python3 but no pip / curl / wget (the case of ~21 tasks)
  gets curl and uv 0.13.5, verifiers' own `_ENSURE_UV` then changes nothing, and the synced venv is
  `environments-v2/2cc22997…-fd57668a0b15dabf` with `uv = 0.13.5` in its `pyvenv.cfg`. A box where neither works
  falls through to verifiers' unchanged path (the host's uv 0.13.1, as in collection).
- Only host-side error messages name the harness differently. `d5_exclude_newer = None` restores the collection's
  `mini_swe_agent` harness unchanged.

## 6. Cleaning up the volume: the effective way

Deleting folders on a volume afterwards does not work in practice. A read-write session first stages the whole
volume and hits the platform's 60-minute cap (3.9 TB timed out), and putting files under `runs/` is refused. The
effective lever is to keep the data off the volume in the first place. Hosted runs stage the volume per node and sync
it after the run, so a file deleted from the staging before the run ends is never uploaded (the smoke's step 5
checks this).

| Item | Size | What happens | Switch |
|---|---|---|---|
| best snapshot `best-weights/` | ~0.48 TB | deleted on every node once the 'best' export passed | `[hf_export.cleanup] best_snapshot = true` (set) |
| duplicate export copies (each node writes a full copy to serve it) | sync traffic, not volume space (all nodes write the same paths) | after the eval each node keeps only the files it uploads; the master keeps the small files | always, with the eval |
| eval work dir (venv, flow stores, vLLM configs) | ~0.1–0.2 TB | deleted on every node before the exit; the stores go to `eval/<ckpt>/stores/*.tar.zst` | `[inrun_eval] keep_work_dir = false` (default) |
| final checkpoint `checkpoints/step_<N>` | ~1.45 TB | **kept** | `[hf_export.cleanup] final_checkpoint = true` deletes it once 'last' is verified on HF. **Your decision**: deleting it means the run cannot be resumed or trained further |
| the exports themselves `hf-export/{best,last}` | 2 x 0.25 TB | kept | `pushed_exports = true` deletes the HF-verified ones after the eval |

The old volume **`nemotron-sft-tb4`** (3.66 TB) cannot be cleaned folder by folder for the same reason. The
effective way is to delete the whole volume (`prime volumes delete nemotron-sft-tb4`) once the new volume has the
model and the v2 dataset. **It needs your confirmation**; nothing has been deleted.

## 7. Defaults worth knowing

- **No wall-clock cap** (`budget_hours` unset): a round lasts until its flow runs exit, under the collection's own
  limits (rollout 86400 s, scoring up to 21600 s per task, stage 1209600 s). The hosted platform's own maximum run
  length is unknown. An optional `budget_hours` safety cap splits what is left equally between rounds; when it
  fires, the flow runs get SIGTERM and the draws still running then (prime-flow's call record `open`, or ended at or
  after the stop) or never started are reported as `cut`: excluded from pass@k, which is then over each task's
  remaining draws (`pass@<k>_tasks` per k), with `coverage` in `summary.json` and `complete: false`. A draw that had
  already errored before the stop stays an error. The smokes set a cap; the run does not.
- **Liveness:** node 0 refreshes `eval/status.json` `heartbeat` every 5 min: `time`, `phase`, `rollouts_done` /
  `rows_landed` / `rollouts_planned` (in all and per checkpoint), `last_progress` and `seconds_since_progress`. It
  only reports; nothing is stopped for lack of progress.
- **Scoring:** per task and draw, the last finished row counts. pass@k is unbiased, with n = 8; errored and missing
  draws of a run that ended on its own are failures, counted separately. Results are given overall, for val (50,
  `tb4-val-tasks-v2.txt`), train (450) and other, and per source (`split_tasks` and the `tasks.tsv` sha256 are in
  each `summary.json`); diagnostics include unmapped jobs, row errors and stop conditions.
- **The only flow-config changes** (checked against an allow-list and recorded per source in `summary.json`): run
  name and dir, store, `flow.model` / `flow.solver.model`, the solver client's `base_url` / `api_key_var`, the
  rollout gate, tb-sci's dataset (@71124a84e6f3 for the 500th task), the D5 harness id and its two settings, and
  only for the smokes `k` and the task subset.
- **Served model name** `nemotron-3.5-super-tb4-sft`: the same for both checkpoints. Mini's args and trajectory show
  it to the agent.
- **prime-rl inference defaults** stay on: fp32 lm_head and fp32 router logits (`enable_fp32_lm_head` /
  `enable_fp32_router_logits`), as in the swe-marathon evals.
- **Failure isolation:** the node processes always exit 0. A failed checkpoint is recorded in `eval/status.json`
  and `eval/summary.json`. The exports are already written, checked and pushed before the hand-off.

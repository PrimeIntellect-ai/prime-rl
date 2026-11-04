# prebuilt-harbor

A verifiers v1 taskset (id `prebuilt-harbor`) for Harbor tasks that grade in a separate
verifier box (`[verifier].environment_mode = "separate"`) and whose agent and verifier
images are already built and pushed. Task dirs load from a local directory or from a
task archive in a (private) Hugging Face dataset repo, instead of Harbor Hub.

After the agent finishes, its declared artifacts (mostly `/app`) are collected and
restored into a fresh verifier box running the task's verifier image, which runs
`tests/test.sh`. The taskset also closes the gaps between a Prime sandbox and Harbor's
Docker containers that change rewards: it shows the image's `/run`, runs `test.sh` in a
private PID namespace (and, for tasks without internet, a loopback-only network
namespace), moves artifacts the way Harbor's `docker compose cp` does, and reads only
`reward.txt`.

## Task sources

Exactly one of:

- **local**: `tasks_dir` (or `$PREBUILT_HARBOR_TASKS_DIR`), a directory of task dirs.
- **Hugging Face**: `hf_repo`, a dataset repo holding a `.tar.gz` archive whose root has
  `tasks/<name>/` and, optionally, `overrides.json`. The archive is downloaded with
  `huggingface_hub.hf_hub_download` using the token in `$HF_TOKEN` (`hf_token_env`),
  checked against `hf_sha256`, and extracted once per machine under a cross-process lock;
  later loads reuse the extracted tree (with a pinned commit, without the network or the
  token). Unless `$HF_XET_CACHE` is set, hf_xet's cache and logs go to `cache_dir/xet`
  rather than under `$HF_HOME`, since its logs name the repo.

| field | default | |
|---|---|---|
| `tasks_dir` | `$PREBUILT_HARBOR_TASKS_DIR` | local directory of task dirs |
| `overrides` | `overrides.json` beside the task dirs' directory, if present | see below |
| `hf_repo` | unset | dataset repo id, `owner/name` |
| `hf_revision` | `main` | pin a commit sha |
| `hf_filename` | `tasks.tar.gz` | archive path in the repo |
| `hf_sha256` | unset | expected archive sha256 |
| `hf_token_env` | `HF_TOKEN` | variable holding the token; `null` for anonymous downloads |
| `cache_dir` | `$PREBUILT_HARBOR_CACHE_DIR`, else `~/.cache/prebuilt-harbor` if writable, else `<tmp>/prebuilt-harbor-<uid>` | |

## Images and boxes

| field | default | |
|---|---|---|
| `registry` | required | image prefix, e.g. `prime/<team-slug>` |
| `tag` | `v1` | |
| `image_template` | `{registry}/{task}:{tag}` | agent image |
| `verifier_image_template` | `{registry}/{task}-verifier:{tag}` | verifier image (built from `tests/`, contains `/tests`) |
| `agent_user` | `agent` | user the agent runs as when the task declares no `[agent].user`; `null` means the runtime default (root) |
| `include_groups` | `[]` | optional hold-back groups to load anyway |
| `ignore_dockerfile` | `true` | |
| `artifact_max_bytes` | 2 GiB | |
| `min_disk_gb` | 16 | disk floor for both boxes; `null` keeps each task's `storage_mb` |
| `verifier_cpus` | 2 | vCPU floor for the verifier box; `null` keeps the task's `cpus` |

`tasks` narrows the selection (an allowlist); a name that is not a task dir raises.
Held-back tasks stay out even when named in `tasks`. A task's `key` leaves out
`task_dir`, so it does not change with where the tasks are kept.

## Overrides

Everything specific to one dataset lives in its overrides file, not in this package
(schema in `prebuilt_harbor/overrides.py`):

```json
{
  "version": 1,
  "holdback_groups": {
    "<group>": {"optional": true, "description": "...", "tasks": {"<task>": "<reason>"}}
  },
  "agent_users": {"<task>": "<user>"},
  "gold_as_root": {"<task>": "<reason>"},
  "home_artifact_excludes": {"<task>": {"<artifact source>": ["<exclude>"]}}
}
```

A task in a hold-back group does not load unless the group is `optional` and named in
`include_groups`. Every task the file names must be a task dir. Without an overrides file
nothing is held back and the load logs a warning: in local mode, pass `overrides` unless
the file sits beside the task dirs' directory.

## Which files a task dir needs

Loading reads `task.toml` and `instruction.md`, and checks for `tests/Dockerfile` (which
marks the verifier image as the one holding the tests) and `environment/Dockerfile`.
Since `verifier_image` is always set, rollouts never upload `tests/` or `environment/`.
Only the gold check (`vf-validate`) reads `solution/`.

## Gold check

`vf-validate prebuilt-harbor` runs `solution/solve.sh` as the agent user (as root for the
overrides' `gold_as_root` tasks, or for every task with
`PREBUILT_HARBOR_GOLD_AS_ROOT=1`) and grades it in a fresh verifier box; it must score 1.

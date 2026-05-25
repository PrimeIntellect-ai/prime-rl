# Prime-RL setup_kaola.sh — Visual Diagrams

## 1. Sync Function Architecture

```
┌─────────────────────────────────────────────────────┐
│         setup_s3_sync() [Called at pod startup]     │
└─────────────────────────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │ FAST_MODE=true?│
                    └───────┬────────┘
                        yes │  no
                       ┌────┴────────┐
                       ▼             ▼
                  RETURN         CONTINUE
                                    │
              ┌─────────────────────▼──────────────────┐
              │  Define inner sync_all() function      │
              │  ├─ aws s3 sync checkpoints            │
              │  ├─ aws s3 sync output                 │
              │  └─ log to s3_sync.log                 │
              └─────────────────────┬──────────────────┘
                                    │
              ┌─────────────────────▼──────────────────┐
              │  Start background loop in subshell     │
              │  (while true; do sleep 300; done) &    │
              └─────────────────────┬──────────────────┘
                                    │
              ┌─────────────────────▼──────────────────┐
              │  Capture PID: SYNC_PID=$!              │
              │  (used later for cleanup)              │
              └─────────────────────┬──────────────────┘
                                    │
              ┌─────────────────────▼──────────────────┐
              │  Register EXIT trap for cleanup:       │
              │  ├─ kill SYNC_PID                      │
              │  ├─ kill OPTIX_PID                     │
              │  ├─ call _blendergym_cleanup()         │
              │  └─ run sync_all() one last time       │
              └─────────────────────┬──────────────────┘
                                    │
                            RETURN to caller
                         (background loop running)
```

## 2. Background Sync Timeline

```
Timeline (each execution):
│
├─ T+0s     : pod startup
│   └─ setup_s3_sync() starts background loop
│   └─ SYNC_PID = <process ID>
│   └─ sync_all() runs once immediately?  NO, waits first
│
├─ T+300s   : first sync
│   └─ aws s3 sync checkpoints/
│   └─ aws s3 sync output/
│   └─ log >> s3_sync.log
│   └─ next iteration: sleep 300
│
├─ T+600s   : second sync
│   └─ aws s3 sync checkpoints/
│   └─ aws s3 sync output/
│   └─ (new or modified files only)
│
├─ T+900s   : third sync
│   └─ aws s3 sync checkpoints/
│   └─ aws s3 sync output/
│
└─ T+EXIT   : pod shutdown or error
    └─ trap EXIT fires
        ├─ kill SYNC_PID (background loop stops)
        ├─ kill OPTIX_PID (if still running)
        ├─ _blendergym_cleanup() (kill service processes)
        └─ sync_all() (final sync)
            └─ ensure all pending data uploaded before exit
```

## 3. Dual-Path Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Pod (Container)                  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    LOCAL NVMe SSD (Fast I/O)                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  /local-ssd/checkpoints/blendergym-9b-dp6/         CKPT_LOCAL
│      └─ step_0001/
│      └─ step_0002/  ← DCP checkpoints written here  │
│      └─ ...         │                               │
│                     │ [every 5 min]                │
│  /local-ssd/prime-rl-output/              OUTPUT_LOCAL
│      ├─ logs/       │                               │
│      ├─ blendergym-work/                           │
│      └─ rollouts/   ├─ aws s3 sync ────────────┐   │
│                     │  (via S3 API)            │   │
│  /local-ssd/hf_cache/                           │   │
│      └─ hub/        ← HuggingFace cache read here  │
│                                                   │
└──────────────────────────────────────────────────────────────┘
         │                                    ▲
         │                                    │
         │ [once at startup]            [every 5 min]
         │ cat tar | tar xf -           aws s3 sync
         │                                  │
         ▼                                   │
┌──────────────────────────────────────────────────────────────┐
│            FUSE Mount (Kernel FS): /threed-code/            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  /threed-code/ericzyma/experiments/blendergym-9b-dp6/       │
│      ├─ checkpoints/                      CKPT_S3            │
│      │   ├─ step_0001/                                      │
│      │   ├─ step_0002/                                      │
│      │   └─ ...                                             │
│      └─ output/                           OUTPUT_S3          │
│          ├─ logs/                                           │
│          ├─ blendergym-work/                               │
│          └─ rollouts/                                      │
│                                                              │
│  /threed-code/ericzyma/tools/                               │
│      └─ hf_cache_qwen3.5-9b.tar        [used at startup]   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
         │                                    ▲
         │ [fast sequential read]        [PUT operations]
         │ FUSE doesn't support:          │
         │ ├─ rename()                    │
         │ ├─ atomic mtime update         │
         │ └─ overwrite safety            │
         │                                 │
         ▼                                 │
┌──────────────────────────────────────────────────────────────┐
│              S3 Bucket (Persistent Storage)                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  s3://arcwm-code-us-west-2/ericzyma/experiments/            │
│  blendergym-9b-dp6/                                          │
│      ├─ checkpoints/               CKPT_S3_BUCKET            │
│      │   ├─ step_0001/                                      │
│      │   ├─ step_0002/                                      │
│      │   └─ ...                                             │
│      └─ output/                    OUTPUT_S3_BUCKET          │
│          ├─ logs/                                           │
│          ├─ blendergym-work/                               │
│          └─ rollouts/                                      │
│                                                              │
│  s3://arcwm-code-us-west-2/ericzyma/tools/                  │
│      └─ hf_cache_qwen3.5-9b.tar                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Legend:
  ─────────► Reads (startup)
  ──────────► Writes (every 5 min, then at exit)
```

## 4. Submission Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Local Mac Shell (your computer)                            │
│                                                             │
│  $ koala submit -m normal -g 8 --sync-code .:/data/... \  │
│      -c "export HF_TOKEN=$HF_TOKEN && ... && \            │
│          . scripts/setup_kaola.sh --env blendergym && \   │
│          uv run rl @ configs/..."                          │
│                                                             │
│  • $HF_TOKEN ← expanded here to actual value              │
│  • $WANDB_API_KEY ← expanded here to actual value         │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Submit to KAOLA
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  KAOLA Cluster: Pod starts, one /bin/zsh process           │
│                                                             │
│  $ [inside pod]                                            │
│  $ export HF_TOKEN=hf_abc123xyz...  ← actual value        │
│  $ export WANDB_API_KEY=xyz123...   ← actual value        │
│  $ export EXP_NAME=blendergym-9b-dp6                       │
│  $ cd /data/work/prime-rl                                  │
│  $ . scripts/setup_kaola.sh --env blendergym              │
│    │                                                       │
│    ├─ [1/7] setup_python_deps                             │
│    │    └─ uv sync --locked --extra flash-attn            │
│    │                                                       │
│    ├─ [2-5/7] env_setup()  ← from envs/blendergym.sh     │
│    │    ├─ apt install libegl1                            │
│    │    ├─ tar xf blender-4.2.0-linux-x64.tar             │
│    │    ├─ tar xf blendergym.tar                          │
│    │    ├─ uv pip install blendergym                      │
│    │    ├─ OPTIX warmup (background) ─────────┐          │
│    │    │   PID stored in $OPTIX_PID          │          │
│    │    └─ uv run python -m blendergym.services.launcher  │
│    │        PID stored in $SVC_PIDS                       │
│    │        (for cleanup in _blendergym_cleanup)          │
│    │                                                       │
│    ├─ [6/7] setup_hf_cache                                │
│    │    └─ tar xf hf_cache_qwen3.5-9b.tar                 │
│    │        (parallel with OPTIX above)                   │
│    │                                                       │
│    ├─ [7/7] setup_s3_sync                                 │
│    │    ├─ define sync_all()                              │
│    │    ├─ (while true; sleep 300; sync_all) &           │
│    │    ├─ SYNC_PID=$!                                    │
│    │    └─ trap EXIT                                      │
│    │                                                       │
│    ├─ wait $OPTIX_PID  ◄──────────────────────────┘      │
│    │    (wait for background warmup to complete)         │
│    │                                                       │
│    └─ return to caller                                    │
│                                                            │
│  $ uv run rl @ configs/multimodal/rl_blendergym_kaola.toml
│    │                                                       │
│    ├─ Training loop starts                                │
│    │   └─ 6 Blender workers spawn                         │
│    │   └─ Render using OPTIX cache                        │
│    │   └─ Write checkpoints to /local-ssd/checkpoints/   │
│    │   └─ Write output to /local-ssd/prime-rl-output/    │
│    │                                                       │
│    ├─ [Background] S3 sync every 5 min                   │
│    │   └─ aws s3 sync checkpoints/ → S3 API              │
│    │   └─ aws s3 sync output/ → S3 API                   │
│    │                                                       │
│    └─ Training continues...                               │
│       (days of computation)                               │
│                                                            │
│  [Pod exits or error]                                     │
│                                                            │
│  trap EXIT fires:                                         │
│  ├─ kill $SYNC_PID                                        │
│  ├─ kill $OPTIX_PID  (if still running)                  │
│  ├─ _blendergym_cleanup()                                 │
│  │   └─ kill $SVC_PIDS                                   │
│  └─ sync_all()  ← final sync before exit                 │
│      └─ aws s3 sync checkpoints/                         │
│      └─ aws s3 sync output/                              │
│         (ensures no data loss)                            │
│                                                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Pod exits
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  S3 Bucket (Persistent)                                    │
│                                                             │
│  ✓ All checkpoints uploaded                                │
│  ✓ All output (logs, renders) uploaded                     │
│  ✓ Training artifacts safe                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 5. Plugin Loading Sequence

```
setup_kaola.sh [main script]
         │
         ├─ Validate EXP_NAME
         ├─ Set S3 paths (dual: FUSE + S3 API)
         ├─ Parse --env, --fast, --resume flags
         │
         ├─ Check if output already exists (using FUSE path)
         │    └─ If yes and not --resume, abort
         │
         ├─ Define shared functions:
         │    ├─ setup_python_deps()
         │    ├─ setup_hf_cache()
         │    └─ setup_s3_sync()
         │
         ├─ [1/7] Call setup_python_deps()
         │    └─ uv sync --locked --extra flash-attn
         │
         ├─ [2-5/7] Load env plugin and call env_setup()
         │    │
         │    ├─ ENV_SCRIPT = "/data/work/prime-rl/scripts/envs/${ENV_NAME}.sh"
         │    ├─ if not found: ERROR
         │    │
         │    ├─ source "${ENV_SCRIPT}"
         │    │   └─ Plugin can now access:
         │    │       • $S3_PREFIX
         │    │       • $FAST_MODE
         │    │       • $RESUME_MODE
         │    │       • $PROJECT_DIR
         │    │       • $OUTPUT_LOCAL
         │    │       • $CKPT_LOCAL
         │    │       • $EXP_NAME
         │    │
         │    ├─ call env_setup()  ← plugin-specific setup
         │    │
         │    └─ Plugin registers cleanup function (optional):
         │        └─ _${ENV_NAME}_cleanup()
         │           used in setup_s3_sync's EXIT trap
         │
         ├─ [6/7] Call setup_hf_cache()
         │    └─ tar xf hf_cache_${HF_MODEL_SHORT}.tar
         │       (can run parallel with OPTIX warmup above)
         │
         ├─ [7/7] Call setup_s3_sync()
         │    ├─ Start background loop
         │    └─ Register EXIT trap
         │        └─ cleanup PIDs + final sync
         │
         ├─ wait for OPTIX_PID if set
         │    └─ Ensures GPU is free for training
         │
         └─ Return to caller (exports all set)

Then training command runs with all exports intact:
  $ uv run rl @ configs/...
    • Uses $EXP_NAME, $HF_TOKEN, $WANDB_API_KEY
    • Background S3 sync running
    • EXIT trap active
```

## 6. Error Handling & Cleanup

```
During normal operation:
┌─────────────────────────┐
│  Training running       │
│  - 6 Blender workers    │
│  - S3 sync every 5 min  │
│  - Services running     │
└─────────────────────────┘
         ▲
         │ Everything OK
         │

Error or interrupt:
┌─────────────────────────┐
│  Training error or      │
│  pod killed/timeout     │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  EXIT trap fires (even if not clean exit)          │
│                                                     │
│  1. kill $SYNC_PID 2>/dev/null || true            │
│     └─ Stop background sync loop                   │
│                                                     │
│  2. [ -n "${OPTIX_PID:-}" ] && kill ... || true   │
│     └─ Kill OPTIX if still compiling              │
│                                                     │
│  3. type _${ENV_NAME}_cleanup &>/dev/null && ... │
│     └─ Call plugin cleanup if defined             │
│     └─ For blendergym: kills services             │
│                                                     │
│  4. sync_all                                        │
│     └─ Final sync to S3                            │
│     └─ Ensures no data loss                        │
│                                                     │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Pod exits cleanly      │
│  (all data in S3)       │
└─────────────────────────┘
```


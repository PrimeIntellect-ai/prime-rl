# Prime-RL setup_kaola.sh — Quick Reference

## 1. Sync Function (Full Code)

```bash
setup_s3_sync() {
    if [ "$FAST_MODE" = true ]; then
        echo "  Background sync: SKIPPED (fast mode)"
        return
    fi
    echo "  Starting background S3 sync..."
    
    # Inner function: does the actual syncing
    sync_all() {
        local _sync_log="${OUTPUT_LOCAL}/logs/s3_sync.log"
        mkdir -p "$(dirname "${_sync_log}")"
        
        # Sync checkpoints
        if [ -d "${CKPT_LOCAL}" ]; then
            aws s3 sync "${CKPT_LOCAL}/" "${CKPT_S3_BUCKET}/" \
                --delete --quiet >> "${_sync_log}" 2>&1 || true
        fi
        
        # Sync output (exclude broadcasts/ and *.bin)
        if [ -d "${OUTPUT_LOCAL}" ]; then
            aws s3 sync "${OUTPUT_LOCAL}/" "${OUTPUT_S3_BUCKET}/" \
                --delete --exclude 'broadcasts/*' --exclude '*.bin' \
                --quiet >> "${_sync_log}" 2>&1 || true
        fi
    }
    
    # Start background loop: every 5 minutes
    (while true; do sleep 300; sync_all; done) &
    SYNC_PID=$!
    
    echo "    PID: ${SYNC_PID} (every 5 min, via S3 API)"
    
    # EXIT trap: graceful cleanup
    trap "kill ${SYNC_PID} 2>/dev/null || true; \
          [ -n \"\${OPTIX_PID:-}\" ] && kill \"\${OPTIX_PID}\" 2>/dev/null || true; \
          type _blendergym_cleanup &>/dev/null && _blendergym_cleanup; \
          sync_all" EXIT
}
```

**Key features:**
- Inner `sync_all()` function can be called standalone or in loop
- Background loop uses subshell `(...)` to isolate it
- Every 5 minutes (300 seconds) the sync runs
- EXIT trap ensures final sync before pod exits
- Uses S3 API (`s3://`), not FUSE mount (`/threed-code/`)

---

## 2. Path Variable Structure

### Example with `EXP_NAME=blendergym-9b-dp6`:

```
LOCAL WORKING DIRECTORIES (NVMe — fast I/O):
├── /local-ssd/checkpoints/blendergym-9b-dp6/   ← CKPT_LOCAL
├── /local-ssd/prime-rl-output/                 ← OUTPUT_LOCAL
│   ├── logs/
│   └── blendergym-work/
└── /local-ssd/hf_cache/                        ← HF models

FUSE MOUNT (for reads/existence checks):
├── /threed-code/ericzyma/experiments/blendergym-9b-dp6/
│   ├── checkpoints/                            ← CKPT_S3
│   └── output/                                 ← OUTPUT_S3
└── /threed-code/ericzyma/tools/
    └── hf_cache_qwen3.5-9b.tar

S3 API (for writes via aws s3 sync):
├── s3://arcwm-code-us-west-2/ericzyma/experiments/blendergym-9b-dp6/
│   ├── checkpoints/                            ← CKPT_S3_BUCKET
│   └── output/                                 ← OUTPUT_S3_BUCKET
└── s3://arcwm-code-us-west-2/ericzyma/tools/
    └── hf_cache_qwen3.5-9b.tar
```

### Variable Definitions:
```bash
S3_PREFIX="/threed-code/ericzyma"
S3_EXP="${S3_PREFIX}/experiments/${EXP_NAME}"
CKPT_LOCAL="/local-ssd/checkpoints/${EXP_NAME}"
OUTPUT_LOCAL="/local-ssd/prime-rl-output"

# FUSE paths (reads only):
CKPT_S3="${S3_EXP}/checkpoints"
OUTPUT_S3="${S3_EXP}/output"

# S3 API paths (writes):
S3_BUCKET="s3://arcwm-code-us-west-2/ericzyma"
CKPT_S3_BUCKET="${S3_BUCKET}/experiments/${EXP_NAME}/checkpoints"
OUTPUT_S3_BUCKET="${S3_BUCKET}/experiments/${EXP_NAME}/output"
```

**Why two paths?**
- FUSE: Limited (no rename), fast sequential reads
- S3 API: Full atomicity, every sync is fresh PUT

---

## 3. Submission Pattern

### Command Template:
```bash
koala submit -m normal -g 8 --sync-code .:/data/work/prime-rl \
    -c "export HF_TOKEN=$HF_TOKEN && \
        export WANDB_API_KEY=$WANDB_API_KEY && \
        export EXP_NAME=blendergym-9b-dp6 && \
        cd /data/work/prime-rl && \
        . scripts/setup_kaola.sh --env blendergym && \
        uv run rl @ configs/multimodal/rl_blendergym_kaola.toml --ckpt.output_dir /local-ssd/checkpoints/\${EXP_NAME}"
```

### Critical Notes:
1. **Use `.` (source), not `bash`**
   - `. scripts/setup_kaola.sh` runs in current shell
   - `bash scripts/setup_kaola.sh` runs in subshell (loses exports)

2. **Single continuous command**
   - Uses `&&` to chain commands
   - All run in one pod's main shell
   - EXIT trap fires when entire chain completes

3. **Variable expansion timing**
   - `$HF_TOKEN` expanded locally (before submission)
   - Pod receives actual values, not `$` strings
   - Use double quotes in koala submit: `-c "... $HF_TOKEN ..."`

4. **Execution order in pod:**
   ```
   export vars
   → source setup_kaola.sh
       → [1/7] setup_python_deps
       → [2-5/7] env_setup() from envs/blendergym.sh
           → install Blender, dataset, etc.
           → OPTIX warmup (background)
       → [6/7] setup_hf_cache
       → [7/7] setup_s3_sync (background loop starts)
       → wait for OPTIX
   → uv run rl (training starts)
       → S3 sync runs every 5 min in background
   → [Pod exits]
       → EXIT trap: final sync + cleanup
   ```

---

## 4. Environment Plugin System

### Plugin Loading (setup_kaola.sh lines 145-152):
```bash
ENV_SCRIPT="${PROJECT_DIR}/scripts/envs/${ENV_NAME}.sh"
if [ ! -f "${ENV_SCRIPT}" ]; then
    echo "ERROR: env script not found: ${ENV_SCRIPT}"
    exit 1
fi
source "${ENV_SCRIPT}"
```

### Plugin Interface:

**Each plugin MUST have:**
1. `env_setup()` function — entry point
2. Prefixed helper functions to avoid collisions:
   - BlenderGym: `setup_bg_*`
   - Articraft: `setup_ac_*`

**Each plugin can assume these variables exist:**
- `$S3_PREFIX` — `/threed-code/ericzyma`
- `$FAST_MODE` — "true" to skip heavy steps
- `$RESUME_MODE` — "true" to skip output exists check
- `$PROJECT_DIR` — `/data/work/prime-rl`
- `$OUTPUT_LOCAL` — `/local-ssd/prime-rl-output`
- `$EXP_NAME` — the experiment name
- `$CKPT_LOCAL` — checkpoint directory

### Plugin Execution Order:
```
[1/7] setup_python_deps       ← shared, before env
[2-5/7] env_setup()           ← load from envs/${ENV_NAME}.sh
[6/7] setup_hf_cache          ← shared, after env
[7/7] setup_s3_sync           ← shared, background loop
```

### Example: BlenderGym Plugin (`scripts/envs/blendergym.sh`)
```bash
env_setup() {
    setup_bg_install_system_libs
    setup_bg_restore_blender
    setup_bg_restore_dataset
    setup_bg_install_python_pkg
    setup_bg_optix_warmup        # background, PID stored
    # ... launch services
}
```

### Example: Articraft Plugin (`scripts/envs/articraft.sh`)
```bash
env_setup() {
    setup_ac_sync_code
    setup_ac_restore_dataset
    setup_ac_install_system_libs
    setup_ac_install_python_pkg
    setup_ac_install_env_pkg
    setup_ac_verify_imports
}
```

---

## 5. Parameter Reference

| Flag | Type | Default | Effect |
|------|------|---------|--------|
| `--fast` | boolean | false | Skip dataset extraction, OPTIX warmup, S3 sync |
| `--resume` | boolean | false | Skip "S3 output exists" check |
| `--env <name>` | string | "blendergym" | Load `scripts/envs/<name>.sh` |

Example usage:
```bash
. scripts/setup_kaola.sh --fast --env articraft
. scripts/setup_kaola.sh --resume
```

---

## Troubleshooting

### S3 Sync Not Running
- Check: Is `FAST_MODE=true`? If so, sync is skipped.
- Check: Is `SYNC_PID` set? If not, background process failed to start.

### Variables Lost After Sourcing
- **Wrong**: `bash scripts/setup_kaola.sh && uv run rl ...`
  - Subshell loses exports
- **Right**: `. scripts/setup_kaola.sh && uv run rl ...`
  - Current shell keeps exports

### Pod Exits Before Final Sync
- EXIT trap should fire automatically
- If interrupted forcibly, data may be lost
- Always use `--resume` if pod exits before training completes

### Plugin Not Loading
- Check: File exists at `/data/work/prime-rl/scripts/envs/<name>.sh`
- Check: Plugin defines `env_setup()` function
- Check: Plugin uses prefixed function names (no collisions)


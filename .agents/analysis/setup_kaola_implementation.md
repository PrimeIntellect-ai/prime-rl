# Prime-RL Setup Analysis: setup_kaola.sh Implementation Details

## 1. The `setup_s3_sync()` Function — Full Implementation

Located in `scripts/setup_kaola.sh` (lines 111-140):

```bash
setup_s3_sync() {
    if [ "$FAST_MODE" = true ]; then
        echo "  Background sync: SKIPPED (fast mode)"
        return
    fi
    echo "  Starting background S3 sync..."
    sync_all() {
        local _sync_log="${OUTPUT_LOCAL}/logs/s3_sync.log"
        mkdir -p "$(dirname "${_sync_log}")"
        # 使用 aws s3 sync 直接走 S3 API，绕过 FUSE 所有限制：
        #   - 不依赖 rename()、mtime、overwrite — 每次是 PUT 新对象
        #   - --delete: 清理远端多余文件（如旧 checkpoint）
        #   - --quiet: 不逐文件打印（减少日志噪音）
        if [ -d "${CKPT_LOCAL}" ]; then
            aws s3 sync "${CKPT_LOCAL}/" "${CKPT_S3_BUCKET}/" \
                --delete --quiet >> "${_sync_log}" 2>&1 || true
        fi
        if [ -d "${OUTPUT_LOCAL}" ]; then
            aws s3 sync "${OUTPUT_LOCAL}/" "${OUTPUT_S3_BUCKET}/" \
                --delete --exclude 'broadcasts/*' --exclude '*.bin' \
                --quiet >> "${_sync_log}" 2>&1 || true
        fi
    }
    (while true; do sleep 300; sync_all; done) &
    SYNC_PID=$!
    echo "    PID: ${SYNC_PID} (every 5 min, via S3 API)"
    echo "    ${CKPT_LOCAL} -> ${CKPT_S3_BUCKET} (--delete)"
    echo "    ${OUTPUT_LOCAL} -> ${OUTPUT_S3_BUCKET} (excl broadcasts/*.bin)"
    trap "kill ${SYNC_PID} 2>/dev/null || true; [ -n \"\${OPTIX_PID:-}\" ] && kill \"\${OPTIX_PID}\" 2>/dev/null || true; type _blendergym_cleanup &>/dev/null && _blendergym_cleanup; sync_all" EXIT
}
```

### Key Components:

#### a. Inner `sync_all()` Function (Lines 117-133)
- **Purpose**: Performs atomic sync of local directories to S3 using AWS CLI
- **Two-part sync**:
  1. **Checkpoints**: `${CKPT_LOCAL}/ → ${CKPT_S3_BUCKET}/` with `--delete` flag
  2. **Output**: `${OUTPUT_LOCAL}/ → ${OUTPUT_S3_BUCKET}/` with:
     - `--delete`: Removes remote files not present locally
     - `--exclude 'broadcasts/*'`: Skips temporary render broadcasts
     - `--exclude '*.bin'`: Skips binary model files
- **S3 API bypass**: Uses `aws s3 sync` (S3 API), NOT FUSE mount (`${OUTPUT_S3}`)
  - FUSE only used for **reads/existence checks** (line 74)
  - S3 API used for **writes** to avoid rename() atomicity issues
- **Logging**: Appends to `${OUTPUT_LOCAL}/logs/s3_sync.log`
- **Error handling**: `|| true` prevents script exit on sync failures

#### b. Background Loop (Line 134)
```bash
(while true; do sleep 300; sync_all; done) &
```
- Runs in **subshell** (`(...)`) to isolate it as background process
- **Frequency**: Every 300 seconds (5 minutes)
- **Non-blocking**: Parent shell continues immediately
- PID saved to `SYNC_PID` variable for later cleanup

#### c. PID Tracking (Line 135)
```bash
SYNC_PID=$!
```
- `$!` captures the PID of the last backgrounded process
- Used in EXIT trap for graceful termination

#### d. EXIT Trap (Line 139)
```bash
trap "kill ${SYNC_PID} 2>/dev/null || true; \
      [ -n \"\${OPTIX_PID:-}\" ] && kill \"\${OPTIX_PID}\" 2>/dev/null || true; \
      type _blendergym_cleanup &>/dev/null && _blendergym_cleanup; \
      sync_all" EXIT
```

**Execution sequence when shell exits**:
1. Kill sync background process: `kill ${SYNC_PID} 2>/dev/null || true`
   - Redirects stderr to /dev/null
   - `|| true` ensures success (even if process already dead)
2. Kill OPTIX warmup if running: `[ -n "${OPTIX_PID:-}" ] && kill "${OPTIX_PID}" 2>/dev/null || true`
   - `${OPTIX_PID:-}` expands to empty string if unset (safe parameter expansion)
3. Call env plugin cleanup: `type _blendergym_cleanup &>/dev/null && _blendergym_cleanup`
   - Only runs if function exists (checking with `type`)
4. **Final sync**: `sync_all`
   - Ensures all pending checkpoint/output changes are uploaded before exit

---

## 2. EXP_NAME and S3 Path Variables — Structure & Duality

### EXP_NAME Handling (Lines 51-65)

```bash
if [ -z "${EXP_NAME:-}" ]; then
    echo "ERROR: EXP_NAME not set. Export it before running setup."
    echo "  e.g.: export EXP_NAME=blendergym-9b-dp6"
    exit 1
fi

HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-9B}"
HF_MODEL_SHORT=$(echo "${HF_MODEL}" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')

S3_PREFIX="/threed-code/ericzyma"
S3_EXP="${S3_PREFIX}/experiments/${EXP_NAME}"
OUTPUT_LOCAL="/local-ssd/prime-rl-output"
CKPT_LOCAL="/local-ssd/checkpoints/${EXP_NAME}"
CKPT_S3="${S3_EXP}/checkpoints"
OUTPUT_S3="${S3_EXP}/output"

# S3 API 路径（绕过 FUSE，用于写入）
S3_BUCKET="s3://arcwm-code-us-west-2/ericzyma"
CKPT_S3_BUCKET="${S3_BUCKET}/experiments/${EXP_NAME}/checkpoints"
OUTPUT_S3_BUCKET="${S3_BUCKET}/experiments/${EXP_NAME}/output"
HF_CACHE_TAR="${S3_PREFIX}/tools/hf_cache_${HF_MODEL_SHORT}.tar"
PROJECT_DIR="/data/work/prime-rl"
```

### **The Dual-Path Pattern**:

| Purpose | Path Variable | Example | Used For |
|---------|---------------|---------|----------|
| **FUSE Mount (Reads Only)** | `CKPT_S3` / `OUTPUT_S3` | `/threed-code/ericzyma/experiments/blendergym-9b-dp6/checkpoints` | Existence checks (line 74), reading pre-existing data |
| **S3 API (Writes)** | `CKPT_S3_BUCKET` / `OUTPUT_S3_BUCKET` | `s3://arcwm-code-us-west-2/ericzyma/experiments/blendergym-9b-dp6/checkpoints` | `aws s3 sync` writes (lines 125-131) |
| **Local NVMe (Working)** | `CKPT_LOCAL` / `OUTPUT_LOCAL` | `/local-ssd/checkpoints/blendergym-9b-dp6` | Actual training I/O |

### **Why Two Different S3 Paths?**

**FUSE path** (`/threed-code/...`): 
- Kernel-level mount point
- No rename() support → problematic for DCP (Distributed CheckPointing)
- Fast for sequential reads
- Used only for **existence verification** before training

**S3 API path** (`s3://bucket/...`):
- Explicit S3 bucket + region
- Bypasses FUSE mount entirely
- Every `sync` is a fresh PUT operation (no atomic rename needed)
- Used for all **write operations** to guarantee atomicity

### **Validation Pattern** (Line 74):

```bash
if [ "$FAST_MODE" = false ] && [ "$RESUME_MODE" = false ] && [ -d "${OUTPUT_S3}/logs" ]; then
    echo "ERROR: S3 output already exists: ${OUTPUT_S3}/logs"
    echo "  Previous training data would be overwritten."
    echo "  To resume:      add --resume"
    echo "  To start fresh:  rclone purge threed-code:arcwm-code-us-west-2/${S3_EXP#/threed-code/}"
    echo "  Or use a different EXP_NAME."
    exit 1
fi
```
- Uses **FUSE path** to check if previous run exists (fast directory lookup)
- Prevents accidental data overwrites
- Can be skipped with `--resume` flag

---

## 3. Invocation Pattern: How `koala submit -c` Executes setup_kaola.sh

### **Invocation Method: Source in a Subshell**

From the quick commands in `.agents/kaola/README.md` (lines 35-36):

```bash
koala submit -m normal -g 8 --sync-code .:/data/work/prime-rl \
    -c "export HF_TOKEN=$HF_TOKEN && export WANDB_API_KEY=$WANDB_API_KEY && export EXP_NAME=blendergym-9b-dp6 && cd /data/work/prime-rl && . scripts/setup_kaola.sh --env blendergym && uv run rl @ configs/multimodal/rl_blendergym_kaola.toml --ckpt.output_dir /local-ssd/checkpoints/\${EXP_NAME}"
```

### **Breakdown of `-c` Argument**:

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `export HF_TOKEN=$HF_TOKEN` | Pass HuggingFace token (expanded locally before submission) |
| 2 | `export WANDB_API_KEY=$WANDB_API_KEY` | Pass WandB token |
| 3 | `export EXP_NAME=blendergym-9b-dp6` | Set experiment name |
| 4 | `cd /data/work/prime-rl` | Navigate to project directory |
| 5 | `. scripts/setup_kaola.sh --env blendergym` | **Source** (not `bash`) the setup script |
| 6 | `uv run rl @ configs/...` | Run training command using returned environment |

### **Key Points**:

1. **Source (`.`) NOT Bash (`bash`)**
   - `. scripts/setup_kaola.sh` sources the script in the **current shell**
   - All `export` statements in setup affect the training command that follows
   - If invoked as `bash scripts/setup_kaola.sh`, exported variables would be lost
   - See comment in setup_kaola.sh (line 12): "set -euo pipefail 会影响调用方 shell"

2. **Single Continuous Command Chain**
   - Uses `&&` to chain success-dependent commands
   - All run in **one KOALA pod's main process** (not separate containers)
   - EXIT trap fires when entire `-c` command completes (pod exits)

3. **Variable Expansion Timing**
   - **Local Mac shell**: `$HF_TOKEN` and `$WANDB_API_KEY` are expanded before submission
   - **Pod shell**: Receives the actual values, not `$`-prefixed strings
   - This is why `.agents/kaola/README.md` line 42 warns about using double quotes, not single quotes

4. **Complete Execution Graph**

```
koala submit -c "..."
  └─ [Pod starts, single /bin/zsh shell]
      ├─ export HF_TOKEN=<actual-token>
      ├─ export WANDB_API_KEY=<actual-key>
      ├─ export EXP_NAME=blendergym-9b-dp6
      ├─ cd /data/work/prime-rl
      ├─ . scripts/setup_kaola.sh --env blendergym
      │   ├─ [1/7] setup_python_deps            (uv sync)
      │   ├─ [2-5/7] env_setup()                (from envs/blendergym.sh)
      │   │   ├─ install system libs
      │   │   ├─ restore Blender
      │   │   ├─ restore dataset
      │   │   ├─ install blendergym package
      │   │   └─ OPTIX warmup (background, PID stored)
      │   ├─ [6/7] setup_hf_cache               (parallel with OPTIX)
      │   ├─ [7/7] setup_s3_sync                (background loop)
      │   └─ wait ${OPTIX_PID}                  (wait for warmup)
      │
      ├─ uv run rl @ configs/...               (training starts)
      │   └─ [6 Blender workers + training]
      │   └─ Background S3 sync runs every 5 min
      │
      └─ [On pod exit or error]
          └─ trap EXIT
              ├─ kill ${SYNC_PID}
              ├─ kill ${OPTIX_PID}
              ├─ _blendergym_cleanup()
              └─ sync_all                      (final upload)
```

---

## 4. Env Plugin Pattern: `--env` and `scripts/envs/*.sh` Mechanism

### **Plugin Loading** (Lines 145-152)

```bash
# ============================================================================
# 加载 env 插件 + 执行主流程
# ============================================================================
cd "${PROJECT_DIR}"

ENV_SCRIPT="${PROJECT_DIR}/scripts/envs/${ENV_NAME}.sh"
if [ ! -f "${ENV_SCRIPT}" ]; then
    echo "ERROR: env script not found: ${ENV_SCRIPT}"
    exit 1
fi
source "${ENV_SCRIPT}"
```

### **Plugin Interface Contract**

Each env plugin (e.g., `scripts/envs/blendergym.sh`, `scripts/envs/articraft.sh`) **must**:

1. **Define `env_setup()` function** — entry point called by base script
2. **Only define functions and variables** at top level (line 14-15 of blendergym.sh)
3. **Assume these variables are already set** from base script:
   - `$S3_PREFIX` — S3 FUSE mount root (e.g., `/threed-code/ericzyma`)
   - `$FAST_MODE` — "true" to skip non-essential steps
   - `$PROJECT_DIR` — prime-rl code directory (e.g., `/data/work/prime-rl`)
   - `$OUTPUT_LOCAL` — training output directory
   - `$EXP_NAME` — experiment name

4. **Use consistent naming** to avoid conflicts:
   - BlenderGym: `setup_bg_*` functions
   - Articraft: `setup_ac_*` functions
   - This prevents function name collisions when both plugins are sourced

### **Example: BlenderGym Plugin** (`scripts/envs/blendergym.sh`)

```bash
# Available variables from base script (already exported)
BLENDER_VERSION="4.2.0"
BLENDER_DIR="/local-ssd/blender-${BLENDER_VERSION}-linux-x64"
OPTIX_CACHE_TAR="${S3_PREFIX}/tools/optix_cache.tar"

# Internal functions (prefixed setup_bg_)
setup_bg_install_system_libs() { ... }
setup_bg_restore_blender() { ... }
setup_bg_restore_dataset() { ... }
setup_bg_install_python_pkg() { ... }
setup_bg_optix_warmup() { ... }

# Main entry point (called by base script)
env_setup() {
    setup_bg_install_system_libs
    setup_bg_restore_blender
    ...
}
```

### **Example: Articraft Plugin** (`scripts/envs/articraft.sh`)

Follows identical pattern:
- Variables: `ARTICRAFT_DIR`, `ARTICRAFT_CODE_TAR`, `ARTICRAFT_DATASET_TAR`
- Functions: `setup_ac_sync_code()`, `setup_ac_restore_dataset()`, `setup_ac_install_python_pkg()`, etc.
- Entry: `env_setup()` calls them in order

### **Order of Execution** (Lines 154-168)

```bash
echo "=== [1/7] Python dependencies ==="
setup_python_deps                    # uv sync (shared by all envs)

echo "=== [2/7~5/7] Environment: ${ENV_NAME} ==="
env_setup                             # Called from sourced env plugin

echo "=== [6/7] HF model cache ==="
setup_hf_cache                        # Shared HuggingFace cache extraction

echo "=== [7/7] Background S3 sync ==="
setup_s3_sync                         # Background sync loop

# Then wait for OPTIX if needed
if [ -n "${OPTIX_PID:-}" ]; then
    wait "${OPTIX_PID}" 2>/dev/null || true
fi
```

**Dependency notes**:
- `[1/7]` must complete before `[2-5/7]` (uv needed for `uv pip install` in env_setup)
- `[6/7]` and `[7/7]` can run in parallel with `[2-5/7]` OPTIX warmup
- `wait` ensures OPTIX compilation finishes before training starts (prevents GPU memory conflicts)

### **Plugin Registration**

Plugins are **auto-discovered** from command line:
```bash
# Default: --env blendergym → loads scripts/envs/blendergym.sh
koala submit ... -c ". scripts/setup_kaola.sh && ..."

# Custom: --env articraft → loads scripts/envs/articraft.sh
koala submit ... -c ". scripts/setup_kaola.sh --env articraft && ..."

# Error if not found:
# ERROR: env script not found: /data/work/prime-rl/scripts/envs/nonexistent.sh
```

---

## 5. Parameter Parsing

```bash
FAST_MODE=false
RESUME_MODE=false
ENV_NAME="blendergym"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast) FAST_MODE=true; shift ;;
        --resume) RESUME_MODE=true; shift ;;
        --env)
            if [[ $# -lt 2 ]]; then echo "ERROR: --env requires a name"; exit 1; fi
            ENV_NAME="$2"; shift 2 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done
```

| Flag | Effect | Use Case |
|------|--------|----------|
| `--fast` | Skips dataset extraction, OPTIX warmup, S3 sync | Local debugging, fast iteration |
| `--resume` | Skips "S3 output already exists" check | Resuming from checkpoint |
| `--env <name>` | Loads `scripts/envs/<name>.sh` | Different environments (blendergym vs articraft) |

---

## Summary Table

| Aspect | Implementation |
|--------|-----------------|
| **Sync Function** | Inner `sync_all()` loops every 5 min, runs `aws s3 sync` on two directories (checkpoints + output) with exclusions |
| **Background Loop** | `(while true; do sleep 300; sync_all; done) &` — nonblocking subshell |
| **PID Tracking** | `SYNC_PID=$!` captures background process ID |
| **EXIT Trap** | Kills sync and OPTIX PIDs, calls env cleanup, runs final sync |
| **FUSE Paths** | `/threed-code/ericzyma/...` — reads only, used for existence checks |
| **S3 API Paths** | `s3://arcwm-code-us-west-2/...` — writes only, uses `aws s3 sync` |
| **Invocation** | `. scripts/setup_kaola.sh --env <name>` in single continuous shell command chain |
| **Plugin Loading** | Auto-loaded via `source "${PROJECT_DIR}/scripts/envs/${ENV_NAME}.sh"` |
| **Plugin Interface** | Must define `env_setup()` function; uses prefixed helpers to avoid collisions |


# Prime-RL setup_kaola.sh Analysis

This directory contains detailed documentation of the `scripts/setup_kaola.sh` script and its environment plugin system.

## Files

### 1. `setup_kaola_implementation.md` ⭐ START HERE
**Complete technical reference** covering all four aspects:
- Full `setup_s3_sync()` implementation with inner `sync_all()` function, background loop, EXIT trap, and PID tracking
- EXP_NAME and S3 path variable structures (both FUSE paths for reads and S3 API paths for writes)
- How `koala submit -c` invokes the script (source vs bash, variable expansion timing, execution graph)
- Environment plugin pattern (`--env` flag, `scripts/envs/*.sh` mechanism, interface contract)

**Use this when:**
- You need to understand how the sync works at every level
- You're debugging path-related issues
- You're writing a new environment plugin
- You need to understand the submission workflow end-to-end

### 2. `setup_kaola_quick_reference.md` 🚀 COPY-PASTE GUIDE
**Practical guide** with ready-to-use code snippets:
- Full sync function code you can copy directly
- Path variable quick lookup with examples
- Submission command template
- Parameter reference table
- Troubleshooting checklist

**Use this when:**
- You need to quickly reference variable names
- You want to debug a specific issue
- You're running a training job and need the exact command
- You need to understand what went wrong

### 3. `setup_kaola_diagrams.md` 📊 VISUAL REFERENCE
**Architecture diagrams** showing:
1. Sync function internal architecture
2. Background sync timeline (T+0s, T+300s, T+600s, etc.)
3. Dual-path architecture (Local NVMe → FUSE → S3 Bucket)
4. Submission execution flow (local shell → pod → training → exit)
5. Plugin loading sequence
6. Error handling & cleanup sequence

**Use this when:**
- You want to understand the big picture
- You're explaining the architecture to someone
- You need to visualize the data flow
- You're designing a new component

## Key Concepts at a Glance

### The Problem
Training on KAOLA cluster needs to:
1. **Persist data** to S3 (survive pod exit)
2. **Sync frequently** (prevent data loss if pod dies)
3. **Respect FUSE limitations** (no rename() support, no atomic mtime)
4. **Clean up gracefully** (kill background processes on exit)

### The Solution
```bash
setup_s3_sync() {
    # Define inner function that can be called repeatedly
    sync_all() {
        aws s3 sync /local-ssd/checkpoints/ s3://...  # via S3 API
        aws s3 sync /local-ssd/prime-rl-output/ s3://...
    }
    
    # Start background loop: every 5 minutes
    (while true; do sleep 300; sync_all; done) &
    SYNC_PID=$!
    
    # Register EXIT trap: ensures final sync before pod exits
    trap "kill $SYNC_PID; sync_all" EXIT
}
```

### The Dual-Path Pattern
| Path | Used For | Example |
|------|----------|---------|
| **FUSE** `/threed-code/...` | Reads & existence checks | Check if `/threed-code/ericzyma/experiments/blendergym-9b-dp6/output/logs` exists |
| **S3 API** `s3://bucket/...` | Writes via `aws s3 sync` | Upload to `s3://arcwm-code-us-west-2/ericzyma/experiments/.../` |
| **Local** `/local-ssd/...` | Fast training I/O | Write checkpoints to `/local-ssd/checkpoints/...` |

**Why?** FUSE doesn't support rename() (needed for atomic DCP), so sync uses S3 API (fresh PUT each time).

### The Submission Pattern
```bash
koala submit -c ". scripts/setup_kaola.sh --env blendergym && uv run rl @ ..."
                  ▲                                         ▲
        Use source (.), not bash               All in one pod's shell
```

**Key:** Use `.` (source), not `bash` — keeps exports intact for training command.

### The Plugin System
1. Base script (`setup_kaola.sh`) defines shared setup
2. Plugin (`scripts/envs/<name>.sh`) defines `env_setup()` function
3. Base script sources plugin and calls `env_setup()`
4. Plugin can access: `$S3_PREFIX`, `$FAST_MODE`, `$PROJECT_DIR`, etc.

**Example:**
```bash
# scripts/envs/blendergym.sh
setup_bg_install_system_libs() { apt install libegl1; }
setup_bg_restore_blender() { tar xf blender.tar; }
env_setup() {
    setup_bg_install_system_libs
    setup_bg_restore_blender
    # ... more steps
}
```

## Frequently Asked Questions

### Q: Why does training need `--sync-code`?
**A:** Without `--sync-code`, the pod uses the Docker image's built-in code. Your local changes won't be synced. Add it every time you test.

### Q: Should I use `bash` or `.` to run setup_kaola.sh?
**A:** Always use `.` (source). `bash` runs in a subshell and loses all exports. See line 12 of setup_kaola.sh.

### Q: What does the EXIT trap do?
**A:** Ensures graceful cleanup on pod exit:
1. Stops the background sync loop
2. Kills OPTIX warmup (if still running)
3. Kills service processes (BlenderGym services)
4. Runs one final sync to S3

Without this, data might be lost if pod exits abruptly.

### Q: Why two S3 paths?
**A:** FUSE mount (`/threed-code/...`) doesn't support rename(), which DCP (Distributed CheckPointing) needs. S3 API (`s3://bucket/...`) bypasses FUSE entirely, so each sync is a fresh PUT with no rename issues.

### Q: How often does background sync run?
**A:** Every 5 minutes (300 seconds). Plus one final sync when pod exits.

### Q: What if I want to resume training?
**A:** Use `--resume` flag: `. scripts/setup_kaola.sh --resume`. This skips the "S3 output already exists" check.

### Q: How do I write a new environment plugin?
**A:** Create `scripts/envs/my_env.sh` with:
1. Helper functions: `setup_myenv_*`
2. Main function: `env_setup()` that calls helpers
3. Access base variables: `$S3_PREFIX`, `$FAST_MODE`, `$PROJECT_DIR`, etc.

See `scripts/envs/articraft.sh` for a complete example.

## Reading Order

1. **First time?** Start with `setup_kaola_diagrams.md` (Section 4: Submission Execution Flow)
2. **Need details?** Read `setup_kaola_implementation.md` section by section
3. **Running a job?** Use `setup_kaola_quick_reference.md` to copy the command
4. **Debugging?** Check `setup_kaola_quick_reference.md` Troubleshooting section

## Files Referenced

- `scripts/setup_kaola.sh` — Main orchestration script
- `scripts/envs/blendergym.sh` — BlenderGym environment plugin
- `scripts/envs/articraft.sh` — Articraft environment plugin
- `.agents/kaola/README.md` — Quick commands and environment setup

## Last Updated

2026-05-25

Analysis of prime-rl commit: (run `git log --oneline scripts/setup_kaola.sh` to see history)

# Prime-RL setup_kaola.sh Documentation Index

## Quick Links

| Document | Lines | Purpose | When to Read |
|----------|-------|---------|--------------|
| [README.md](README.md) | 164 | Overview + FAQ + key concepts | First! |
| [setup_kaola_implementation.md](setup_kaola_implementation.md) | 385 | Complete technical reference | For deep understanding |
| [setup_kaola_quick_reference.md](setup_kaola_quick_reference.md) | 252 | Code snippets + troubleshooting | While debugging/coding |
| [setup_kaola_diagrams.md](setup_kaola_diagrams.md) | 368 | Visual architecture diagrams | For system design |

---

## Answer to Your Four Questions

### 1️⃣ The `setup_s3_sync()` Function

**Q:** Full implementation including `sync_all()` inner function, background loop, EXIT trap, and PID tracking

**Answer:**
- **Code:** See `setup_kaola_implementation.md` § 1
- **Quick:** See `setup_kaola_quick_reference.md` § 1 (copy-paste ready)
- **Visual:** See `setup_kaola_diagrams.md` Diagram 1 (architecture) + Diagram 2 (timeline)

**Key Points:**
```bash
setup_s3_sync() {
    sync_all() {                    # Inner function (lines 117-133)
        aws s3 sync checkpoints/    # Sync via S3 API
        aws s3 sync output/         # with exclusions
    }
    (while true; do sleep 300; sync_all; done) &  # Background loop
    SYNC_PID=$!                     # PID tracking
    trap "kill ${SYNC_PID}; sync_all" EXIT        # EXIT trap
}
```

---

### 2️⃣ EXP_NAME and S3 Path Variables

**Q:** How are FUSE paths for reads and S3 API paths for writes structured?

**Answer:**
- **Full Explanation:** See `setup_kaola_implementation.md` § 2
- **Variable Lookup:** See `setup_kaola_quick_reference.md` § 2
- **Visual Diagram:** See `setup_kaola_diagrams.md` Diagram 3 (dual-path architecture)

**Path Structure with Example (EXP_NAME=blendergym-9b-dp6):**

| Layer | Purpose | Variables | Paths |
|-------|---------|-----------|-------|
| **Local NVMe** | Fast I/O | `CKPT_LOCAL`, `OUTPUT_LOCAL` | `/local-ssd/checkpoints/...`, `/local-ssd/prime-rl-output/` |
| **FUSE Mount** | Reads only | `CKPT_S3`, `OUTPUT_S3` | `/threed-code/ericzyma/experiments/blendergym-9b-dp6/...` |
| **S3 Bucket** | Writes only | `CKPT_S3_BUCKET`, `OUTPUT_S3_BUCKET` | `s3://arcwm-code-us-west-2/ericzyma/experiments/...` |

**Why two S3 paths?** FUSE doesn't support rename(), breaking atomic sync. S3 API bypasses FUSE.

---

### 3️⃣ How `koala submit -c` Invokes setup_kaola.sh

**Q:** Is it `source scripts/setup_kaola.sh && ...` or `bash -c "source ... && ..."`?

**Answer:** 
- **Full Explanation:** See `setup_kaola_implementation.md` § 3
- **Command Template:** See `setup_kaola_quick_reference.md` § 3
- **Execution Flow:** See `setup_kaola_diagrams.md` Diagram 4

**The Answer:**
```bash
koala submit -m normal -g 8 -c ". scripts/setup_kaola.sh && uv run rl @ ..."
                                ▲ Use source (.), not bash
                                  All run in ONE pod's main shell
```

**Key Points:**
1. **Use `.` (source), NOT `bash`** — keeps exports intact for training
2. **Single continuous command** with `&&` chaining
3. **Variable expansion** happens locally (before pod starts)
4. **EXIT trap** fires automatically when entire chain completes

---

### 4️⃣ Environment Plugin Pattern

**Q:** The `--env` flag and `scripts/envs/*.sh` mechanism

**Answer:**
- **Full System Design:** See `setup_kaola_implementation.md` § 4
- **Plugin Examples:** See BlenderGym and Articraft plugins
- **Loading Sequence:** See `setup_kaola_diagrams.md` Diagram 5

**How it Works:**
```bash
# Base script loads plugin:
source "${PROJECT_DIR}/scripts/envs/${ENV_NAME}.sh"

# Plugin MUST define env_setup() function:
env_setup() {
    setup_bg_install_system_libs
    setup_bg_restore_blender
    # ... more setup steps
}

# Base script calls it:
env_setup  # All plugin-specific setup runs here
```

**Plugin Interface:**
- Plugins can assume these variables exist: `$S3_PREFIX`, `$FAST_MODE`, `$PROJECT_DIR`, `$OUTPUT_LOCAL`, `$CKPT_LOCAL`, `$EXP_NAME`
- Plugins should use prefixed helpers: `setup_bg_*` for BlenderGym, `setup_ac_*` for Articraft
- Optional: Define `_${ENV_NAME}_cleanup()` function (called in EXIT trap)

---

## Directory Structure

```
.agents/analysis/
├── INDEX.md                              ← You are here
├── README.md                             ← Overview + FAQ
├── setup_kaola_implementation.md         ← Full technical details
├── setup_kaola_quick_reference.md        ← Copy-paste snippets
└── setup_kaola_diagrams.md              ← Visual diagrams
```

---

## Reading Paths

### 🚀 Fast Track (5 minutes)
1. README.md (key concepts)
2. setup_kaola_diagrams.md (Diagram 4: submission flow)
3. setup_kaola_quick_reference.md (reference while working)

### 📖 Deep Dive (30 minutes)
1. setup_kaola_diagrams.md (all 6 diagrams)
2. setup_kaola_implementation.md (sections 1-4)
3. setup_kaola_quick_reference.md (troubleshooting)

### 🔧 Debugging (when broken)
1. README.md (FAQ section)
2. setup_kaola_quick_reference.md (troubleshooting checklist)
3. setup_kaola_diagrams.md (Diagram 6: error handling)

---

## Common Questions Quick Answers

**Q: Why does my training lose `$HF_TOKEN` after setup?**
- **A:** Use `. scripts/setup_kaola.sh` not `bash scripts/setup_kaola.sh`. See README FAQ.

**Q: My S3 sync isn't running. What's wrong?**
- **A:** Check if `FAST_MODE=true`. If so, sync is skipped. See quick_reference.md troubleshooting.

**Q: How do I write a new environment plugin?**
- **A:** Create `scripts/envs/my_env.sh`, define `env_setup()`, use `setup_myenv_*` prefixes. See implementation.md § 4.

**Q: What's the difference between `/threed-code/...` and `s3://bucket/...` paths?**
- **A:** FUSE vs S3 API. FUSE for reads, S3 API for writes. See implementation.md § 2 or quick_reference.md § 2.

**Q: Can I resume training from a checkpoint?**
- **A:** Use `--resume` flag: `. scripts/setup_kaola.sh --resume`. See quick_reference.md § 5.

---

## Source Files

These docs are extracted from / reference:
- `scripts/setup_kaola.sh` (main script)
- `scripts/envs/blendergym.sh` (plugin example)
- `scripts/envs/articraft.sh` (plugin example)
- `.agents/kaola/README.md` (quick commands)
- `.agents/kaola/paths.md` (path mappings)

---

## Last Updated

**2026-05-25** — Complete analysis of prime-rl `setup_kaola.sh`

**To check for updates:**
```bash
cd /Users/zhiyuanma/Desktop/codes/prime-rl
git log --oneline scripts/setup_kaola.sh
```

---

## Next Steps

- **Ready to submit a job?** Copy the template from `setup_kaola_quick_reference.md` § 3
- **Writing a plugin?** Follow the pattern in `setup_kaola_implementation.md` § 4
- **Debugging an issue?** Check troubleshooting in `setup_kaola_quick_reference.md` § 5
- **Understanding the architecture?** Study `setup_kaola_diagrams.md` Diagram 4

---

**Questions?** Check README.md FAQ section or raise a GitHub issue.

Happy coding! 🚀

# Bottleneck Investigation - Documentation Index

## 📋 Quick Start
1. Read **[INVESTIGATION_SUMMARY.md](INVESTIGATION_SUMMARY.md)** for overview
2. Follow **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** to deploy and test
3. Use **[COMMANDS.md](COMMANDS.md)** for quick command reference
4. Check **[LOG_PATTERNS.md](LOG_PATTERNS.md)** to understand logs
5. Run **[analyze_logs.sh](analyze_logs.sh)** to analyze results

## 📚 Documentation Files

### Main Documents

#### [INVESTIGATION_SUMMARY.md](INVESTIGATION_SUMMARY.md)
**Purpose:** Complete technical overview of the investigation
**Contains:**
- Problem statement
- Solution architecture (multi-layer instrumentation)
- Detailed code changes in each file
- Diagnostic signatures for different bottleneck types
- Expected results before/after fix
- Remote machine status

**Read this if:** You need to understand the full scope of the investigation

---

#### [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
**Purpose:** Step-by-step guide to deploy and test
**Contains:**
- Git commands to commit and push
- SSH instructions for remote machine
- Commands to start inference/trainer/orchestrator
- Real-time monitoring commands
- Log collection and analysis workflow

**Read this if:** You're ready to run the test on the A100 machine

---

#### [README.md](README.md)
**Purpose:** Comprehensive reference for instrumentation and analysis
**Contains:**
- Detailed description of each instrumentation point
- Log format specifications
- Metrics definitions and expected ranges
- Testing protocol
- Analysis commands
- Troubleshooting tips

**Read this if:** You need detailed technical reference

---

### Quick Reference Guides

#### [LOG_PATTERNS.md](LOG_PATTERNS.md)
**Purpose:** Visual examples of log patterns and what they mean
**Contains:**
- Example log sequences for normal operation
- Example log sequences for each bottleneck type
- Grep commands to extract specific patterns
- Expected metric ranges
- Issue signatures and fixes

**Read this if:** You have logs and need to interpret them

---

#### [COMMANDS.md](COMMANDS.md)
**Purpose:** Comprehensive command cheat sheet
**Contains:**
- SSH and file transfer commands
- GPU monitoring commands
- Process management (kill, tmux)
- Log management (tail, grep, compress)
- Service startup commands
- Analysis commands
- One-liner recipes

**Read this if:** You need specific commands quickly

---

### Tools

#### [analyze_logs.sh](analyze_logs.sh)
**Purpose:** Automated log analysis script
**Usage:**
```bash
./analyze_logs.sh <path_to_orchestrator.log> [path_to_inference.log]
```
**Produces:**
- `analysis_<timestamp>/` directory with:
  - `SUMMARY.md` - Diagnosis and recommendations
  - Statistics files for each metric
  - Timeline correlation
  - Top 10 slowest operations

**Use this:** After collecting logs from a test run

---

## 🎯 Use Cases

### "I want to understand what was changed"
→ Read [INVESTIGATION_SUMMARY.md](INVESTIGATION_SUMMARY.md) sections:
- "Solution: Multi-Layer Instrumentation"
- "Files Modified"

### "I want to run a test"
→ Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) from top to bottom

### "I need to start services on the remote machine"
→ See [COMMANDS.md](COMMANDS.md) section: "Starting Services"

### "I have logs and need to analyze them"
→ Run: `./analyze_logs.sh <log_file>`
→ Then read the generated `SUMMARY.md`

### "I see strange patterns in logs"
→ Compare with examples in [LOG_PATTERNS.md](LOG_PATTERNS.md)

### "I need a specific command"
→ Search [COMMANDS.md](COMMANDS.md) using Cmd+F / Ctrl+F

### "I want to understand a specific metric"
→ See [README.md](README.md) section: "Metrics to Track"

### "The test didn't work as expected"
→ See [README.md](README.md) section: "Troubleshooting"
→ Or [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) section: "Troubleshooting"

---

## 🔬 Investigation Workflow

```
1. Understanding Phase
   ├─→ Read INVESTIGATION_SUMMARY.md (overview)
   ├─→ Read README.md (technical details)
   └─→ Review LOG_PATTERNS.md (expected outputs)

2. Deployment Phase
   ├─→ Follow DEPLOYMENT_GUIDE.md (step-by-step)
   └─→ Use COMMANDS.md (command reference)

3. Monitoring Phase
   ├─→ Use COMMANDS.md (real-time monitoring)
   └─→ Compare with LOG_PATTERNS.md (interpretation)

4. Analysis Phase
   ├─→ Run analyze_logs.sh (automated analysis)
   └─→ Read generated SUMMARY.md (results)

5. Diagnosis Phase
   ├─→ Compare metrics with README.md (expected ranges)
   └─→ Match patterns with LOG_PATTERNS.md (signatures)

6. Fix Validation Phase
   ├─→ Compare before/after logs
   └─→ Verify queue_ms reduction
```

---

## 📊 Key Metrics Reference

| Metric | Location | Good | Warning | Critical |
|--------|----------|------|---------|----------|
| `queue_ms` | orchestrator.log | <100ms | 100-1000ms | >1000ms |
| `rpc_ms` | orchestrator.log | <2000ms | 2000-5000ms | >5000ms |
| `write_ms` | trainer.log | <1000ms | 1000-5000ms | >5000ms |
| `wait_ms` | orchestrator.log | <500ms | 500-2000ms | >2000ms |
| `trunc_pct` | orchestrator.log | <10% | 10-30% | >30% |

---

## 🗂️ Directory Structure

```
bottleneck_logs/
├── INDEX.md                    ← You are here
├── INVESTIGATION_SUMMARY.md    ← Start here for overview
├── README.md                   ← Technical reference
├── DEPLOYMENT_GUIDE.md         ← Testing guide
├── LOG_PATTERNS.md             ← Log interpretation
├── COMMANDS.md                 ← Command cheat sheet
├── analyze_logs.sh             ← Analysis tool
│
├── analysis_<timestamp>/       ← Generated by analyze_logs.sh
│   ├── SUMMARY.md
│   ├── queue_ms_stats.txt
│   ├── rpc_ms_stats.txt
│   ├── ckpt_wait_stats.txt
│   ├── slowest_updates.txt
│   └── timeline.txt
│
└── <collected_logs>/           ← Logs from test runs
    ├── orchestrator.log
    ├── inference.log
    └── trainer_rank_0.log
```

---

## 🎓 Learning Path

### For New Team Members
1. [INVESTIGATION_SUMMARY.md](INVESTIGATION_SUMMARY.md) - Understand the problem
2. [LOG_PATTERNS.md](LOG_PATTERNS.md) - Learn to read logs
3. [COMMANDS.md](COMMANDS.md) - Familiarize with tools

### For Testing/QA
1. [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - How to run tests
2. [COMMANDS.md](COMMANDS.md) - Commands to execute
3. Run `analyze_logs.sh` - Automated analysis

### For Debugging
1. [LOG_PATTERNS.md](LOG_PATTERNS.md) - Identify patterns
2. [README.md](README.md) - Understand instrumentation
3. [COMMANDS.md](COMMANDS.md) - Extract specific data

---

## 🚀 Expected Outcomes

### Success Criteria
✅ `queue_ms < 300ms` consistently
✅ `rpc_ms` reasonable for model size
✅ No regression in generation throughput
✅ RL quality metrics maintained

### Before Fix (Baseline)
```
Queue Delay: Mean = 28453.2 ms ❌
RPC Time:    Mean = 1234.5 ms  ✅
```

### After Fix (Target)
```
Queue Delay: Mean = 87.2 ms  ✅
RPC Time:    Mean = 1234.5 ms  ✅
```

---

## 📞 Contact & Support

- **Code Changes:** See git commit history
- **Remote Machine:** ubuntu@216.81.248.153
- **Issues:** Document in analysis output or team communication

---

## 🔄 Update History

- **2025-XX-XX**: Initial investigation and instrumentation
- **Latest**: All files created, ready for deployment and testing

---

## 📝 Notes

- All timestamps use `time.monotonic()` for accuracy
- Trace IDs correlate requests across logs
- Dedicated admin client already implemented
- Remote A100 machine currently idle (ready for testing)

---

**Navigation:**
- [↑ Back to Top](#bottleneck-investigation---documentation-index)
- [→ Start Investigation](INVESTIGATION_SUMMARY.md)
- [→ Deploy & Test](DEPLOYMENT_GUIDE.md)
- [→ Command Reference](COMMANDS.md)

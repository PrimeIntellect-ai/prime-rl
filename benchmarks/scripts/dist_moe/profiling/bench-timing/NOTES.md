# dist-moe-bench-timing

Node primeintellect-tur-r8n2 (8x B300), Slurm job 2760, prime-rl worktree prime-rl-feat-dist-moe at acf7e6354 (feat/dist-moe) plus the uncommitted bench fix. Bench args: --num-experts 128 --top-k 8 --hidden 2048 --intermediate 768, NCCL_MAX_CTAS=16.

- inputs/profile_bench.py: old bench() (back-to-back events) with per-rank event stats printed, plus a 6-step torch.profiler capture (fwd/bwd record_function, ProfilerStep) per bench() call.
- inputs/profile_fixed_bench.py: fixed bench() run under torch.profiler with fwd/bwd annotations (--warmup 3 --iters 3), fwd index 3..5 are GPU-timed, the last 3 are back-to-back.
- traces/t8192, traces/t16384: old scheme captures, all 8 ranks, per arm subdir. traces/t8192_stack: same with Python stacks.
- traces/fixed_t8192, traces/fixed_t16384: fixed scheme captures.
- derived/run_*_profiled.txt: logs of the capture runs (EVENTS lines are per-rank old-scheme event stats, untraced).
- derived/fwd_python_t8192_rank0.txt: Python-level host attribution of one dist_moe fwd (stack trace, inflated).
- derived/fixed_<tokens>_run<n>.txt: untraced fixed bench results.
- Finding: old dist_moe numbers were host launch time (fwd host ~2.3 ms vs 0.93-1.06 ms GPU at 8192 tok), with spin-waiting dist kernels on ranks whose host is ahead masking it as busy time.

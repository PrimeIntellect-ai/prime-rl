# DeepSeek-V4-Flash P/D disaggregation — workstream notes

Started 2026-11-17 on B200 (spawned from the b200-nvfp4 session).

## Layout
- Worktree: `/home/phi/lib/prime-rl-pd` (branch `feat/dsv4-pd`, off origin/main 505b59b28), venv synced (vLLM 0.33.0)
- Here: `configs/`, `scripts/` (hand launch `pd1node.sh`), `logs/<test>/`, `outputs/`
- Node: prime-yotta-nm1-pyrite-gpu-309 (job 66, tmux `pd-node1` on login holds the srun). Peer uses gpu-310.
- SSH to nodes hangs on the forwarded 1Password agent: use `SSH_AUTH_SOCK= ssh <node> ...`.

## Hardware (gpu-309)
- 8x B200 full NVLink (NV18). 8x 400G NDR IB (mlx5_4,7..13, one PIX per GPU) + 4x 100G HDR (mlx5_0..3) + RoCE bond.
- No rdma-core dev headers (verbs.h/rdma_cma.h): scripts/install_nixl_from_source.sh refuses to build as is.
- CUDA 12.9 + 13.0 toolkits in /usr/local; 2 TB RAM; 1.7 TB overlay / + 3.5 TB nvme1n1 (not mounted?).

## Findings
- vLLM 0.33 MXFP8 MoE vs all2all (code + dummy-weight boot, 1 node P DP4 GPUs0-3 / D DP4 GPUs4-7):
  - TRT-LLM MXFP8 experts (baseline kernel) support: no-a2a, ag_rs, deepep_v2, flashinfer_nvlink_one/two_sided. NOT deepep_ht/ll.
  - `deepep_high_throughput` (P/D's auto prefill backend) -> falls back to MARLIN MXFP8 (W8A16). Boots, but slow + different numerics.
  - `deepep_low_latency` (P/D's auto decode backend) -> crash "No MXFP8 MoE backends available" (no batched-format MXFP8 experts).
  - `deepep_v2` needs DeepEP v2 (`deep_ep.ElasticBuffer`); our pinned deep_ep 1.2.1 wheel is v1 only.
  => For roles that fit one node, use flashinfer_nvlink_one_sided (same as the aggregated baseline).
- vllm-router 0.2.4 rejects `reasoning_effort="max"` (enum Low/Medium/High in src/protocols/spec.rs). P/D must go through the
  router, so chat-completions evals with effort=max need a router fix (or another way to pass the effort).
- NIXL connector in vLLM 0.33 has DSV4 support (compressed MLA "dense virtual transfer pages", CSA compressor scratch regions).

## Log
- t1 (dummy, 1 node, P=deepep_ht DP4 / D=deepep_ll DP4): D crashes "No MXFP8 MoE backends available"; P picks MARLIN.
- t2 (dummy, 1 node, both flashinfer_nvlink_one_sided): boots; both FLASHINFER_TRTLLM MxFp8. EP4: 76.9 GiB weights/GPU,
  KV 639K tok (P) / 661K tok (D) per rank at 0.85. First FlashInfer autotune ~10 min for each new config hash.
  pd_smoke.py (greedy, 6 prompts up to 8K tokens): router P/D == direct prefill engine == direct decode engine. PASS.
  NIXL (pip wheel, bundled UCX) NixlPullConnector: 6/6 transfers, 0 failed, ~6.8 KB KV/token (54 MB for 8K tokens).
  Gotcha: `pkill -f vllm` misses `VLLM::EngineCore` (prctl names) -> stale procs held GPUs; use `pkill -9 -f "[V]LLM::"`.
  Gotcha: two DP groups on one node need distinct data_parallel_rpc_port and NIXL side-channel ports.
- Code (worktree, staged, not committed yet: the SSH agent is down so signed commits fail):
  per-role all2all/cudagraph/overrides moved into InferenceConfig.for_pd_role (role engine JSONs); explicit
  inference.vllm.all2all_backend applies to both roles; templates no longer splice ROLE_EXTRA. Docs updated.
- t3: 1-node real weights (configs/pd_1node.toml -> scripts/pd1node.sh), logs/t3_1node_real.
- Job 105: 2-node via the real launcher (configs/pd_2node.toml), prefill gpu-302, decode gpu-303.
- t3 smoke (real weights, 1 node): coherent text; router P/D == direct engines (4/6 prompts identical for 48 tokens, the
  others diverge at 13-26 tokens on both direct engines alike: batch-shape numerics). PASS.
- Evals through P/D need `--client.admin-base-url <each engine>/v1 ...` + `--inference-metrics-roles prefill.. decode..`
  (adaptive concurrency reads engine /metrics; the router has none). Eval: outputs/aime26-pd-1node-64k (gpu-309 tmux eval1).
- Job 107 (HiSparse decode): FAILS at startup on DSV4. V2 runner get_kv_cache_spec -> resolve_hisparse_specs ->
  get_hisparse_kv_cache_groups -> get_kv_cache_groups -> _get_packed_kv_cache_groups reads
  cache_config.get_resolved_kv_cache_layout() before the engine core resolves it (it resolves from the specs). DSV4's
  mixed page sizes hit the packed path (GLM-5's uniform MLA would not). Upstream vLLM 0.33 ordering bug.
  record_kv_cache_layout accepts a pre-set identical layout, so resolving it worker-side first may work (resolved: BLHNC).
- Job 105 (2-node, real launcher): HOSTNAMES order from scontrol can be 303 before 302 -> prefill/router on gpu-303.
  942K KV tok/rank (EP8). Cross-node NIXL with the pip wheel's bundled UCX: works (no segfault so far).
  pd_smoke.py now teacher-forces: router logprobs vs direct-engine scores of the router's tokens, mean |dlp| 0.005-0.05,
  max 0.33 (MXFP8 numerics), ~1e-4 on 5-8K-token prompts. PASS. Eval: outputs/aime26-pd-2node-64k (gpu-303 tmux eval2).
- HiSparse + DSV4: after a worker-side layout fix, vLLM 0.33 raises "HiSparse does not support DeepSeek V4."
  (v1/hisparse/layout.py _partition_hisparse_specs, model_version == deepseek_v4). Not an integration issue: HiSparse
  targets DSA (GLM-5, DSv3.2) sparse MLA. Dropped the patch; docs say DSV4 is unsupported.
- Mooncake job 106: P/D correct, but every put failed: `Failed to register memory ...: Bad address` ->
  AddressNotRegistered. Mooncake's WITH_NVIDIA_PEERMEM defaults to TRUE (ibv_reg_mr on GPU memory), and these nodes
  have no nvidia_peermem (only gdrdrv). WITH_NVIDIA_PEERMEM=0 -> DMA-BUF registration works (mc_reg_test.py).
  Fix in _mooncake_store.sh.j2: export WITH_NVIDIA_PEERMEM=0 unless /sys/module/nvidia_peermem exists. Rerun: job 116.
- AIME26 x16, 64k, effort max (via chat_template_kwargs; prompt tokens identical to baseline: mean 240.17):
  | arm | reward | trunc | out tok mean | wall |
  | aggregated MXFP8 DP8, 1 node (peer, aime26-mxfp8-dp8-64k) | 0.9375 | 5.4% | 22.0K | 19m15s |
  | P/D 1 node: P DP4 GPUs0-3 / D DP4 GPUs4-7 (aime26-pd-1node-64k) | 0.9521 | 4.0% | 21.9K | 21m53s |
  | P/D 2 nodes via launcher: P DP8 gpu-303 / D DP8 gpu-302 (aime26-pd-2node-64k) | 0.9583 | 4.0% | 21.8K | 17m35s |
  2-node: all 251 decode KV-transfer metric intervals report 0 failed transfers (pip-wheel NIXL/UCX, cross-node IB).
  All three within noise (~+-0.02); P/D accuracy matches aggregated. Wall times are a rough throughput signal only
  (one 480-rollout eval with a long tail; the 2-node arm has 2x the GPUs).
- Job 116 (Mooncake + WITH_NVIDIA_PEERMEM=0): registration OK, P/D smoke PASS, but still 0 external hits: master holds
  45 keys; other puts fail "Process failed for slice ... local_nic mlx5_7 -> peer_nic ...@mlx5_0..3" and
  "Cannot make connection ... @mlx5_bond_0". Auto-discovery uses all 13 HCAs; mlx5_0-3 (100G HDR, LIDs 1-6) are
  another IB fabric than the 400G NDR GPU NICs (LIDs ~2500-3500), the bond is RoCE. A prefix with one missing block
  (across the 5 DSV4 KV groups) gives 0 hits. Fix: device_name = NDR NICs (mlx5_4,7..13) in both role offloads -> job 132.
  NIXL's UCX_NET_DEVICES (template: all ACTIVE IB ports) also includes mlx5_0-3; it worked, but the same risk applies.
- Job 132 (Mooncake, device_name = NDR NICs): 0 transfer failures. Cross-rank reuse WORKS: rank B loads 8192/8431
  tokens rank A stored. Load correctness (mooncake_load_check.py, 7K-token high-entropy prompt): B (store load) scored on
  A mean|dlp| 0.09-0.15 vs control "A again" (own GPU prefix hit) 0.09-0.20 -> store loads as accurate as a local hit.
  (The smoke test's ~1e-4 on long prompts was low entropy, not exactness.)
  Decode save (save_decode_cache) does NOT extend hits for DSV4: next turn (1205 prompt + 1000 completion) on another
  prefill rank hits 1024 = prefill-saved prompt blocks only; decode considered 12 keys for ~4 completion blocks vs
  prefill's 26 for 4.7 prompt blocks (5 KV groups). vLLM Mooncake store decode-save + hybrid groups; not fixed.

## Stress test (2026-11-17, user request): 1000+ concurrency, equal compute, PI-DeepSWE + RLM at effort max
- Decisions (user): rollout timeout 24 h (task default 1.5 h) and whole test within 24 h; 2 nodes/arm; arms in parallel.
- Arms (configs/stress/*.toml, all MXFP8 + flashinfer_nvlink_one_sided + VLLM_HTTP_TIMEOUT_KEEP_ALIVE=3600, router from
  templates/ copy: --disable-circuit-breaker --retry-max-retries 3 lenient health, log-level info):
  agg = 2 independent DP8 replicas (job 153); pd = 1P:1D (job 154); pdmc = 1P:1D + prefill Mooncake 1 TB, decode none (job 155).
- Workload (configs/stress/stress_eval.toml): 4 identical pi-deepswe sources (shuffle seeds 0-3), group 16, rlm harness,
  timeout.rollout 86400, effort max via chat_template_kwargs, temperature 1. Concurrency pinned by
  scripts/stress_driver.py through the eval control API (:8045); window rows + level summaries in stress/<run>.jsonl.
- Launch: scripts/run_stress_arm.sh <arm> <run> <schedule>.
- Smoke (8 episodes each, pd + pdmc): sandbox setup ~1 min; ~5 s/turn at low load; effort max verified server-side
  (replayed first call: 5765 prompt tokens = max render, 5752 without/high; scripts/effort_probe.py). Episodes reach
  ~70-120 turns and ~130-160K tokens in ~15 min (reasoning grows to ~700 tok/turn later); prefill GPU hit 93-98%.
- agg arm boot issues: cold nodes (no page cache) load the 569 GB ckpt from NFS at 20-75 s/shard (P/D arms landed on
  warm nodes, ~2.4 s/shard); first agg config lacked enable_expert_parallel -> multi_node template ran 16 single-GPU
  engines -> OOM. Fixed (explicit in all arm configs), resubmitted on the now-warm nodes gpu-318/354 (job 168).
  New external-LB DP8 config hash -> ~15 min FlashInfer autotune.
- KV per arm: agg 920,655 tok/rank x16 = 14.7M; pd/pdmc 942K/rank: prefill 7.5M (holds idle sessions) + decode 7.5M.
- STARTED 04:53 all three (stress-agg on gpu-318/354, stress-pd on 305/340, stress-pdmc on 338/334), schedule
  32:40,64:40,128:45,192:45,256:50,384:60,512:60,768:75,1024:90,1536:90,2048:90 (ends ~16:20).
- 05:58 user asked for high concurrency first: ascending staircase stopped after level 32 (full) + 20 min of 64; the
  drivers restarted on the same evals with a DESCENDING schedule 2048:120,1536:90,1024:90,768:75,512:60,384:60,256:60,
  128:45,64:40 (stress/<run>-desc.jsonl). A lower cap cancels no eval episode, so each lower level drains first
  (phase=drain, <= 60 min) and holds once in flight <= level. Level 32 (ascending, all arms ~155 out tok/s/GPU,
  ~230 prefill computed tok/s/GPU, 96% hit, ITL 11 ms): no difference between arms at low load.
- Level 2048 (05:58-07:58, desc run): agg no thrash (recompute 4-8%, ~2000 running, ITL 135 ms) but out tok/s/GPU
  1590 peak (40 min) sliding to 750 as contexts grow; pd ~2.6-3.0K out/GPU for 30 min, then THRASH at ~40 min: prefill
  hit 0.93 -> 0, recompute 100%, ~2000 waiting, TTFT ~460 s, ~150 out/GPU, never recovers; pdmc thrashes at ~40 min
  too but Mooncake hits (ext 0.2-0.6) hold it at 860-1760 out/GPU (TTFT 17-65 s). Level means (2nd half): agg 980,
  pd 165, pdmc 1225 out tok/s/GPU. Episodes done in 2 h: agg 352, pd 314, pdmc 747.
- Draining an eval is too slow (no cancellation; agg ~3 episodes/min, pd ~0), so from 08:11 each lower level is a
  fresh eval burst stress-<arm>-L<level> (scripts/level_loop.sh, tmux loop-<arm> on login): 1536,1024,768,512 x 75 min,
  384,256,128 x 60 min. Stop detection waits on PRL::Eval (the eval's proctitle; pgrep on run.name only finds the
  tmux server). After stopping, engines drained orphaned requests within ~10 min (queues 0 at 08:11).
- 08:11 loop bug: the driver-liveness check piped ssh through `grep -v Warning` under pipefail, so empty output read as
  "driver gone" and every level was skipped in ~8 min (no data; runs moved to outputs/aborted). Fixed (echo alive |
  grep -q) and tested on a node; loops relaunched 08:23, level 1536 pinned in all arms.
- Stress results (second-half means, out tok/s/GPU agg/pd/pdmc): 2048 980/165(thrash)/1225; 1536 1213/1872/1673;
  1024 967/1696/1673; 768 853/1495/1496 (cut at ~70 min); 32 156/158/154. pd burst point between 1536 and 2048
  (~40 min into 2048, prefill cache overflow -> 100% recompute); Mooncake rescues pdmc at 2048; agg never thrashes but
  is decode-bound (ITL ~2x P/D). Levels <= 512 not run: jobs 154/155/168 CANCELLED by root 12:10 (cluster full after).
  Loop bug #2 at 12:10: the cancelled nodes made ssh fail -> "driver session gone" -> remaining levels skipped.
- PR #4565 (draft): https://github.com/PrimeIntellect-ai/prime-rl/pull/4565 — 4 signed commits on main ea848cc38
  (per-role P/D settings; Mooncake WITH_NVIDIA_PEERMEM=0; HiSparse/DSV4 docs; configs/advanced/deepseek-v4-flash/pd.toml).
  Body in PR_4565_body.md. Pushed with the live forwarded agent /tmp/ssh-w72OAz0C30/agent.1242019 (~/.ssh/agent.sock
  symlink was stale).
- 18:00 resume (user: keep going, update PR #4565 as results come in): cluster full (75 alloc, 1 idle); run-1 outputs
  moved to outputs/stress_<arm>_run1; arms resubmitted as jobs 344/345/346 (queued); tmux resume-stress on login waits
  for all three (16/16 engines) and starts level loops 512:75,384:60,256:60,128:60. level_loop.sh now retries on ssh
  failure (only an explicit "gone" ends a level) and exits if its inference job ended.
  Branch rebased on main ea848cc38: only prime-kernels trainer wheels changed in uv.lock (inference stack unchanged).
- 18:42 user: keep usage to the 4 running nodes -> pdmc job 346 cancelled before it started; levels 512-128 run only
  agg (job 344, gpu-358 router / 306) and pd (job 345, gpu-299 router / 362). Loops started pd 18:33, agg 18:41
  (each arm starts when up). pdmc ~= pd below the thrash point (Mooncake <1% of prompt tokens at <= 1536).
- Level 512 (run 2, agg + pd only): agg 633 / pd 1083 out tok/s/GPU (1.71x), calls 13.6 / 22.9, recompute 3.0% / 2.6%,
  no thrash. Episodes done: agg 112, pd 328. Report artifact https://claude.ai/artifact/XSqsFynwxqbRZo3aKZBmj5
  (report/template.html + scripts/build_report.py -> report/index.html; republish same path). PR tables:
  scripts/pr_results.py PR_4565_body.md && gh pr edit 4565 --body-file PR_4565_body.md.
- Level 384: agg 556 / pd 898 out tok/s/GPU (1.62x), calls 12.0 / 19.1, recompute 2.9% / 2.7%; done 70 / 201.

## Tuning (user, 21:30: "after the current runs, do the tuning"; stay on gpu-299/362/306/358)
- Diagnosis from stress data: pd 1:1 at 1536 is PREFILL-bound (prefill waiting ~147, ~14.7K computed tok/s per prefill
  GPU; decode waiting ~4, ~129 running/rank, ITL 34 ms). pdmc's store writes add prefill work (waiting ~382).
- Plan (tmux `tune` on login, scripts/tune_kickoff.sh -> tune_round.sh; configs/tune/*.toml; runs tune-<cfg>-L<level>):
  R1 (2+2 nodes, parallel, 1536:75): pd_mbt32k (prefill max_num_batched_tokens 32768, default 16384) on gpu-[299,362],
      agg_mbt32k (32768) on gpu-[306,358]; compare with stress 1536 (pd 1872, agg 1213 out tok/s/GPU).
  R2 (4 nodes, 3072:75): pd3p1d = 3P:1D (+ R1's better prefill chunk).  R3 (4 nodes, 3072:75): agg4 = 4x aggregated.
  R4 (stretch): pd3p1d at 4096 (thrash point with 3x prefill cache).
  3072 on 32 GPUs = 1536 on 16 GPUs per GPU. Kickoff waits for the stress loops (128 ends ~23:00), cancels 344/345,
  resubmits pinned with --nodelist.
- 21:35 user: grab free nodes (59 idle). All rounds launched in parallel on new nodes (scripts/tune_round.sh, tmux
  tune-r1 / tune-r23, logs/tune_r1.log, tune_r23.log): R1 1536:75 pd_mbt32k (job 387, gpu-343/349) + agg_mbt32k
  (388, gpu-318/320); R2+R3 3072:75 then 4096:75, pd3p1d (385, gpu-298/305/338/340) + agg4 (386, gpu-316/335/346/356),
  default chunk (ratio effect isolated). Stress jobs 344/345 released by tmux release-stress after level 128.
  Peak sandboxes ~9-11K (R2/R3 at 4096 + R1) -- watch for sandbox creation errors.
- 21:40 R1 failed at startup: HF Hub 429 (rate limit) on the model file listing with ~100 engines starting and no HF_TOKEN in the job env (my shell had not sourced ~/.env). Fix: HF_HUB_OFFLINE=1 in configs/tune/*.toml + tune_round.sh sources ~/.env. R2/R3 survived (past hub calls). R1 relaunched.
- Level 256: agg 454 / pd 702 out tok/s/GPU (1.55x), calls 9.8 / 13.6; done 87 / 179.
- Level 128: agg 330 / pd 435 out tok/s/GPU (1.32x), calls 5.8 / 7.3. Stress staircase complete (agg, pd: 32-2048;
  pdmc: 32, 768-2048). Jobs 344/345 released 23:04.
- 23:05 tuning mishaps: (1) editing tune_round.sh at 21:41 while R2/R3's copy ran shifted bash's read offset -> it hung
  in a stray `grep -q .` from 22:05 (engines idle ~1 h). (2) My replacement tune_levels.sh had
  `squeue ... | grep -q . </dev/null` (grep read /dev/null) -> every level "ended" -> it cancelled jobs 385/386.
  Fixed; R2/R3 resubmitted 23:06 on the same nodes (jobs 397/398) with tune_round.sh (unchanged since 21:41).
  Rule: never edit a script a running process executes; write a new file.
- R1 (1536, 16 GPUs): 32K step size is worse for both: agg 1,085 vs 1,213 out tok/s/GPU; pd THRASHED at ~25-35 min
  (200 out/GPU, 99.9% recompute, ~1,476 prefill waiting) where the 16K run oscillated (723/571/419 queue spikes) and
  recovered: 1P:1D is metastable at 1536 on 16 GPUs. Best step size: default 16K for both.
- 23:24-23:26: agg4 job 398 never started (gpu-335 drained); resubmitted as 400 on gpu-[306,316,346,356].
  P/D 3P:1D + NVFP4 routed experts (configs/tune/pd3p1d_nvfp4.toml, exp worktree /home/phi/lib/prime-rl-pd-exp =
  feat/dsv4-pd + merge of feat/dsv4-nvfp4-inference) job 399 on gpu-[312,314,319,341]. One starter (tmux tune-4n,
  scripts/levels_when_up.sh) runs 3072:75 then 4096:75 for pd3p1d=397, agg4=400, pd3p1d_nvfp4=399 together.
- exp branch exp/dsv4-pd-stress: merge staged + configs/experiments/dsv4-pd/ staged; scripts/commit_exp.sh commits
  (signed) and pushes once an SSH agent is live (none since ~23:10).
- 23:51 NVFP4 job 399 FAILED at startup: gpu-319 GPU 7 not visible ('No CUDA GPUs are available'). R2/R3 (397/400) levels started now via tune_levels.sh; NVFP4 resubmitted as job 404 (excluding 319, 335), its levels start when up (~40 min offset).
- 00:20 user: tuning table must compare P/D ratios (1P:1D vs 3P:1D), PD > agg is known. Added pd2p2d (4 nodes, two 1:1 pairs) job 410, levels 3072:75,4096:75 (tmux tune-2p2d); agg4 to be stopped after 3072. Exp PR #4574 (draft, exp/dsv4-pd-stress -> feat/dsv4-pd).
- 01:10 agg4 job 400 cancelled after its 3072 level (PD > agg already established; frees sandboxes for 2P:2D).
- Level 3072 on 32 GPUs (96 episodes/GPU, second half of 75 min): 3P:1D MXFP8 796 out tok/s/GPU -- DECODE-bound
  (decode waiting 728, ITL 89 ms, TTFT 23 s; prefill idle), fell from 1,690 to ~750 after 20 min; agg4 1,230;
  2P:2D (1:1) 2,324 at 32 min (running); 3P:1D NVFP4 1,049 at 52 min (decode-bound too, NVFP4 decode faster).
  -> 1:1 is the best ratio for this workload; 3:1 overshoots. Sandbox-side failures: pd3p1d 446, agg4 314 episodes
  failed at setup ("prime egress policy failed: 429 per-IP limit", "VM sandbox creation burst limit"), slots refilled.
- 01:11 P/D + NVFP4 "best settings" = 2P:2D: pd2p2d_nvfp4 job 437 (gpu-306/316/346/356, warm), 3072:75,4096:75 (tmux
  tune-2p2d-fp4). gpu-319 removed from Slurm (exclude of it now fails "Invalid node name"); agg4 job 400 cancelled.
- 01:34 NVFP4 3P:1D job 404 cancelled after 3072 (3:1 is decode-bound; NVFP4 continues as 2P:2D, job 437).
- Level 3072 (96 episodes/GPU): 2P:2D MXFP8 1,966 out tok/s/GPU (2-node 1P:1D 1,872: scales linearly), calls 40.1 per
  16 GPUs, prefill waiting 211 (spike 476 at 50 min, recovered), decode 11; vs 3P:1D 796 (decode-bound), 3P:1D NVFP4
  820 (decode-bound; decode waiting 550 vs 728), agg 1,230. Best ratio so far 1:1; the optimum sits between 1:1 and 3:1.
- 01:54 pd3p2d (3P:2D, 5 nodes) job 447 on gpu-[310,326,343,358,370], level 3840:75 (= 96/GPU), tmux tune-3p2d.
- 3P:1D @4096 (128/GPU): 917 out tok/s/GPU, decode-bound (decode waiting ~1,450 by the end), recompute 4.3%: three prefill nodes hold 4096 sessions without the prefill thrash 1P:1D hit at 128/GPU. 2P:2D @4096 at 33 min: ~2,555, prefill waiting 138.
- 02:47 user: stop 3P:2D (job 447 cancelled at ~37 min of 3840: 1,481 out/GPU at 33 min, partial); still run 1P:3D; test bidirectional KV (router PR #76, prime-rl PR #4575).
- 02:47 user: max 8 B200 nodes across this session + b200-nvfp4. Were at 12 (mine 8: 410, 437; peer 4: 180, 181, 408, 415). Cancelled 437 (NVFP4 2P:2D; its data since ~02:24 is compromised by InterceptionError/404 from its deployment).

## 2026-11-18 03:00 — Bidirectional KV (prime-rl PR #4575, vllm-router 0.2.6 / router PR #76)

What it does: NIXL `bidirectional_kv_xfer`. Decode keeps a finished turn's KV blocks (prompt + completion) for
`decoder_kv_blocks_ttl` (480 s) and returns a handle; the router stores it per `x-session-id` (TTL 450 s) and sends it
with the session's next turn to prefill, which pulls those blocks from decode over NIXL and computes only the new
tokens. Single-use: the router drops the handle when a turn starts, so a retry runs a plain prefill. Prefill runs
with `kv_load_failure_policy = "recompute"`. Only token-in `/inference/v1/generate` turns (RL path) carry handles;
chat completions never do, so the DeepSWE eval runs token-in here (`PRL_EVAL_TOKEN_IN=1`, a runner.py patch in
worktree prime-rl-bidir: the eval uses the renderer TrainClient, DeepSeekV4 renderer, thinking, effort max).

Setup: worktree /home/phi/lib/prime-rl-bidir (branch exp/dsv4-pd-bidir off origin/feat/pd-bidirectional-kv 0c2082cf4,
vLLM 0.34). Templates copied to templates-bidir/ with our router flags. NIXL: pip wheel (bundled UCX). The PR's
source build (UCX 1.19) cannot run here: compute nodes lack rdma-core headers and autotools; the pip wheel ran the
whole stress test with 0 failed transfers. Watch for `signal 11` in libucs.

Plan (4 nodes, cap agreed with b200-nvfp4): 1P:3D bidirectional, then 1P:3D one-way, each at 4096 in flight (128
episodes/GPU, the 2P2D/3P1D tuning level) for 75 min. configs/bidir/, scripts/bidir_round.sh + bidir_arm.sh.

### 1P:3D bidirectional at 4096 (job 461, 03:42–04:57)

- 0–35 min: ~99% of follow-up turns read decode KV; prefill computes ~720–820 tokens per turn (one-way 2P2D, chat
  path: ~1,900–2,500), output peaks ~3,900 tok/s/GPU at 12–20 min. Contexts grow to ~85K tokens (chat-path runs: 40–65K;
  the token-in renderer keeps reasoning in history when tools are present).
- 35–50 min: collapse. Router prefix mismatches 1% -> 7–10% of turns; each one is a full prefill of the whole prompt
  (median 85K tokens), since the handle is all-or-nothing and prefill's own cache holds nothing (GPU hit 0). Prefill
  GPUs 100% busy, queue 3.6K, TTFT ~170 s, output ~640–1,070 tok/s/GPU. NIXL: 0 failed transfers, D->P reads 0.4–1 s
  for ~340 MB (P->D pulls ~70 ms); not the bottleneck.
- Mismatches from the traces (scripts/branch_analysis.py, branch_detail.py; 400 traces, 436 mismatches):
  - 42%: RLM compaction (new prompt ~10% of the old path, history replaced after the first user message). Cheap.
  - 54%: same-length prompt that diverges at an earlier assistant message (keeps 10–90% of the old path): the
    harness changed something, the renderer could not bridge, and the full re-render of an older assistant turn
    differs from its sampled tokens. One-way prefix caching would still reuse up to the divergence; bidirectional
    recomputes all of it.
  - rate by minute of the run: 0.0% (0–10), 0.2%, 0.7%, 2.4% (30–40), 7.4% (40–50), 6.9% (50–60).
- Root cause of the expensive mismatches (scripts/branch_tail.py, branch_diff.py): 215 of 277 same-length mismatches
  are RLM's compaction request ("Create a concise continuation summary...") at ~115K tokens (131072 - 16K), 52 are
  harness `<runtime_event>` notices. The renderer cannot bridge those turns and fully re-renders the conversation,
  and the DeepSeek-V4 renderer does not round-trip an assistant turn with text + tool calls: the model sampled
  `...details.\n\n<｜DSML｜tool_calls>`, the parsed content keeps the trailing `\n\n`, and the render adds its own
  separator (`emit_text(content)` then `"\n\n<｜DSML｜tool_calls>..."`, renderers/deepseek_v4.py ~L699-711), giving
  `\n\n\n\n`. The prompt diverges at the first such assistant turn (often early), so the whole ~115K prompt is
  recomputed under bidirectional transfer (one-way prefix caching reuses up to the divergence). renderers 9f4943f.

### 1P:3D one-way at 4096 (job 482, 06:55–07:37), same branch, token-in
Thrashed at ~25 min (prefill GPU hit 0.43 -> 0.02, 4K queued, TTFT 275 s, 81% recompute); stopped at 41 min.
Output tok/s/GPU one-way vs bidirectional: 0–15 min 1,437 vs 1,894; 15–25 1,849 vs 3,815; 25–35 337 vs 3,383;
35–41 188 vs 2,853. Bidirectional moves the 1P:3D thrash point from ~25 min to ~45 min, where the renderer
round-trip mismatches take over.

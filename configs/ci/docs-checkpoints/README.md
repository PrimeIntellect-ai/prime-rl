# Documented checkpoint startup smoke coverage

These TOMLs are recipe inputs, not safe direct-launch commands: Qwen30B has no source step cap, and other source caps are50/100/10/4. Use only the manual workflow, which requires an explicit positive finite cap and prepares a temporary config. Live resource approval and listed prerequisites are still pending. Scheduled docs-checkpoint coverage is NOT enabled.

These are hosted training recipes copied from public-docs `90ad2fbfb1d487fcb8129a385e9eff3fbd362e2d`, with explicit current-schema migrations below. They are not all91 TOML fences as independent jobs. Startup acceptance is observed first hosted progress, not merely run creation, allocated pods, convergence or a completed training run.

| Case | Exact checkpoint | Docs source | GPUs | Prerequisites |
| --- | --- | --- | --- | --- |
| reverse-text-rl | PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT | hosted-training/full-finetuning:44; getting-started:68 | 1train+1infer | cached checkpoint |
| alphabet-sort-rl | Qwen/Qwen3-4B-Instruct-2507 | hosted-training/end-to-end-run:76 | 1train+1infer | cached checkpoint; alphabet-sort taskset |
| qwen30b-math-rl | Qwen/Qwen3-30B-A3B-Thinking-2507 | hosted-training/full-finetuning:87 | 2train+2infer nodes×8=32 | exact Thinking checkpoint cache; i3_math taskset |
| reverse-text-sft | PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT | hosted-training/hosted-sft:67 | 1train | volume dataset datasets/reverse-text from documented willcb/R1-reverse-wikipedia-paragraphs-v1-1000 |
| glm53-sft | zai-org/GLM-5.3-BF16 | hosted-training/hosted-sft:256 | 8train nodes×8H200=64 | volume dataset datasets/intellect-3-sft-10k math split; source not supplied in docs |

Source links resolve under https://github.com/PrimeIntellect-ai/public-docs/blob/90ad2fbfb1d487fcb8129a385e9eff3fbd362e2d/ . Reverse SFT/RL share a checkpoint but exercise different training modes.

## Current-schema migrations

- reverse RL: move orchestrator.group_size to orchestrator.train.group_size.
- alphabet RL: default harness was renamed bash; no null-harness substitution.
- GLM SFT: remove obsolete model.impl; use model.offload="optimizer" instead of model.optim_cpu_offload=true. Exact model, dtype and documented topology are preserved.
- Hosted multi-node validation uses RLK8sConfig/SFTK8sConfig, not the local SLURM launcher. Its preprocessing supplies the ignored SLURM block. A plain local RLConfig SLURM error does not establish hosted incompatibility.

Literal docs recipes are not claimed schema-valid. Optional volume/resume/SFT-eval/topology overlays and placeholder /volume/models/glm-5.3-hf are not extra pinned checkpoints. The comprehensive checkpoint catalog records guide/reference training IDs and missing prerequisites separately.

Queue wait is not trainer startup failure. Existing runs have queued over14hours. Any bounded startup observer must report queued/startup-unverified on expiry; it does not stop or resubmit hosted work.

## Manual-only workflow

`docs-checkpoint-startup.yaml` has no schedule. Operators must supply a positive finite `steps` value (2recommended, no default), published image tag and explicit observation budget within the existing30minute job cap. SFT requires an existing prepared volume. It checks that the SFT volume exists before dispatch and never intentionally provisions one, stops training or retries run creation. The CLI can auto-create a volume if it is deleted between that check and dispatch; operators must not delete the selected volume during a run. Matrix submissions are serial, but an observer may exit while finite training continues, so serial jobs do not guarantee released prior allocation;101GPUs remains the all-case allocation ceiling.

Pass means exact runId/name, RUNNING or COMPLETED, and uploaded hosted progress latest_step>=1. RL progress can originate from orchestrator rollouts, not independent trainer optimizer acknowledgement. This startup smoke does not prove optimizer completion, reward convergence, full completion, checkpoint durability or online-eval coverage. FAILED/STOPPED never pass. Queue/API timeout is startup-unverified and leaves hosted work untouched.

The required observation budget has no default and starts after setup/submission; the overall30minute runner cap may interrupt it sooner. Polling is30seconds; the Actions job cap remains30minutes. If the job is cancelled or reaches its runner cap, training can continue; use saved names/IDs to reconcile before any resubmit. Exact Thinking checkpoint cache and GLM dataset-source blockers must be resolved before an all-case run can pass. No live jobs were executed during PR preparation.

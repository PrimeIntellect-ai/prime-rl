# NGU preparation

Status: design and source review, not an implementation or a launched experiment. Prepared 2026-09-17.

Recommendation: implement binary-reward NGU as a group lifecycle plus a named credit-assignment algorithm. Make the adoption decision with one large PrimeIntellect/GLM-4.5-Air-Scaleswe baseline on a new SWE taskset (provisionally SWE-rebench-V2) and one matching NGU run, evaluated on SWE-Bench Verified at equal H200-hours. Preserve prime-rl's trainer loss across all comparison arms; this tests NGU in prime-rl rather than claiming an exact reproduction of Open-Instruct.

## Sources and confidence

- [Paper, v1](https://arxiv.org/html/2609.13443v1), especially sections 3–6 and appendices C–E; [PDF](https://arxiv.org/pdf/2609.13443).
- [Authors' code snapshot](https://github.com/mnoukhov/never-give-up/tree/96e70fd001af0dc710136be66300b2b795f63bbb), especially `open_instruct/data_loader.py` and `data_loader_utils.py`.
- The supplied validation transcript is useful secondary context. Reported scores below agree with its tables; qualifications below affect how to interpret them.
- Repository architecture was inspected at this worktree. Dependency directories are empty; available tasksets were inspected remotely at the pinned revisions: verifiers `28f0893`, prime-envs `1f1e050`.

The paper is CC BY-NC-SA 4.0, by Michael Noukhovitch, Hamish Ivison, Nathan Lambert, and Aaron Courville. This document paraphrases its method and distinguishes our proposed implementation choices.

## Method, in detail

NGU allocates **training rollout compute**. It is not a test-time search algorithm and does not lengthen a single completion. A retry starts another independent completion of the same task under the current policy; it does not feed the previous failed answer back to the model.

For binary reward, ordinary centered GRPO assigns `A_i = r_i - mean(r)`. With success probability `q`, a fixed group of K independent samples has nonzero reward variation with probability

`1 - q^K - (1-q)^K`.

Thus increasing K admits both rare successes on hard tasks and rare failures on easy tasks. It also reduces prompt diversity at fixed completion count. Active sampling replaces zero-signal groups, so fixing the number of *trained* completions does not fix the number of *generated* completions or GPU-hours.

NGU processes a logical task visit in rounds of K completions:

1. Sample and score one round.
2. If the visit has informative successes and failures, finish it and train on the usable cohort.
3. A fresh all-correct group is filtered; future independent visits to that task remain possible.
4. If all attempts failed, continue this visit with probability p; otherwise discard it and draw another task.
5. Requeue the continuation without blocking other visits or weight updates. Accumulate reward statistics over the visit, retain usable completion payloads, and repeat.

The order matters when a later round is all-correct: earlier usable failures can still make the combined cohort informative. Do not filter based solely on the latest round.

For an always-unsolved task, expected rounds are `1/(1-p)`, expected completions `K/(1-p)`. At a stationary success rate q, the simplified independent-round expectation is

`E[completions] = K / (1 - p(1-q)^K)`.

The values 32/64/128 quoted for K=16 and p=.5/.75/.875 are therefore failure-only expectations, not matched actual budgets. Policy updates and variable completion lengths further separate these from compute.

### Stale payloads versus historical counts

Keep two distinct things:

- Recent payloads: tokens, behavior logprobs, reward, and actual policy provenance, eligible for training only within a specified age.
- Lifetime statistics for this visit: valid completion count C and reward sum S, even after old payloads expire.

The historical baseline is `b = S/C`. With n+ retained positive and n- retained negative completions, positive anchoring gives

`A+ = 1-b`

`A- = -(n+/n-) * (1-b)`.

The negative sign is essential. Then `n+ A+ + n- A- = 0`. If every sample is retained this reduces to ordinary centered GRPO. If failures are removed, positives retain the large advantage justified by the full history while the remaining negatives absorb the balancing mass.

Example: 20 attempts, one success; only that success and three failures are fresh. b=.05, so advantages are `[.95, -.3166667, -.3166667, -.3166667]`. Re-centering only the retained four gives `[.75, -.25, -.25, -.25]` and loses the history-dependent scale.

An all-positive retained cohort with only stale failures cannot both preserve positive advantages and have zero sum. This needs an explicit policy; our binary implementation should drop and count that cohort. The reference rescaling helper leaves groups without a balancing set unchanged, so our proposed choice is a documented deviation, not a paper requirement.

Do not divide by reward standard deviation afterward: that can erase the intended anchoring scale. Group-level zero sum is not token-weighted zero sum when response lengths differ.

### Coding / partial rewards are a separate extension

The supplied transcript's “wait until a correct completion” explanation fits binary math, but is incomplete for Manufactoria. The paper describes sampling for improvement over previous solutions. In the inspected reference code, a first round with reward variation is accepted. Once a retry chain exists, the default `better` mode requires its new maximum reward to exceed the previous best. The code also provides a `different` alternative.

The reference anchoring helper uses max-reward samples as anchors, including ties, and rescales the complement. That is not automatically equivalent to the binary formula above. Stage 1 should reject nonbinary task rewards and multi-agent credit assignments, rather than silently treating a partially correct program as binary success. Both proposed SWE runs use unshaped binary solved rewards, with no length penalty. A later coding extension needs separately specified improvement, reward-bound, and anchoring semantics.

## What the experiments establish

| Experiment | Comparison and controls | What was measured | Interpretation |
|---|---|---|---|
| GSM8K Platinum, Qwen2.5-0.5B-Instruct | K=4/8/16/32, corresponding N=64/32/16/8 in main text; asynchronous centered GRPO, active replacement, three seeds | Overall and four difficulty-specific pass@1 curves; training-batch composition; steps | Small K can beat large K even on hard problems; NGU p=.95 improves the allocation tradeoff |
| GSM8K ablations | p=1; sync versus async; downsampling versus positive anchoring; retained ages 1/4/8/16 | Per-difficulty pass@1, mean and SD | Full p=.75 configuration: extra-hard 87.9±2.2 versus GRPO 47.1±6.4; p=1 is particularly bad; age 4 beats 1, 8 and 16 in this setup |
| DeepScaler, Qwen3-4B-Base | Random 10k training subset; K=16/32/64 at 128 nominal completions; NGU K=16 and p=.5/.75/.875; three seeds; approximately 120 H100-hours per run | AIME25 and BRUMO25 pass@1; difficulty-specific gains; composition | Best reported NGU average 26.5±.6 versus best fixed-K 25.3±.5; hard gain 4.3±1.2 versus 2.5±1.1 for K=32 |
| Static difficulty curriculum | Initial 32-sample estimate assigns larger K to hard prompts | Same difficulty-specific gains | Hard gain 4.0±1.1, total gain 12.1±.8; supports online allocation but does not establish superiority over every curriculum |
| Manufactoria, Qwen3-4B-Instruct-2507 | GRPO per-test reward versus NGU p=.95; 14–30 tests/problem; additional shared-checkpoint comparison after 3000 GRPO steps | Mean test pass rate, hard-test pass rate, full-problem success, compute-time curves | NGU escapes the partial-test plateau; roughly matches a switch to all-tests reward from the same checkpoint |

The difficulty diagnosis is observational, not proof of a single cause. The ablations support the mechanism, but dynamic allocation, variable effective group size and advantage scaling interact. Three seeds and small contest test sets warrant caution about the modest overall math gain.

### Reproduction caveats to resolve

- GSM8K main text fixes N*K=256, but appendix D.1 lists N=32, K=16 (512). The public launch defaults also use 512. Do not label those defaults a reproduction of the plotted main sweep.
- GSM8K describes selecting eight samples corresponding to four difficulty levels; the exact per-bin construction and train/eval separation should be recovered from dataset manifests. Do not treat the extra-hard ablation as a broad held-out GSM8K estimate without that check.
- Appendix D.2 specifies learning rate 1e-6, temperature 1, response length 8192, async steps 4, no reference KL, centered advantages. Initial difficulty sampling uses temperature .7/top-p .8, whereas final evaluation uses 1/1: those are different distributions.
- The authors' current DeepScaler launch defaults use Qwen3-4B-Thinking-2507, response length 16384, K=8, N=32, async steps 2 and extra evals. They conflict with the reported Base-model experiment. Pin a source version and explicit overrides; do not copy the launch script uncritically.
- Appendix E's inclusive age comparison and the text's “less than T” / T=1 interpretation leave an indexing ambiguity. Define our age in policy versions and report the mapping.
- The paper calls the aggregate an average, but the base row 7.63/16.3/12.6 is not their unweighted arithmetic mean. Recover dataset weighting; our experiment will report both equal-benchmark macro and pooled-prompt scores.
- Empirical zero successes in 64 or 1024 samples does not imply zero true success probability. Use “0 observed successes,” preserve counts, and report bucket sizes.
- “Does not permanently alter the dataset” does not mean unchanged training distribution: NGU deliberately changes exposure and accepted-cohort composition.

See [stage 1](stage-1.md) and [stage 2](stage-2.md) for the implementation and experiment proposals.

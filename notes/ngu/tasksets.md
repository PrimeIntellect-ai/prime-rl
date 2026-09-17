# Available SWE tasksets for the NGU comparison

Inspected prime-envs at this worktree's pinned revision `1f1e050ab0cd273bca39eed5c3e5315e6a8ae9d1`, including per-taskset README, config and loader. [Catalog source](https://github.com/PrimeIntellect-ai/prime-envs/blob/1f1e050ab0cd273bca39eed5c3e5315e6a8ae9d1/environments/swe/README.md).

“Available” here means implemented in the pinned dependency. The local submodule is empty, so these are not verified installed imports, live registry checks or completed sandbox smokes. Counts and validation/image status below are reported by that source snapshot; usable counts after filtering may differ.

## Training candidates

| Taskset ID | Documented size | Coverage / dataset | Readiness and fit |
|---|---:|---|---|
| `swerebench-v2` | 6,275 | Multilingual; `PrimeIntellect/SWE-rebench-V2-Filtered-Verified`, train | Recommended provisional choice: binary all-tests reward, documented Prime images and gold/no-op validation, raw-row filtering. Use full verified set, not the easy slice. |
| `r2e-gym` | 4,522 | Python; `PrimeIntellect/R2E-Gym-Subset-Verified`, train | Strong simpler alternative, already used by other repo SWE recipes. Documented images and validation. Binary agreement with gold test outcomes; tests hidden during solving. |
| `swelego` | 4,323 | Python real GitHub issues; `PrimeIntellect/SWE-Lego-Real-Data-Verified`, resolved | Documented images and validation; binary all-tests reward. Another suitable single-source comparison. |
| `multiswe` | 2,232 | C/C++, Go, Java, JS/TS, Rust; `PrimeIntellect/Multi-SWE-RL-Verified`, train | Documented images and validation; useful for a deliberate non-Python shift, but Python-only SWE-Bench Verified would measure transfer rather than broad coverage. |
| `openswe` | 36,884 | Python; `GAIR/OpenSWE`, openswe_oss.jsonl | Much larger corpus; images documented, dataset gated, catalog does not claim complete validation. Would first verify access and reward reliability. |
| `swesmith-env` | 88,130 source tasks | Eight languages; `SWE-bench/SWE-smith-*`, train | Largest source pool. Images documented; catalog does not claim full validation. Loader warns/skips source tasks whose profiles are missing; actual usable count must be measured. |

SWE-smith uses distribution name `swesmith-env` and import module `swesmith_env`; confirm discovery after dependency installation.

## Prefer to keep these as evaluation

| Taskset ID | Size | Notes |
|---|---:|---|
| `swebench-verified` | 500 | Existing primary eval. Prime images documented. Keep out of training. |
| `deep-swe` | 113 | Original long-horizon Python/Go/JS/TS/Rust tasks. Images and validation documented. Requires Prime VM and a separate pristine verifier container; committed-patch semantics need harness support. |
| `senior-swe-bench` | 50 | Investigation/design tasks. Images documented. Design reward requires upstream judge/validation-agent settings; default judges-off is unsuitable for meaningful design-task comparison. |
| `swebench-multilingual` | 300 | Multilingual held-out benchmark; implemented via Harbor. Catalog says Prime images are not yet mirrored. |
| `swebench-pro` | 731 | Harder Python/Go/JS/TS benchmark; implemented via Harbor. Catalog says Prime images are not yet mirrored. |

ScaleSWE is implemented too, but excluded from training by user instruction because the selected starting model already trained on it.

## Recommendation and overlap

Use SWE-rebench-V2 provisionally in both draft configs. It gives a different multilingual source with a manageable verified task count and binary rewards suited to initial NGU. This is an engineering recommendation, not evidence that its tasks are harder for this checkpoint. R2E-Gym is the closest alternative if retaining a Python-only train/eval distribution is preferred.

A distinct corpus is not proof of fresh tasks. Before committing the run manifest, compare candidate task identities against the prior ScaleSWE training manifest and SWE-Bench Verified, using normalized repository, issue/PR and commit/patch information. The model's broader pretraining exposure cannot be inferred from this comparison. Do not claim the corpus is unseen or overlap-free yet.

If training multilingual SWE-rebench-V2, retain per-language training allocation/success metrics. SWE-Bench Verified remains the agreed primary measure but covers Python transfer, not multilingual performance. A held-out slice of the new corpus is an optional diagnostic, not an additional training run or selection stage.

## Sources

- [SWE-rebench-V2 README](https://github.com/PrimeIntellect-ai/prime-envs/blob/1f1e050ab0cd273bca39eed5c3e5315e6a8ae9d1/environments/swe/swerebench_v2/README.md), [dataset](https://huggingface.co/datasets/PrimeIntellect/SWE-rebench-V2-Filtered-Verified).
- [R2E-Gym README](https://github.com/PrimeIntellect-ai/prime-envs/blob/1f1e050ab0cd273bca39eed5c3e5315e6a8ae9d1/environments/swe/r2e_gym/README.md), [dataset](https://huggingface.co/datasets/PrimeIntellect/R2E-Gym-Subset-Verified).
- Other taskset README/config paths are linked from the pinned catalog above.

# Omni-MATH-2 Stratified Samples

Dataset: `martheballon/Omni-MATH-2` split `train`.
Revision: `1c10fd492252173c73468badf6dc1804225eb5bb`. Fingerprint: `2b42505da9ff316a`.
Population rows: 4428. Seed: 42.

Sampling strata: `primary_domain × difficulty × source_bucket`, where `source_bucket` keeps the top sources separate and folds the tail into `other`.
Sample relationship: `separate_stratified_draws_not_nested`. Each sample manifest entry includes the realized allocation and fractional quota for every stratum.

## Samples

### n=500

File: `benchmarks/datasets/omni_math2/omni_math2_stratified_500_seed42.jsonl`
Unique strata selected: 197 / 439.

| marginal | max abs share error | total variation |
|---|---:|---:|
| primary_domain | 0.0093 | 0.0126 |
| difficulty | 0.0075 | 0.0263 |
| source_bucket | 0.0069 | 0.0151 |
| has_proof_tag | 0.0100 | 0.0100 |
| has_image_tag | 0.0042 | 0.0042 |

### n=600

File: `benchmarks/datasets/omni_math2/omni_math2_stratified_600_seed42.jsonl`
Unique strata selected: 215 / 439.

| marginal | max abs share error | total variation |
|---|---:|---:|
| primary_domain | 0.0096 | 0.0140 |
| difficulty | 0.0057 | 0.0211 |
| source_bucket | 0.0047 | 0.0103 |
| has_proof_tag | 0.0043 | 0.0043 |
| has_image_tag | 0.0071 | 0.0071 |

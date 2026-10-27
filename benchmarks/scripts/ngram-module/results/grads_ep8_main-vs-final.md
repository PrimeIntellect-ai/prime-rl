# Grad norms: ep8

- main r0: `/home/garrett/tmp/ngram-module/drivers/final-ab/main/ep8/grads_r0.json` (main at `1810d5b9a`, dirty=False, optimizer step=False, replica shards not run)
- main r1: `/home/garrett/tmp/ngram-module/drivers/final-ab/main/ep8/grads_r1.json` (main at `1810d5b9a`, dirty=False, optimizer step=False, replica shards not run)
- branch r0: `/home/garrett/tmp/ngram-module/drivers/final-ab/final/ep8/grads_r0.json` (final at `6305757bf`, dirty=False, optimizer step=False, replica shards not run)
- branch r1: `/home/garrett/tmp/ngram-module/drivers/final-ab/final/ep8/grads_r1.json` (final at `6305757bf`, dirty=False, optimizer step=False, replica shards not run)

- mesh ['dp_shard_mod_ep', 'dp_shard_in_ep', 'dp_shard_mod_head'] [1, 8, 1], world size 8, grad accum steps 1
- relative difference `|a - b| / max(|a|, |b|)`; floor: max over main repeat pairs; arm diff: max over main x branch pairs; flag when arm diff > floor (> 1e-06 when the floor is 0)
- 213 rows compared, 0 flagged, max floor 0.00e+00, max arm diff 0.00e+00

## n-gram tables, loss and total norm

| Parameter                                          | main norm  | br / main | floor    | arm diff | Flag      |
| -------------------------------------------------- | ---------- | --------- | -------- | -------- | --------- |
| ngram.tables.1.weight                              | 9.1151e-02 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| ngram.tables.5.weight                              | 2.8916e-02 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| (loss)                                             | 1.2473e+01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| (total grad norm, pre-clip)                        | 7.3347e-01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |

## Flagged

None.

## Largest 10 arm differences

| Parameter                                          | main norm  | br / main | floor    | arm diff | Flag      |
| -------------------------------------------------- | ---------- | --------- | -------- | -------- | --------- |
| (loss)                                             | 1.2473e+01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| (total grad norm, pre-clip)                        | 7.3347e-01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| lm_head.weight                                     | 1.0730e-01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| embed_tokens.weight                                | 3.5114e-01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| hyper_connection_mixer.hc_norm.weight              | 1.9119e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| ...r_connection_mixer.input_mix_weight_down.weight | 1.0257e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| hyper_connection_mixer.input_mix_weight_up.weight  | 1.0395e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| ...ttn_hyper_connection.block_inject_weight.weight | 9.5887e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| layers.0.attn_hyper_connection.hc_norm.weight      | 2.4999e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| ...n_hyper_connection.input_mix_weight_down.weight | 2.3318e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |

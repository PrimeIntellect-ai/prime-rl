# Grad norms: dp2

- main r0: `/home/garrett/tmp/ngram-module/drivers/main/dp2/grads_r0.json` (main at `1810d5b9a`, dirty=False, optimizer step=True, replica shards equal)
- main r1: `/home/garrett/tmp/ngram-module/drivers/main/dp2/grads_r1.json` (main at `1810d5b9a`, dirty=False, optimizer step=True, replica shards equal)
- main r2: `/home/garrett/tmp/ngram-module/drivers/main/dp2/grads_r2.json` (main at `1810d5b9a`, dirty=False, optimizer step=True, replica shards equal)
- branch r0: `/home/garrett/tmp/ngram-module/drivers/main/dp2/grads_r0.json` (main at `1810d5b9a`, dirty=False, optimizer step=True, replica shards equal)
- branch r1: `/home/garrett/tmp/ngram-module/drivers/main/dp2/grads_r1.json` (main at `1810d5b9a`, dirty=False, optimizer step=True, replica shards equal)
- branch r2: `/home/garrett/tmp/ngram-module/drivers/main/dp2/grads_r2.json` (main at `1810d5b9a`, dirty=False, optimizer step=True, replica shards equal)

- mesh ['dp_replicate', 'dp_shard'] [2, 1], world size 2, grad accum steps 4
- relative difference `|a - b| / max(|a|, |b|)`; floor: max over main repeat pairs; arm diff: max over main x branch pairs; flag when arm diff > floor (> 1e-06 when the floor is 0)
- 213 rows compared, 0 flagged, max floor 0.00e+00, max arm diff 0.00e+00

## n-gram tables, loss and total norm

| Parameter                                          | main norm  | br / main | floor    | arm diff | Flag      |
| -------------------------------------------------- | ---------- | --------- | -------- | -------- | --------- |
| ngram.tables.1.weight                              | 8.9412e-02 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| ngram.tables.5.weight                              | 2.9022e-02 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| (loss)                                             | 1.2472e+01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| (total grad norm, pre-clip)                        | 7.2032e-01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |

## Flagged

None.

## Largest 10 arm differences

| Parameter                                          | main norm  | br / main | floor    | arm diff | Flag      |
| -------------------------------------------------- | ---------- | --------- | -------- | -------- | --------- |
| (loss)                                             | 1.2472e+01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| (total grad norm, pre-clip)                        | 7.2032e-01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| lm_head.weight                                     | 1.0534e-01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| embed_tokens.weight                                | 3.4473e-01 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| hyper_connection_mixer.hc_norm.weight              | 1.8956e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| ...r_connection_mixer.input_mix_weight_down.weight | 1.0107e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| hyper_connection_mixer.input_mix_weight_up.weight  | 1.0219e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| ...ttn_hyper_connection.block_inject_weight.weight | 9.6865e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| layers.0.attn_hyper_connection.hc_norm.weight      | 2.4549e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |
| ...n_hyper_connection.input_mix_weight_down.weight | 2.2817e-03 |  1.000000 |  0.0e+00 |  0.0e+00 |           |

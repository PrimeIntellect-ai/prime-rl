# Export identity: ep8

- main: `/home/garrett/tmp/ngram-module/e2e2/node-b/drivers/main/ep8/export_r0.json` (main at `522bdc015`, dirty=False)
- branch: `/home/garrett/tmp/ngram-module/e2e2/node-b/drivers/branch/ep8/export_r0.json` (branch at `4766d44a9`, dirty=False)
- world size 8 vs 8, mesh ['dp_shard_mod_ep', 'dp_shard_in_ep', 'dp_shard_mod_head'] [1, 8, 1] vs ['dp_shard_mod_ep', 'dp_shard_in_ep', 'dp_shard_mod_head'] [1, 8, 1]

| Check                                     | Result  | Detail                                         |
| ----------------------------------------- | ------- | ---------------------------------------------- |
| NCCL integer                              | pass    | 265 vs 265                                     |
| NCCL packet count                         | pass    | 265 vs 265                                     |
| NCCL dtype, shape, sha256 per key         | pass    |                                                |
| NCCL send phase, non-table keys           | pass    |                                                |
| NCCL send phase, n-gram table keys        | pass    | 262 moved layers 1,5 -> non-layer, 0 unchanged |
| NCCL chunk packet sequence                | pass    |                                                |
| NCCL intra-packet key order (info)        | same    |                                                |
| Filesystem key set                        | pass    |                                                |
| Filesystem dtype, shape, sha256 per key   | pass    |                                                |
| Filesystem shard file count (info)        | same    | 8 vs 8 files                                   |
| NIXL group names                          | pass    | ['non_layer', 'layer.0', 'layer.1', 'layer.2', |
| NIXL group, non-table keys                | pass    |                                                |
| NIXL group, n-gram table keys             | pass    | 2 moved layers 1,5 -> non-layer, 6 unchanged   |
| Save partition owner, non-table keys      | pass    | same grouping, relabeled {2: 1, 1: 2}          |
| Save partition owner, n-gram table keys   | pass    | 8 moved layers 1,5 -> non-layer, 0 unchanged   |

## NCCL packet layout

- main: 0: non-layer (338 keys); 1: layer 0 (24 keys); 2: layer 1 (33 keys); 3-130: layer 1 chunk (128 keys); 131: layer 2 (24 keys); 132: layer 3 (24 keys); 133: layer 4 (24 keys); 134: layer 5 (33 keys); 135-262: layer 5 chunk (128 keys); 263: layer 6 (24 keys); 264: layer 7 (24 keys)
- branch: 0: non-layer (344 keys); 1-256: non-layer chunk (256 keys); 257: layer 0 (24 keys); 258: layer 1 (30 keys); 259: layer 2 (24 keys); 260: layer 3 (24 keys); 261: layer 4 (24 keys); 262: layer 5 (30 keys); 263: layer 6 (24 keys); 264: layer 7 (24 keys)

## n-gram tables (trainer side)

- main `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight`: {'shape': [1282048, 160], 'mesh_dims': ['dp_shard_mod_head', 'dp_shard_cp'], 'mesh_shape': [1, 8], 'placements': ['_S(0, 8)', 'S(0)'], 'master_local_shape': [160256, 160]}
- main `model.language_model.layers.5.ple.ple_embedding.ngram_embedding.weight`: {'shape': [1284608, 160], 'mesh_dims': ['dp_shard_mod_head', 'dp_shard_cp'], 'mesh_shape': [1, 8], 'placements': ['_S(0, 8)', 'S(0)'], 'master_local_shape': [160576, 160]}
- branch `model.language_model.ngram.tables.1.weight`: {'shape': [1282048, 160], 'mesh_dims': ['dp_shard_mod_head', 'dp_shard_cp'], 'mesh_shape': [1, 8], 'placements': ['_S(0, 8)', 'S(0)'], 'master_local_shape': [160256, 160]}
- branch `model.language_model.ngram.tables.5.weight`: {'shape': [1284608, 160], 'mesh_dims': ['dp_shard_mod_head', 'dp_shard_cp'], 'mesh_shape': [1, 8], 'placements': ['_S(0, 8)', 'S(0)'], 'master_local_shape': [160576, 160]}

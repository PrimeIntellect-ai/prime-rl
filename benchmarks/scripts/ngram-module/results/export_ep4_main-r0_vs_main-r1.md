# Export identity: ep4

- main: `/home/garrett/tmp/ngram-module/drivers/main/ep4/export_r0.json` (main at `1810d5b9a`, dirty=False)
- branch: `/home/garrett/tmp/ngram-module/drivers/main/ep4/export_r1.json` (main at `1810d5b9a`, dirty=False)
- world size 4 vs 4, mesh ['dp_shard_mod_ep', 'dp_shard_in_ep', 'dp_shard_mod_head'] [1, 4, 1] vs ['dp_shard_mod_ep', 'dp_shard_in_ep', 'dp_shard_mod_head'] [1, 4, 1]

| Check                                   | Result  | Detail                                         |
| --------------------------------------- | ------- | ---------------------------------------------- |
| NCCL integer                            | pass    | 265 vs 265                                     |
| NCCL packet count                       | pass    | 265 vs 265                                     |
| NCCL packet order and key sets          | pass    |                                                |
| NCCL dtype, shape, sha256 per key       | pass    |                                                |
| NCCL intra-packet key order (info)      | same    |                                                |
| Filesystem key set                      | pass    |                                                |
| Filesystem dtype, shape, sha256 per key | pass    |                                                |
| Filesystem shard file count (info)      | same    | 4 vs 4 files                                   |
| NIXL group names                        | pass    | ['non_layer', 'layer.0', 'layer.1', 'layer.2', |
| NIXL group per key                      | pass    |                                                |
| Save partition owner per key            | pass    |                                                |

## NCCL packet layout

- main: 0: non-layer (5 keys); 1: layer 0 (24 keys); 2: layer 1 (33 keys); 3-130: layer 1 chunk (128 keys); 131: layer 2 (24 keys); 132: layer 3 (24 keys); 133: layer 4 (24 keys); 134: layer 5 (33 keys); 135-262: layer 5 chunk (128 keys); 263: layer 6 (24 keys); 264: layer 7 (24 keys)
- branch: 0: non-layer (5 keys); 1: layer 0 (24 keys); 2: layer 1 (33 keys); 3-130: layer 1 chunk (128 keys); 131: layer 2 (24 keys); 132: layer 3 (24 keys); 133: layer 4 (24 keys); 134: layer 5 (33 keys); 135-262: layer 5 chunk (128 keys); 263: layer 6 (24 keys); 264: layer 7 (24 keys)

## n-gram tables (trainer side)

- main `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight`: {'shape': [1282048, 160], 'mesh_dims': ['dp_shard_mod_head', 'dp_shard_cp'], 'mesh_shape': [1, 4], 'placements': ['_S(0, 4)', 'S(0)'], 'master_local_shape': [320512, 160]}
- main `model.language_model.layers.5.ple.ple_embedding.ngram_embedding.weight`: {'shape': [1284608, 160], 'mesh_dims': ['dp_shard_mod_head', 'dp_shard_cp'], 'mesh_shape': [1, 4], 'placements': ['_S(0, 4)', 'S(0)'], 'master_local_shape': [321152, 160]}
- branch `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight`: {'shape': [1282048, 160], 'mesh_dims': ['dp_shard_mod_head', 'dp_shard_cp'], 'mesh_shape': [1, 4], 'placements': ['_S(0, 4)', 'S(0)'], 'master_local_shape': [320512, 160]}
- branch `model.language_model.layers.5.ple.ple_embedding.ngram_embedding.weight`: {'shape': [1284608, 160], 'mesh_dims': ['dp_shard_mod_head', 'dp_shard_cp'], 'mesh_shape': [1, 4], 'placements': ['_S(0, 4)', 'S(0)'], 'master_local_shape': [321152, 160]}

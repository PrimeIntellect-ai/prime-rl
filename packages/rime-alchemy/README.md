# RIME Alchemy runtime

The `rime_study` package implements visible-state Alchemy interactions, isolated
cross-trial memory, and the Clef-flash exploration judge. The `rime-alchemy`
verifiers plugin/schema is distributed by `prime-rl-configs`.

`src/rime_study/assets/sources.lock.json` records the public training-data revision
and checksums of upstream Alchemy Python sources and episode metadata. Runtime
verification checks both installed environment source and packaged assets.

The vendored `dm_alchemy-1.0.0-py3-none-any.whl` was built from
`google-deepmind/dm_alchemy` commit
`68a26254b5c0f15e84fa0c15d66bf0c626ede8e0` using Python 3.11,
setuptools 80.9.0 and grpcio-tools 1.62.3. It runs under Python 3.12 without
executing upstream's Python-3.11-only setup script. Alchemy is Apache 2.0
licensed; its license accompanies the wheel. The public dataset's README,
including its MIT license declaration, is included with the assets.

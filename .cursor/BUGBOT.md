# BugBot Instructions

Any PR that introduces a new custom model must also provide a table showing mean KL mismatch across 20 steps on math environment on this new model with `batch_size=64`. All the entries in the table must lower than 0.015. If this is not present, request the author to add such a table.

A PR that adds a decision model (an entry in `SYSTEM_ONE_STRATEGIES`) must provide the tables of "Requirements for merging a decision model" in `docs/development.md`, with every `compare` table passing. It needs the `math` table only if it also adds a new backbone. If the tables are not present, request the author to add them.

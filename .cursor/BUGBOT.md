# BugBot Instructions

Any PR that introduces a new custom model must also provide a table showing mean KL mismatch across 20 steps on math environment on this new model with `batch_size=64`. All the entries in the table must lower than 0.015. If this is not present, request the author to add such a table.

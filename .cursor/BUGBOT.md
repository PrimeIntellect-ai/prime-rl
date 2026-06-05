# BugBot Instructions

## Changelog Enforcement

Any PR that introduces **breaking** configuration changes must update `CHANGELOG.md`. Breaking changes are those that require users to update existing configs:

- **Renamed** config fields (old name no longer accepted)
- **Removed** config fields (field deleted or moved to a different path)
- **Moved** config fields (field relocated in the config hierarchy)

Additive changes (new fields with defaults, new optional features) and default value changes do **not** require a changelog entry.

Config files live in:

- `src/prime_rl/configs/`

If breaking changes are detected without a corresponding `CHANGELOG.md` update, request that the author add an entry.

Any PR that introduces a new custom model must also provide a table showing mean KL mismatch across 20 steps on math environment on this new model with `batch_size=64`. All the entries in the table must lower than 0.015. If this is not present, request the author to add such a table.

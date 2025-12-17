# BugBot Instructions

## Changelog Enforcement

Any PR that modifies configuration structures or usage patterns must update `CHANGELOG.md`. This includes changes to config fields (added, removed, renamed, moved, or default value changes) in:

- `src/prime_rl/*/config.py`
- `src/prime_rl/rl.py`
- `src/prime_rl/utils/config.py`

If such changes are detected without a corresponding `CHANGELOG.md` update, request that the author add an entry.

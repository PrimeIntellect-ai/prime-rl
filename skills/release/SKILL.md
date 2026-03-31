---
name: release
description: How to prepare and publish GitHub releases for prime-rl. Use when drafting release notes, tagging versions, or publishing releases.
---

# Releases

## Preparing release notes

1. **Style reference**: check the previous release (`gh release list --limit 1` then `gh release view <tag>`) to match the tone and formatting.
2. **Gather changes**: use `git log <last-tag>..origin/main --oneline --no-merges` to list all commits since the last release.
3. **Check for new commits**: always `git fetch origin main` and re-check right before publishing, since PRs may have been merged while drafting.
4. **Structure**: organize notes into numbered highlight sections (`# 1.`, `# 2.`, ...), then `# Breaking Changes`, `# Bug Fixes`, and `# Misc`.
5. **Highlights**: group related PRs under a single highlight. Use `##` subsections when a highlight contains multiple items (e.g. Performance & Parallelism). Keep the top highlights for the most impactful user-facing features.
6. **Config examples**: when referencing TOML config, verify the exact field names against the actual code or docs — don't guess.
7. **Links**: use clickable links for docs (`[docs/foo.md](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/foo.md)`) and PR references (`[#1234](https://github.com/PrimeIntellect-ai/prime-rl/pull/1234)`).
8. **Contributors**: list all contributors ranked by number of commits, using their GitHub `@username`. Get usernames via the GitHub API, not git author names (which can be inconsistent).
9. **Draft first**: always create releases as `--draft` first, iterate on content, then publish.

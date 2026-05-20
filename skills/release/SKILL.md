---
name: release
description: How to prepare and publish GitHub releases for prime-rl. Use when drafting release notes, tagging versions, or publishing releases.
---

# Releases

1. **Match prior style** — `gh release list --limit 1` then `gh release view <tag>` to mirror tone and structure.
2. **Gather commits** — `git fetch origin main` and `git log <last-tag>..origin/main --oneline --no-merges`. Re-fetch right before publishing in case PRs landed during drafting.
3. **Structure** — numbered highlight sections (`# 1.`, `# 2.`, ...) for impactful user-facing features, then `# Breaking Changes`, `# Bug Fixes`, `# Misc`. Use `##` subsections inside a highlight when it bundles multiple items.
4. **Verify TOML field names** — when referencing config in notes, check the actual code; don't guess.
5. **Links** — clickable for docs (`[docs/foo.md](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/foo.md)`) and PRs (`[#1234](https://github.com/PrimeIntellect-ai/prime-rl/pull/1234)`).
6. **Contributors** — list by commit count, using GitHub `@username` from the API (git author names can be inconsistent).
7. **Always draft first** — `gh release create ... --draft`, iterate, then publish.

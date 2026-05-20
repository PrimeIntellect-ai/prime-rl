---
name: release
description: How to prepare and publish GitHub releases for prime-rl. Use when drafting release notes, tagging versions, or publishing releases.
---

# Releases

Prime-rl releases are driven by [`.github/workflows/tag-and-release.yaml`](../../.github/workflows/tag-and-release.yaml). When a PR that bumps `version` in `pyproject.toml` lands on `main`, the workflow auto-creates the `v{version}` tag and the matching GitHub Release using the body from `assets/release/RELEASE_v{version}.md`. **Prime-rl is not published to PyPI** — the workflow does not publish a package.

This skill covers **stable** releases only. The `devx_tag.yaml` workflow handles `.dev` tags independently — it runs on every push to `main` and creates `vX.Y.Z.devN` tags (no GitHub Release, no release notes file). The two workflows coexist: `tag-and-release.yaml` ignores `*dev*` tags via its trigger filter.

Your job in this skill is to prepare the release PR so a maintainer can simply review and merge.

## Preparing the release PR

1. **Sync and confirm prerequisites**
   - `git fetch origin --tags`
   - Read the current version: `grep '^version' pyproject.toml`
   - Confirm the previous release tag matches: `gh release list --repo PrimeIntellect-ai/prime-rl --limit 1`
2. **Decide the new version** with the user. Follow SemVer for prime-rl (`MAJOR.MINOR.PATCH`); past versions are listed under `assets/release/`. If the user does not specify, suggest the next minor bump and ask before continuing.
3. **Create a branch off main**: `git switch -c chore/release-v{NEW_VERSION}`
4. **Bump the version** in `pyproject.toml` (the only place — there is no `__version__` in the source tree).
5. **Draft release notes** at `assets/release/RELEASE_v{NEW_VERSION}.md` (see structure below).
6. **Commit** the version bump and release notes together (`chore: release v{NEW_VERSION}`).
7. **Open a draft PR** with `gh pr create --draft --title "chore: release v{NEW_VERSION}" --body-file -`. The body should summarize the highlights and link to the rendered release notes file (`assets/release/RELEASE_v{NEW_VERSION}.md`).
8. **Stop there.** Do not tag, push tags, or publish a release. Merging the PR triggers the workflow, which tags the commit and publishes the GitHub Release automatically.

## Drafting release notes

The `assets/release/RELEASE_v{tag}.md` file is the canonical release body. The Tag and Release workflow refuses to run if it is missing, and it is reused verbatim as the GitHub Release body.

1. **Style reference**: read the previous release notes file under `assets/release/` to match tone and structure. Past releases also live in `gh release view <tag>`.
2. **Gather changes**: `git log v{PREV_VERSION}..origin/main --oneline --no-merges` lists commits since the last release. For PR-level granularity, prefer `gh pr list --base main --state merged --search "merged:>={prev_release_date}" --limit 500`.
3. **Re-fetch before publishing**: `git fetch origin main` right before finalizing, in case PRs landed while you were drafting.
4. **Structure** (match recent releases):
   - Numbered highlight sections (`# 1.`, `# 2.`, ...) for the most impactful user-facing changes. Use `##` subsections when a highlight bundles multiple related PRs (e.g. "Performance & Parallelism").
   - `# Breaking Changes` for renamed/removed config fields, default flips, or dependency major bumps. Cross-check with `CHANGELOG.md`.
   - `# Bug Fixes` listing notable fixes.
   - `# Misc` for the long tail of refactors, dep bumps, docs.
   - `# Contributors` — every distinct GitHub `@username` who landed a PR in this window, ordered by commit count.
5. **PR references**: link as `[#1234](https://github.com/PrimeIntellect-ai/prime-rl/pull/1234)`. Doc links go to the canonical path on main, e.g. `[docs/foo.md](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/foo.md)`.
6. **Contributors**: get GitHub usernames from the PR author field (`gh pr list --json author`), not git author names — those are inconsistent.
7. **Config examples**: verify any TOML field names in the notes against the actual config classes — don't guess.

## After the PR merges

The maintainer merges to `main`. The workflow then:

1. Detects the `version` change in `pyproject.toml` between `before` and `after`.
2. Creates and pushes `v{NEW_VERSION}`.
3. Creates the GitHub Release using `assets/release/RELEASE_v{NEW_VERSION}.md` as the body.

If something goes wrong (missing notes file, version mismatch), the workflow fails fast. You can re-run an existing tag via the workflow's `workflow_dispatch` input — useful if release notes were patched after the fact.

## Sanity checks before opening the PR

- `grep '^version' pyproject.toml` matches the filename suffix of the new `RELEASE_v*.md`.
- `ls assets/release/RELEASE_v*.md` shows the new file alongside the historical ones.
- `git diff origin/main -- pyproject.toml assets/release/` shows only the intended changes.
- Render the release notes (`gh markdown`) or read them to confirm formatting before pushing.

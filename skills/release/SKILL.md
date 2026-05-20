---
name: release
description: How to prepare and publish GitHub releases for prime-rl. Use when drafting release notes, tagging versions, or publishing releases.
---

# Releases

Prime-rl releases are driven by [`.github/workflows/tag-and-release.yaml`](../../.github/workflows/tag-and-release.yaml). The flow:

1. You create a **draft GitHub Release** with the release notes inline (via `gh release create --draft`).
2. You open a **draft PR** that bumps `version` in `pyproject.toml`.
3. A maintainer reviews and merges the PR.
4. The workflow detects the version bump on `main`, pushes the `v{version}` tag, and promotes the draft release to published.

Release notes do **not** live in the repo — they live on the GitHub Release. Prime-rl is **not** published to PyPI; the workflow does not publish a package.

This skill covers **stable** releases only. The `devx_tag.yaml` workflow handles `.dev` tags independently — it runs on every push to `main` and creates `vX.Y.Z.devN` tags (no GitHub Release). The two workflows coexist: `tag-and-release.yaml` ignores `*dev*` tags via its trigger filter.

## Step 1 — sync and decide the version

```bash
git fetch origin --tags
grep '^version' pyproject.toml                       # current version
gh release list --repo PrimeIntellect-ai/prime-rl    # most recent stable tags
```

Decide the next version with the user. Follow SemVer (`MAJOR.MINOR.PATCH`). If the user does not specify, suggest the next minor bump and ask before continuing.

## Step 2 — draft the release notes

Write the notes to a temporary file (do **not** commit it). Match the style of the most recent release:

```bash
gh release view "$(gh release list --limit 1 --json tagName --jq '.[0].tagName')" --json body --jq .body
```

Structure (match recent releases):

- Numbered highlight sections (`# 1.`, `# 2.`, ...) for the most impactful user-facing changes. Use `##` subsections when a highlight bundles multiple related PRs.
- `# Breaking Changes` — renamed/removed config fields, default flips, dependency major bumps. Cross-check against `CHANGELOG.md`.
- `# Bug Fixes` — notable fixes.
- `# Misc` — refactors, dep bumps, docs.
- `# Contributors` — every distinct GitHub `@username` who landed a PR in this window, ordered by commit count.

Gather changes:

```bash
PREV=$(gh release list --limit 1 --json tagName --jq '.[0].tagName')
git log "$PREV"..origin/main --oneline --no-merges
gh pr list --base main --state merged --search "merged:>=$(gh release view "$PREV" --json publishedAt --jq .publishedAt)" --limit 500 --json number,title,author
```

Tips:

- Re-fetch right before publishing — PRs may have landed while you were drafting (`git fetch origin main`).
- PR links: `[#1234](https://github.com/PrimeIntellect-ai/prime-rl/pull/1234)`. Docs links go to canonical path on main, e.g. `[docs/foo.md](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/foo.md)`.
- Contributors: use `gh pr list --json author` for usernames — git author names are inconsistent.
- Config examples: verify TOML field names against the actual config classes — don't guess.

## Step 3 — create the draft GitHub Release

```bash
NEW=v0.6.0   # whatever the new tag is
gh release create --draft "$NEW" \
  --title "$NEW" \
  --target main \
  --notes-file /tmp/release-notes-$NEW.md
```

Verify it exists as a draft:

```bash
gh release view "$NEW" --json isDraft,tagName,name --jq '{tagName, name, isDraft}'
# expect: {"tagName":"v0.6.0","name":"v0.6.0","isDraft":true}
```

If you need to revise: `gh release edit "$NEW" --notes-file /tmp/release-notes-$NEW.md`.

## Step 4 — open the draft version-bump PR

```bash
git switch -c chore/release-$NEW
# edit pyproject.toml: bump the `version = "..."` line
git add pyproject.toml
git commit -m "chore: release $NEW"
git push -u origin "chore/release-$NEW"

gh pr create --draft --title "chore: release $NEW" --body-file - <<EOF
## Summary

Bumps version to \`${NEW#v}\`. The draft GitHub Release at $NEW will be published automatically when this PR merges (see [\`.github/workflows/tag-and-release.yaml\`](../blob/main/.github/workflows/tag-and-release.yaml)).

Draft release: https://github.com/PrimeIntellect-ai/prime-rl/releases/tag/$NEW
EOF
```

Stop there. Do **not** tag, push tags, or flip the draft to published yourself.

## After merge

The workflow:

1. Detects the `version` change in `pyproject.toml` between `before` and `after`.
2. Verifies a draft release exists for `v{new}`. (Fails fast if you skipped Step 3.)
3. Creates and pushes the `v{new}` tag at the merge commit.
4. Promotes the draft release to published (`gh release edit --draft=false --latest`).

If the release step failed but the tag was already pushed, the workflow on the next main push will see the tag exists with a still-draft release and re-promote it. You can also recover manually via the workflow's `workflow_dispatch` input (`tag: v{new}`).

## Sanity checks before opening the PR

- `gh release view v{new} --json isDraft --jq .isDraft` returns `true`.
- `grep '^version' pyproject.toml` shows the new version on the PR branch.
- The PR body links to the draft release URL.

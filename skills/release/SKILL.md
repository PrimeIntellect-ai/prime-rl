---
name: release
description: How to prepare and publish GitHub releases for prime-rl. Use when drafting release notes, tagging versions, or publishing releases.
---

# Releases

Releases are driven by [`.github/workflows/tag-and-release.yaml`](../../.github/workflows/tag-and-release.yaml). The flow:

1. You create a **draft GitHub Release** with the notes inline (`gh release create --draft`).
2. You open a **draft PR** that bumps `version` in `pyproject.toml`.
3. Maintainer merges. The workflow tags the commit and promotes the draft.

Release notes live on the GitHub Release, not in the repo. Prime-rl is **not** on PyPI. `.dev` tags are handled separately by `devx_tag.yaml`; `tag-and-release.yaml` ignores them.

## 1. Decide the version

```bash
git fetch origin --tags
grep '^version' pyproject.toml
gh release list --repo PrimeIntellect-ai/prime-rl --limit 5
```

SemVer (`MAJOR.MINOR.PATCH`). Confirm with the user before continuing.

## 2. Draft the notes

Match the prior release's structure: numbered highlights (`# 1.`, ...), then `# Breaking Changes`, `# Bug Fixes`, `# Misc`, `# Contributors`. Use `##` subsections inside a highlight when it bundles multiple items.

```bash
PREV=$(gh release list --limit 1 --json tagName --jq '.[0].tagName')
gh release view "$PREV" --json body --jq .body          # style reference
git log "$PREV"..origin/main --oneline --no-merges      # commits since
gh pr list --base main --state merged --search \
  "merged:>=$(gh release view "$PREV" --json publishedAt --jq .publishedAt)" \
  --limit 500 --json number,title,author                # for PR links + contributors
```

Tips:
- PR refs: `[#1234](https://github.com/PrimeIntellect-ai/prime-rl/pull/1234)`.
- Contributors: order by commit count, use the GH `@username` from the API (not git author names).
- Verify any TOML field names against the actual config classes.

## 3. Create the draft release

```bash
NEW=v0.6.0
gh release create --draft "$NEW" --title "$NEW" --target main --notes-file /tmp/release-notes-$NEW.md
gh release view "$NEW" --json isDraft,tagName --jq '{tagName, isDraft}'   # expect isDraft: true
```

Iterate with `gh release edit "$NEW" --notes-file /tmp/release-notes-$NEW.md`.

## 4. Open the version-bump PR

```bash
git switch -c chore/release-$NEW
# bump `version = "..."` in pyproject.toml
git add pyproject.toml
git commit -m "chore: release $NEW"
git push -u origin "chore/release-$NEW"
gh pr create --draft --title "chore: release $NEW" --body "Bumps version to ${NEW#v}. Draft release: https://github.com/PrimeIntellect-ai/prime-rl/releases/tag/$NEW"
```

Stop. Do not tag, push tags, or flip the draft to published — the workflow does that on merge.

## Recovery

If the workflow tagged the commit but failed to promote the draft, the next main push (or `workflow_dispatch` with `tag: v{new}`) re-promotes it.

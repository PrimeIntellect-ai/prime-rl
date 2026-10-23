# Custom code, branches, and commits

Read this when the run must use code other than the managed build. The
[public dispatch API](https://docs.primeintellect.ai/api-reference/training/create-dedicated-run)
defines `sourceRef`: a branch, tag, or commit in **PrimeIntellect-ai/prime-rl**.
The service resolves it to a commit, validates against that schema, and overlays
the source on the image at pod startup.

## Access and revision selection

`prime train run CONFIG.toml --ref REF` is restricted: the run must belong to a
team with source-ref access enabled by Prime, and the deployment must support
source overlays. The CLI hides `--ref` from ordinary help in some releases;
absence from help does not prove it is unsupported. Check the
[CLI implementation](https://github.com/PrimeIntellect-ai/prime/blob/main/packages/prime/src/prime_cli/commands/rl.py)
or installed command definition when needed. A 403 is an access issue; do not
silently remove `--ref` and launch different code.

For a custom branch or commit:

1. Inspect the selected checkout's remote, branch, working tree, and submodule
   revisions. Identify the exact changes the user intends to run.
2. Commit and push those changes to the canonical repository within the user's
   authorization. Confirm the remote URL before pushing; a fork named `origin`
   is not the canonical repo. Local edits, a local branch, and commits available
   only in a private fork are not uploaded by `prime train`.
3. Verify the branch exists remotely and resolve the full commit SHA. Prefer
   that SHA for reproducibility. A branch or tag is also accepted, but record
   the resolved SHA; the CLI's `sourceRef` echo may be the input branch name.
4. Validate the TOML using that commit's config schema and record any required
   image build. Launch with `--ref` and verify the startup log's
   `source overlay at <SHA>` matches. The entrypoint also exports
   `PRIME_RL_SOURCE_COMMIT` inside the container.

After selecting `CANONICAL_REMOTE` and `BRANCH`, a typical publication check is:

```bash
git remote get-url "$CANONICAL_REMOTE"
git push "$CANONICAL_REMOTE" "HEAD:refs/heads/$BRANCH"
CODE_REF=$(git rev-parse HEAD)
git ls-remote "$CANONICAL_REMOTE" "refs/heads/$BRANCH"
```

Compare the remote SHA with `CODE_REF`, then include it in the prepared launch:

```bash
prime train run "$CONFIG_PATH" --ref "$CODE_REF" \
  --gpu-type "$GPU_TYPE" --yes --plain -o json
```

Carry over the volume and secret flags required by the main workflow. If supported,
`--pr NUMBER` selects the head of an unmerged PR in the canonical repo; it is
mutually exclusive with `--ref`, has the same access restrictions, and does not
support fork PRs. Do not guess that a GitHub URL or a fork's branch is a valid ref.

If the required code is private, preserve that visibility. Do not publish it to
the canonical public repo just to make `--ref` work. Explain the limitation and
resolve a supported private delivery mechanism with the user/Prime before launch.

## Code and dependency delivery

`--image-tag TAG` chooses a build from the platform's prime-rl image repository;
`--ref` chooses the source overlaid on that build. An image tag is not an arbitrary
Docker image URI. Check the bootstrap baked into the selected image. This repo's
[entrypoint](../../../scripts/docker-entrypoint.sh) fetches the source and pinned
public submodules, then runs a locked workspace sync into the container's
environment. Commit dependency changes and the updated `uv.lock` together.
Heavy package pins (including torch, vLLM, and prime-kernels) must match the image:
the entrypoint rejects drift instead of rebuilding those wheels. Changes to
these pins, CUDA kernels, or system packages need a compatible image build.

For custom tasksets, harnesses, scorers, and tools, check that their package and
registration are available in the remote runtime. Include source/assets and
dependency declarations in the launched revision, or use a supported published
environment/image delivery path. A local editable install or `prime env install`
does not install a package in hosted pods. A Hub ID also needs to be accessible
to the run's owner; use a published version compatible with the chosen runtime.

For a Hub delivery path, publish the package/version with `prime env push` from
its package directory and explicitly preserve its intended visibility. For
example, `prime env push --team "$TEAM_SLUG" --visibility PRIVATE --plain`
uploads a private team environment; `--team` here takes the team slug, whereas
CLI account setup uses the team ID. This requires authorization to publish that
package. Include dependencies in the package metadata; local `tool.uv.sources`
alone is not wheel dependency metadata. See
[environment publishing](https://docs.primeintellect.ai/tutorials-environments/create).

If code lives in a submodule, push its commit to an accessible remote and commit
the parent gitlink too. Verify how the hosted source overlay fetches submodules
and installs environment packages. Do not assume a dirty local submodule or a
private Git dependency is readable by the service. Never embed Git credentials
in dependency URLs.

For files read at runtime, verify their location in the container or attached
volume. Relative paths resolve in the remote process's working directory. Local
absolute paths, shell environment changes, and untracked prompt files do not
travel with the submitted TOML.

#!/usr/bin/env bash
set -euo pipefail

mode="env"
if [ "${1:-}" = "--shell" ] || [ "${1:-}" = "--github-output" ]; then
    mode="${1#--}"
    shift
fi

root="${1:-.}"
root="$(cd "$root" && pwd)"

dependency_floor_version() {
    local package="$1"

    find "$root" \
        \( -path "$root/.git" -o -path "$root/.venv" \) -prune -o \
        -type f -name pyproject.toml -print |
        while IFS= read -r pyproject; do
            sed -nE "s/.*[\"']${package}(\\[[^]]+\\])?>=([^\"',; ]+).*/\\2/p" "$pyproject"
        done |
        sort -V |
        tail -n 1
}

next_patch_dev_version() {
    local tag="$1"
    local distance="$2"
    local version="${tag#v}"
    local major minor patch

    if [ "$distance" = "0" ]; then
        printf '%s\n' "$version"
        return 0
    fi

    if [[ "$version" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
        major="${BASH_REMATCH[1]}"
        minor="${BASH_REMATCH[2]}"
        patch="${BASH_REMATCH[3]}"
        printf '%s.%s.%s.dev%s\n' "$major" "$minor" "$((patch + 1))" "$distance"
        return 0
    fi

    return 1
}

describe_distance() {
    sed -E 's/.*-([0-9]+)-g[0-9a-f]+$/\1/' <<< "$1"
}

describe_tag() {
    sed -E 's/-[0-9]+-g[0-9a-f]+$//' <<< "$1"
}

verifiers_git_version() {
    local dir="$root/deps/verifiers"
    local desc tag distance

    [ -d "$dir" ] || return 1
    desc="$(git -C "$dir" describe --tags --long --match 'v[0-9]*.[0-9]*.[0-9]*' --exclude '*dev*' 2>/dev/null)" || return 1
    tag="$(describe_tag "$desc")"
    distance="$(describe_distance "$desc")"
    next_patch_dev_version "$tag" "$distance"
}

renderers_git_version() {
    local dir="$root/deps/renderers"
    local desc tag distance version

    [ -d "$dir" ] || return 1
    desc="$(git -C "$dir" describe --tags --long --match 'renderers-v*' 2>/dev/null)" || return 1
    tag="$(describe_tag "$desc")"
    distance="$(describe_distance "$desc")"
    version="${tag#renderers-v}"

    if [ "$distance" = "0" ]; then
        printf '%s\n' "$version"
        return 0
    fi

    next_patch_dev_version "v$version" "$distance"
}

resolve_version() {
    local package="$1"
    local env_name="$2"
    local git_func="$3"
    local version

    version="${!env_name:-}"
    if [ -n "$version" ]; then
        printf '%s\n' "$version"
        return 0
    fi

    if version="$("$git_func")" && [ -n "$version" ]; then
        printf '%s\n' "$version"
        return 0
    fi

    version="$(dependency_floor_version "$package")"
    if [ -n "$version" ]; then
        printf '%s\n' "$version"
        return 0
    fi

    printf 'Could not infer a pretend version for %s under %s\n' "$package" "$root" >&2
    return 1
}

verifiers_version="$(resolve_version verifiers VERIFIERS_PRETEND_VERSION verifiers_git_version)"
renderers_version="$(resolve_version renderers RENDERERS_PRETEND_VERSION renderers_git_version)"

printf 'Resolved verifiers pretend version: %s\n' "$verifiers_version" >&2
printf 'Resolved renderers pretend version: %s\n' "$renderers_version" >&2

case "$mode" in
shell)
    printf "export VERIFIERS_PRETEND_VERSION='%s'\n" "$verifiers_version"
    printf "export RENDERERS_PRETEND_VERSION='%s'\n" "$renderers_version"
    ;;
github-output)
    printf 'verifiers=%s\n' "$verifiers_version"
    printf 'renderers=%s\n' "$renderers_version"
    ;;
env)
    printf 'VERIFIERS_PRETEND_VERSION=%s\n' "$verifiers_version"
    printf 'RENDERERS_PRETEND_VERSION=%s\n' "$renderers_version"
    ;;
*)
    printf 'Unknown output mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac

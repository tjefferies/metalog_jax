#!/usr/bin/env bash
# Build the GPU benchmark's image on GitHub, and print the image to run, by digest.
#
# Usage:
#   benchmarks/gpu/build_image.sh [--ref REF] [--no-wait]
#   benchmarks/gpu/build_image.sh --for-run RUN_ID
#
# What it does:
#   Starts the "GPU benchmark image" workflow (.github/workflows/gpu-bench-image.yml).
#   The workflow builds benchmarks/gpu/Dockerfile from REF, pushes the image to
#   ghcr.io/<owner>/<repo>/gpu-bench:<version>-<short commit>, and attests where it was
#   built. That takes about 8 minutes. This script waits for it, then prints the
#   image by its digest, which is the form the other benchmark scripts take:
#
#     ghcr.io/tjefferies/metalog_jax/gpu-bench@sha256:<64 hex digits>
#
#   A digest names exactly one image, so a run on it can be repeated; a tag can be
#   moved.
#
# When to rebuild:
#   After changing anything the image contains: metalog_jax/, benchmarks/gpu/*.py,
#   benchmarks/gpu/Dockerfile or uv.lock. A change to benchmarks/gpu/plan.json alone
#   needs no new image: the launcher passes the plan to each pod.
#
# Options:
#   --ref REF         The branch, tag or commit to build. Default: development.
#   --no-wait         Start the build and return without waiting for it.
#   --for-run RUN_ID  Build nothing: wait for an earlier build, then print its image.
#                     Its ID is in: gh run list --workflow gpu-bench-image.yml
#   -h, --help        Show this help.
#
# Requires:
#   gh, signed in with permission to run the repository's workflows, and curl.
#
# Costs:
#   Nothing. The build runs on GitHub's runners.
#
# Notes:
#   The digest is read from GHCR without credentials, so the gpu-bench package must
#   be public. That's a one-time setting on GitHub, which Runpod needs too. Until
#   then, the build run's summary shows the digest: gh run view RUN_ID --web
#
# Examples:
#   benchmarks/gpu/build_image.sh                       # build development
#   benchmarks/gpu/build_image.sh --ref my-branch       # build a branch
#   benchmarks/gpu/build_image.sh --for-run 45733275193 # an earlier build's image
#   IMAGE=$(benchmarks/gpu/build_image.sh | tail -n 1)  # keep the image for later
#
# Exit status:
#   0  the image is built, and printed as the last line (with --no-wait: started)
#   1  the build, or reading its digest, failed
#   64 a usage error
set -euo pipefail

WORKFLOW="gpu-bench-image.yml"
WORKFLOW_NAME="GPU benchmark image"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$SELF")/../.."

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$SELF"; }
die() {
	echo "gpu/build_image.sh: $*" >&2
	exit 64
}
fail() {
	echo "gpu/build_image.sh: $*" >&2
	exit 1
}

# A minute ago in UTC (ISO 8601), to find a run despite clock skew from GitHub
a_minute_ago() {
	date -u -v-60S +%Y-%m-%dT%H:%M:%SZ 2>/dev/null ||
		date -u -d '60 seconds ago' +%Y-%m-%dT%H:%M:%SZ
}

# The newest run of workflow $1 that this user started at or after $2
find_run() {
	local user id _
	user=$(gh api user --jq .login)
	for _ in $(seq 1 30); do
		id=$(gh run list --workflow "$1" --user "$user" --event workflow_dispatch \
			--limit 10 --json databaseId,createdAt \
			--jq "[.[] | select(.createdAt >= \"$2\")] | sort_by(.createdAt) | last | .databaseId // empty")
		if [ -n "$id" ]; then
			echo "$id"
			return 0
		fi
		sleep 2
	done
	return 1
}

# The digest GHCR serves for repository $1 and tag $2, read without credentials
ghcr_digest() {
	local token
	token=$(curl -fsS "https://ghcr.io/token?scope=repository:$1:pull" |
		sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
	[ -n "$token" ] || return 1
	curl -fsSI -H "Authorization: Bearer $token" \
		-H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json" \
		"https://ghcr.io/v2/$1/manifests/$2" |
		tr -d '\r' | awk -F': ' 'tolower($1) == "docker-content-digest" { print $2 }'
}

ref=development
wait=1
run_id=
while [ $# -gt 0 ]; do
	case $1 in
	--ref)
		[ $# -ge 2 ] || die "--ref needs a branch, tag or commit"
		ref=$2
		shift 2
		;;
	--no-wait)
		wait=0
		shift
		;;
	--for-run)
		[ $# -ge 2 ] || die "--for-run needs a run ID"
		run_id=$2
		shift 2
		;;
	-h | --help)
		usage
		exit 0
		;;
	*) die "unknown argument: $1 (see --help)" ;;
	esac
done

command -v gh >/dev/null || die "gh is required: https://cli.github.com"
command -v curl >/dev/null || die "curl is required"
repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
image=$(echo "ghcr.io/$repo/gpu-bench" | tr '[:upper:]' '[:lower:]')

if [ -z "$run_id" ]; then
	started=$(a_minute_ago)
	gh workflow run "$WORKFLOW" --ref "$ref"
	run_id=$(find_run "$WORKFLOW" "$started") ||
		fail "the build started, but its run didn't appear: gh run list --workflow $WORKFLOW"
else
	name=$(gh run view "$run_id" --json workflowName --jq .workflowName) ||
		die "no run $run_id"
	[ "$name" = "$WORKFLOW_NAME" ] || die "run $run_id is a \"$name\" run, not a build"
fi
echo "Build: $(gh run view "$run_id" --json url --jq .url)" >&2
if [ "$wait" = 0 ]; then
	echo "When it's done: benchmarks/gpu/build_image.sh --for-run $run_id" >&2
	exit 0
fi

gh run watch "$run_id" --exit-status --interval 15 >&2 ||
	fail "the build failed: gh run view $run_id --log-failed"

# The workflow tags the image <version>-<first 7 characters of the commit>
sha=$(gh run view "$run_id" --json headSha --jq .headSha)
version=$(gh api -H "Accept: application/vnd.github.raw" \
	"repos/$repo/contents/pyproject.toml?ref=$sha" |
	sed -n 's/^version = "\(.*\)"$/\1/p' | head -n 1)
[ -n "$version" ] || fail "couldn't read the version in pyproject.toml at $sha"
tag="$version-$(printf %.7s "$sha")"
digest=$(ghcr_digest "${image#ghcr.io/}" "$tag") && [ -n "$digest" ] ||
	fail "couldn't read $image:$tag's digest from GHCR. Is the package public? The run's summary shows it: gh run view $run_id --web"

cat >&2 <<EOF

Built $image:$tag from $sha.
Dry run on Runpod:   benchmarks/gpu/run_runpod.sh --image $image@$digest
Where it was built:  gh attestation verify oci://$image@$digest --repo $repo

EOF
echo "$image@$digest"

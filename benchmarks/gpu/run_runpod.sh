#!/usr/bin/env bash
# Run the GPU benchmark on Runpod: one pod per GPU type, within a budget.
#
# Usage:
#   benchmarks/gpu/run_runpod.sh --image IMAGE [--gpus LIST] [--budget USD]
#       [--cloud SECURE|COMMUNITY] [--run [--yes]] [--watch] [--ref REF]
#   benchmarks/gpu/run_runpod.sh --local --image IMAGE [--gpus LIST] [--budget USD]
#       [--cloud SECURE|COMMUNITY] [--plan FILE] [--out DIR] [--prices FILE] [--run [--yes]]
#
# What it does:
#   Without --run, a dry run. It reads Runpod's live prices, then prints:
#     - each GPU type's price and availability
#     - the time cap every pod would get
#     - the estimated cost
#     - the plan's digest
#   It creates nothing and costs nothing. With --run, it creates the pods, and each
#   pod measures until its cap:
#     - correctness
#     - single batches up to the GPU's memory
#     - a chunked run
#     - term counts
#     - sampling
#   Then the launcher downloads and checks each pod's results, terminates the pods,
#   and reads the charges.
#
# Where the launcher runs:
#   On GitHub (the default). This starts the "GPU benchmark" workflow
#   (.github/workflows/gpu-bench.yml) on REF:
#     - RUNPOD_API_KEY comes from the repository's runpod environment, which may ask
#       a reviewer to approve each run on the run's page.
#     - The pods run REF's benchmarks/gpu/plan.json, so commit and push a plan
#       change first.
#     - The results become the run's gpu-benchmark-results artifact, which
#       benchmarks/report/collect.sh downloads.
#     - The run's page shows a progress line per measurement while it runs. The
#       launcher's full output is printed here at the end with --watch, and always
#       for a dry run.
#   Here, with --local. benchmarks/gpu/launch.py runs from this checkout:
#     - RUNPOD_API_KEY comes from your environment.
#     - The results go to --out.
#     - Keep the terminal open until it ends: the launcher terminates the pods when
#       their results are in, and on Ctrl-C.
#
# Cost:
#   - The cap: the launcher gives every pod one time cap, so that the GPUs' total
#     list price for the cap stays within 92% of --budget. It refuses a cap under 15
#     minutes.
#   - Real charges: each pod is terminated as soon as its results are in, so the
#     charges come in under the estimate.
#   - At 2026-11-24's Secure Cloud prices (A100 $2.19, H100 $4.39, H200 $5.29 an
#     hour):
#
#       --gpus                       --budget   cap      estimate
#       a100-sxm,h100-sxm,h200-sxm   5.00       23 min   $4.55
#       a100-sxm                     1.00       25 min   $0.91
#       h200-sxm                     2.00       20 min   $1.76
#
# Options:
#   --image IMAGE  Required. The benchmark image by digest, as
#                  benchmarks/gpu/build_image.sh prints it:
#                  ghcr.io/<owner>/<repo>/gpu-bench@sha256:<digest>. The
#                  sha256-<digest> tag on the package page is the image's attestation,
#                  not the image.
#   --gpus LIST    Comma-separated GPU types: a100-sxm, h100-sxm, h200-sxm. Default:
#                  all three. A type Runpod can't place is skipped, and costs nothing.
#   --budget USD   The most the run may cost. Default: 5.00.
#   --cloud CLOUD  SECURE (the default) or COMMUNITY. Community Cloud is cheaper, and
#                  its hosts vary more.
#   --run          Create the pods. Without it, a dry run.
#   --yes          Don't ask before creating pods. Needed without a terminal.
#   --watch        On GitHub: wait for the run, then print the launcher's output. A
#                  dry run always waits.
#   --ref REF      On GitHub: the branch or tag to run. Default: development.
#   --local        Run the launcher here, instead of on GitHub.
#   --plan FILE    With --local: the plan the pods run. Default:
#                  benchmarks/gpu/plan.json.
#   --out DIR      With --local: where results go. Default: benchmarks/gpu/results.
#                  Each run gets a subdirectory named by its run ID.
#   --prices FILE  With --local: a saved GPU catalog (JSON) instead of Runpod's live
#                  prices, for a dry run without RUNPOD_API_KEY.
#   -h, --help     Show this help.
#
# Requires:
#   On GitHub: gh, signed in with permission to run the repository's workflows.
#   With --local: uv, and RUNPOD_API_KEY.
#
# Examples:
#   IMAGE=ghcr.io/tjefferies/metalog_jax/gpu-bench@sha256:...
#   benchmarks/gpu/run_runpod.sh --image "$IMAGE"            # what would $5 buy?
#   benchmarks/gpu/run_runpod.sh --image "$IMAGE" --gpus a100-sxm --budget 1.00 --run --watch
#   benchmarks/gpu/run_runpod.sh --image "$IMAGE" --run --watch   # all three, $5
#   benchmarks/gpu/run_runpod.sh --image "$IMAGE" --gpus h200-sxm --budget 2.00 --run
#   RUNPOD_API_KEY=... benchmarks/gpu/run_runpod.sh --local --image "$IMAGE" \
#       --plan my-plan.json --run
#   RUNPOD_API_KEY=... uv run python benchmarks/gpu/launch.py cleanup   # leftover pods
#
# Exit status:
#   0   a dry run, a run that stayed within its budget, or a run started without
#       --watch
#   1   an error, or a failed workflow run (including charges over budget)
#   2   with --local: the launcher found the charges over budget
#   64  a usage error
set -euo pipefail

WORKFLOW="gpu-bench.yml"
GPU_TYPES="a100-sxm h100-sxm h200-sxm"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
CALLER_DIR=$PWD
cd "$(dirname "$SELF")/../.."

# A path as given on the command line, made absolute from where the script was run
from_caller() {
	case $1 in
	/*) echo "$1" ;;
	*) echo "$CALLER_DIR/$1" ;;
	esac
}

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$SELF"; }
die() {
	echo "gpu/run_runpod.sh: $*" >&2
	exit 64
}
fail() {
	echo "gpu/run_runpod.sh: $*" >&2
	exit 1
}

# An image pinned by digest, as the launcher requires
check_image() {
	case $1 in
	*:sha256-*)
		die "$1 is the image's attestation, not the image; pass ${1%:sha256-*}@sha256:${1##*:sha256-}"
		;;
	*@sha256:*)
		[[ $1 =~ @sha256:[0-9a-f]{64}$ ]] || die "the image's digest should be 64 hex digits: $1"
		;;
	*) die "pass --image by digest (ghcr.io/...@sha256:...), not $1" ;;
	esac
}

# Ask before spending money, unless --yes
confirm() {
	[ "$yes" = 1 ] && return 0
	[ -t 0 ] || die "--run without a terminal needs --yes"
	printf '%s [y/N] ' "$1" >&2
	local answer
	read -r answer
	case $answer in
	y | Y | yes | YES) ;;
	*)
		echo "Nothing created." >&2
		exit 0
		;;
	esac
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

# The launcher's output in a finished run's log, without timestamps
launcher_output() {
	gh run view "$1" --log |
		awk -F'\t' '$2 == "Run the benchmark" { print $3 }' |
		sed -E 's/^[^0-9]*[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z //' |
		sed -n '/^##\[endgroup\]/,$p' | sed '1d'
}

image=
gpus=a100-sxm,h100-sxm,h200-sxm
budget=5.00
cloud=SECURE
run=0
yes=0
watch=0
ref=development
ref_given=0
local_mode=0
plan=
out=
prices=
while [ $# -gt 0 ]; do
	case $1 in
	--image | --gpus | --budget | --cloud | --ref | --plan | --out | --prices)
		[ $# -ge 2 ] || die "$1 needs a value"
		case $1 in
		--image) image=$2 ;;
		--gpus) gpus=$2 ;;
		--budget) budget=$2 ;;
		--cloud) cloud=$2 ;;
		--ref)
			ref=$2
			ref_given=1
			;;
		--plan) plan=$(from_caller "$2") ;;
		--out) out=$(from_caller "$2") ;;
		--prices) prices=$(from_caller "$2") ;;
		esac
		shift 2
		;;
	--run)
		run=1
		shift
		;;
	--yes)
		yes=1
		shift
		;;
	--watch)
		watch=1
		shift
		;;
	--local)
		local_mode=1
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*) die "unknown argument: $1 (see --help)" ;;
	esac
done

[ -n "$image" ] || die "--image is required (see --help)"
check_image "$image"
[ -n "$gpus" ] || die "--gpus needs at least one GPU type"
for gpu in $(echo "$gpus" | tr ',' ' '); do
	case " $GPU_TYPES " in
	*" $gpu "*) ;;
	*) die "unknown GPU type $gpu; choose from: $GPU_TYPES" ;;
	esac
done
[[ $budget =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--budget should be in USD, such as 5.00, not $budget"
case $cloud in
SECURE | COMMUNITY) ;;
*) die "--cloud is SECURE or COMMUNITY, not $cloud" ;;
esac

if [ "$local_mode" = 1 ]; then
	[ "$ref_given" = 0 ] || die "--ref is for the workflow; --local runs this checkout"
	[ "$watch" = 0 ] || die "--watch is for the workflow; --local always runs to the end"
	command -v uv >/dev/null || die "uv is required: https://docs.astral.sh/uv/"
	plan=${plan:-benchmarks/gpu/plan.json}
	[ -f "$plan" ] || die "no plan at $plan"
	if [ -z "${RUNPOD_API_KEY:-}" ] && { [ "$run" = 1 ] || [ -z "$prices" ]; }; then
		die "set RUNPOD_API_KEY to a Runpod API key"
	fi
	args=(run --image "$image" --gpus "$gpus" --budget "$budget" --cloud "$cloud" --plan "$plan")
	if [ -n "$out" ]; then args+=(--out "$out"); fi
	if [ -n "$prices" ]; then args+=(--prices "$prices"); fi
	if [ "$run" = 1 ]; then
		confirm "Create $gpus pods on Runpod's $cloud cloud, for up to \$$budget?"
	else
		args+=(--dry-run)
	fi
	exec uv run python benchmarks/gpu/launch.py "${args[@]}"
fi

[ -z "$plan$out$prices" ] || die "--plan, --out and --prices are for --local; the workflow runs $ref's plan.json"
command -v gh >/dev/null || die "gh is required: https://cli.github.com"
dry_run=true
if [ "$run" = 1 ]; then
	dry_run=false
	confirm "Create $gpus pods on Runpod's $cloud cloud from $ref's plan, for up to \$$budget?"
fi

started=$(a_minute_ago)
gh workflow run "$WORKFLOW" --ref "$ref" -f image="$image" -f gpus="$gpus" \
	-f budget="$budget" -f cloud="$cloud" -f dry-run="$dry_run"
run_id=$(find_run "$WORKFLOW" "$started") ||
	fail "the run started, but didn't appear: gh run list --workflow $WORKFLOW"
echo "Run: $(gh run view "$run_id" --json url --jq .url)" >&2
echo "If the runpod environment needs approval, approve the run on that page." >&2

if [ "$dry_run" = false ] && [ "$watch" = 0 ]; then
	echo "Follow it there, or: gh run watch $run_id" >&2
	echo "Then collect the results: benchmarks/report/collect.sh $run_id" >&2
	exit 0
fi
status=0
gh run watch "$run_id" --exit-status --interval 15 >&2 || status=$?
echo >&2
launcher_output "$run_id"
if [ "$dry_run" = false ]; then
	echo >&2
	echo "Collect the results: benchmarks/report/collect.sh $run_id" >&2
fi
[ "$status" = 0 ] || fail "the run failed: gh run view $run_id --log-failed"

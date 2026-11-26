#!/usr/bin/env bash
# Download a GPU benchmark run's results, then print its tables and draw its charts.
#
# Usage:
#   benchmarks/report/collect.sh [RUN_ID] [--dest DIR] [--baseline PATH]... [--no-charts]
#   benchmarks/report/collect.sh --dir DIR [--baseline PATH]... [--no-charts]
#
# What it does:
#   1. Download: the gpu-benchmark-results artifact of a "GPU benchmark" workflow
#      run, by default the latest that has one. It goes into
#      DEST/<benchmark run ID>/:
#        - launcher.json: the pods, their prices, states and charges
#        - one directory per GPU type, with that pod's results
#      With --dir, the results are already here, so nothing is downloaded. DIR is
#      a run's directory like that, or a single results directory: a pod's, or one
#      from gpu/run_docker.sh or cpu/run.sh.
#   2. Tables: benchmarks/report/tables.py prints the README's tables in Markdown, and
#      saves them as DIR/report.md:
#        - batch sizes
#        - term counts
#        - fits per hour and cost per million fits
#        - sampling
#   3. Charts: benchmarks/report/charts.py draws them into DIR/charts, as SVG
#      for the README and PNG previews:
#        - gpu_scaling_<gpu>_*: batch fitting time and memory, with projections
#        - gpu_sampling_<gpu>_*: sampling throughput and memory
#        - with several devices (the run's GPUs and any --baseline),
#          gpu_scaling_comparison_* and gpu_sampling_comparison_*
#      A CPU's results draw as feasible_scaling_* and cpu_sampling_*.
#
# Options:
#   RUN_ID           The GitHub Actions run, as gh run list --workflow
#                    gpu-bench.yml shows it, or as gpu/run_runpod.sh prints it.
#                    Default: the latest run with results.
#   --dest DIR       Where downloads go. Default: benchmarks/gpu/results.
#   --dir DIR        Results already downloaded or written: skip the download.
#   --baseline PATH  Another results directory to compare with in the comparison
#                    charts, such as the M3's in benchmarks/cpu/results/ or one from
#                    cpu/run.sh. Repeat it for several.
#   --no-charts      Print the tables only.
#   -h, --help       Show this help.
#
# Requires:
#   - uv. The charts add matplotlib for that one command.
#   - gh, signed in, for downloads.
#
# Examples:
#   benchmarks/report/collect.sh                           # the latest run
#   benchmarks/report/collect.sh 45739582867               # a given run
#   benchmarks/report/collect.sh --dir ~/Downloads/20261126T020128Z-5ac6
#   benchmarks/report/collect.sh --baseline benchmarks/cpu/results/20261201T120000Z
#   # Redraw the README's charts from a run
#   benchmarks/report/collect.sh --dir benchmarks/gpu/results/<run> --no-charts &&
#     uv run --with matplotlib python benchmarks/report/charts.py \
#       benchmarks/gpu/results/<run>/*/ docs/source/_static
#
# Exit status:
#   0 when done; 1 if the download, the tables or the charts fail; 64 for a usage error.
set -euo pipefail

WORKFLOW="gpu-bench.yml"
ARTIFACT="gpu-benchmark-results"
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
	echo "report/collect.sh: $*" >&2
	exit 64
}
fail() {
	echo "report/collect.sh: $*" >&2
	exit 1
}

# The latest run of the benchmark workflow whose results artifact hasn't expired
latest_run() {
	local repo id
	repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
	for id in $(gh run list --workflow "$WORKFLOW" --limit 20 --json databaseId --jq '.[].databaseId'); do
		if [ -n "$(gh api "repos/$repo/actions/runs/$id/artifacts" \
			--jq ".artifacts[] | select(.name == \"$ARTIFACT\" and (.expired | not)) | .id")" ]; then
			echo "$id"
			return 0
		fi
	done
	return 1
}

run_id=
dest=benchmarks/gpu/results
dir=
charts=1
baselines=()
while [ $# -gt 0 ]; do
	case $1 in
	--dest | --dir | --baseline)
		[ $# -ge 2 ] || die "$1 needs a path"
		case $1 in
		--dest) dest=$(from_caller "$2") ;;
		--dir) dir=$(from_caller "$2") ;;
		--baseline)
			[ -e "$(from_caller "$2")" ] || die "no results at $2"
			baselines+=("$(from_caller "$2")")
			;;
		esac
		shift 2
		;;
	--no-charts)
		charts=0
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	-*) die "unknown argument: $1 (see --help)" ;;
	*)
		[ -z "$run_id" ] || die "one run at a time: $run_id and $1"
		[[ $1 =~ ^[0-9]+$ ]] || die "a run ID is a number, not $1"
		run_id=$1
		shift
		;;
	esac
done
[ -z "$dir" ] || [ -z "$run_id" ] || die "pass a RUN_ID or --dir, not both"
command -v uv >/dev/null || die "uv is required: https://docs.astral.sh/uv/"

if [ -z "$dir" ]; then
	command -v gh >/dev/null || die "gh is required: https://cli.github.com"
	if [ -z "$run_id" ]; then
		run_id=$(latest_run) || fail "no recent $WORKFLOW run has results"
	fi
	echo "Downloading run $run_id's results" >&2
	download=$(mktemp -d "${TMPDIR:-/tmp}/metalog-bench-results.XXXXXX")
	trap 'rm -rf "$download"' EXIT
	gh run download "$run_id" --name "$ARTIFACT" --dir "$download" ||
		fail "run $run_id has no $ARTIFACT artifact (a dry run, or expired after 90 days)"
	mkdir -p "$dest"
	for got in "$download"/*/; do
		[ -d "$got" ] || continue
		name=$(basename "$got")
		[ ! -e "$dest/$name" ] || fail "$dest/$name exists already; pass --dir $dest/$name to redraw it"
		mv "$got" "$dest/$name"
		dir=$dest/$name
	done
	[ -n "$dir" ] || fail "run $run_id's artifact is empty"
fi
[ -d "$dir" ] || die "no directory $dir"

# The pods' directories: DIR itself if it's one, else those under it
pods=()
if [ -f "$dir/manifest.json" ]; then
	pods=("$dir")
else
	for pod in "$dir"/*/; do
		if [ -f "$pod/manifest.json" ]; then pods+=("${pod%/}"); fi
	done
fi
[ ${#pods[@]} -gt 0 ] || fail "no results in $dir: a results directory has manifest.json"
if [ -f "$dir/launcher.json" ]; then
	uv run python - "$dir/launcher.json" <<'PY' >&2
import json, sys

run = json.load(open(sys.argv[1]))
print(f"Run {run['run_id']}: {run['image']}")
for pod in run["pods"]:
    error = f" ({pod['error']})" if pod["error"] else ""
    print(f"  {pod['gpu']}: {pod['state']}{error}")
billed = run["total_billed"]
cost, kind = (billed, "billed") if billed is not None else (run["total_estimated"], "estimated")
print(f"  cost: ${cost:.2f} {kind}, of a ${run['budget']:.2f} budget")
PY
fi

echo >&2
uv run python benchmarks/report/tables.py "${pods[@]}" | tee "$dir/report.md"
echo >&2
echo "Tables: $dir/report.md" >&2

if [ "$charts" = 1 ]; then
	mkdir -p "$dir/charts"
	uv run --with matplotlib python benchmarks/report/charts.py \
		${baselines[@]+"${baselines[@]}"} "${pods[@]}" "$dir/charts" --png >/dev/null
	echo "Charts: $dir/charts" >&2
fi

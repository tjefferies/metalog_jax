#!/usr/bin/env bash
# Run the benchmark plan on this machine's CPU, from this checkout.
#
# Usage:
#   benchmarks/cpu/run.sh [--minutes N] [--plan FILE] [--out DIR] [--dry-run]
#
# What it does:
#   Runs benchmarks/gpu/run.py, the program a GPU pod runs, on the CPU. It uses
#   the library in this checkout, and measures until the time budget ends:
#     - single batches
#     - a chunked run
#     - term counts
#     - sampling
#   This is how the README's M3 numbers are measured, and the CPU line in the
#   comparison charts.
#
#   Each measurement runs in its own process, so that its peak memory is its own.
#   As in the image, JAX's compilation cache keeps every compilation for the later
#   measurements. Here the cache is a fresh directory for each run, deleted
#   afterwards, so the run doesn't depend on earlier ones.
#
# The plan:
#   The default is benchmarks/cpu/plan.json. It has the GPU plan's measurements
#   at the README's M3 sizes:
#     - single batches of 5 to 20,000 datasets
#     - term counts at 5,000 datasets
#     - sampling up to 10^8 points
#   It also has no correctness phase, since that compares a GPU with the CPU.
#   Don't run the GPU plan here: its batches reach millions of datasets, which need
#   tens of gigabytes. Close other programs, and keep a laptop on power, so that
#   the timings are steady.
#
# Options:
#   --minutes N  The time budget. Default: 60. The phases share it by the plan's
#                shares, and skip what doesn't fit.
#   --plan FILE  The plan to run. Default: benchmarks/cpu/plan.json.
#   --out DIR    Where results go. Default: benchmarks/cpu/results/<UTC time>.
#   --dry-run    Print the environment and command, and run nothing.
#   -h, --help   Show this help.
#
# Requires:
#   uv. The first run installs the locked dependencies.
#
# Results:
#   In --out:
#   - results.jsonl, a record per measurement
#   - manifest.json, the setup: the CPU, the memory, every package's version
#   - plan.json, measurements.log and checksums.sha256
#   Tables and charts: benchmarks/report/collect.sh --dir DIR
#
# Examples:
#   benchmarks/cpu/run.sh                    # an hour, as for the README
#   benchmarks/cpu/run.sh --minutes 10       # a quick look
#
# Exit status:
#   0 when the benchmark ends (manifest.json's "state" says whether it finished
#   or failed), 64 for a usage error, and anything else if it couldn't start.
set -euo pipefail

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
	echo "cpu/run.sh: $*" >&2
	exit 64
}

minutes=60
plan=benchmarks/cpu/plan.json
out=
dry_run=0
while [ $# -gt 0 ]; do
	case $1 in
	--minutes | --plan | --out)
		[ $# -ge 2 ] || die "$1 needs a value"
		case $1 in
		--minutes) minutes=$2 ;;
		--plan) plan=$(from_caller "$2") ;;
		--out) out=$(from_caller "$2") ;;
		esac
		shift 2
		;;
	--dry-run)
		dry_run=1
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*) die "unknown argument: $1 (see --help)" ;;
	esac
done

[[ $minutes =~ ^[0-9]+$ ]] && [ "$minutes" -gt 0 ] || die "--minutes should be a whole number, not $minutes"
[ -f "$plan" ] || die "no plan at $plan"
out=${out:-benchmarks/cpu/results/$(date -u +%Y%m%dT%H%M%SZ)}

settings=(
	BENCH_LOCAL=1
	BENCH_BACKEND=cpu
	JAX_PLATFORMS=cpu
	BENCH_PLAN="$plan"
	BENCH_RESULTS="$out"
	BENCH_TIME_BUDGET_S=$((minutes * 60))
	JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
)
command=(uv run python -m benchmarks.gpu.run)
if [ "$dry_run" = 1 ]; then
	printf '%q ' env "${settings[@]}" JAX_COMPILATION_CACHE_DIR="<a new temporary directory>" "${command[@]}"
	echo
	exit 0
fi
command -v uv >/dev/null || die "uv is required: https://docs.astral.sh/uv/"

cache=$(mktemp -d "${TMPDIR:-/tmp}/metalog-bench-cache.XXXXXX")
trap 'rm -rf "$cache"' EXIT
echo "Benchmarking for $minutes minutes with $plan; results go to $out" >&2
env "${settings[@]}" JAX_COMPILATION_CACHE_DIR="$cache" "${command[@]}"
echo "Tables and charts: benchmarks/report/collect.sh --dir $out" >&2

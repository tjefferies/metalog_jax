#!/usr/bin/env bash
# Run the GPU benchmark's image on this machine's NVIDIA GPU, with Docker.
#
# Usage:
#   benchmarks/gpu/run_docker.sh --image IMAGE [--gpu TYPE] [--device N]
#       [--minutes N] [--plan FILE] [--out DIR] [--dry-run]
#
# What it does:
#   Runs the image a Runpod pod runs, with the same plan, on your own GPU:
#   - The benchmark: it measures until its time budget ends, writes its results to
#     --out, and exits.
#   - Nothing is served or rented, so it costs nothing.
#   - Reproducing a published run: pass that run's image digest and its plan.json,
#     as the README for benchmarks/gpu describes.
#
# The phases share the time budget as on a pod. The plan's batch sizes stop at
# what fits in the GPU's memory. A batch that runs out of memory is recorded as
# "oom", and ends that method's batches.
#
# Options:
#   --image IMAGE  Required. The benchmark image, best by digest
#                  (ghcr.io/<owner>/<repo>/gpu-bench@sha256:<digest>, as
#                  benchmarks/gpu/build_image.sh prints it), so that the run is
#                  reproducible. A tag, or a local image's name, works too.
#   --gpu TYPE     a100-sxm, h100-sxm or h200-sxm. Checks that the device is that
#                  Runpod GPU type, and names the results' directory. Leave it out
#                  for any other card.
#   --device N     Which GPU to use, by index (docker run --gpus device=N).
#                  Default: all of them visible, of which the benchmark uses the first.
#   --minutes N    The time budget. Default: 30.
#   --plan FILE    The plan to run, in place of the image's own plan.json.
#   --out DIR      Where results go. Default:
#                  benchmarks/gpu/results/local-<UTC time>/<TYPE, or gpu>.
#   --dry-run      Print the docker command, and run nothing.
#   -h, --help     Show this help.
#
# Requires:
#   - An NVIDIA GPU, with a driver for CUDA 12 or later. The CUDA libraries come
#     with the image.
#   - Docker, with the NVIDIA Container Toolkit for docker run --gpus.
#   - Pulling the image: it's public on GHCR, about 7 GB uncompressed.
#
# Results:
#   In --out:
#   - results.jsonl, a record per measurement
#   - manifest.json, the setup
#   - plan.json, nvidia-smi.xml and telemetry.csv
#   - measurements.log and checksums.sha256
#   Tables and charts: benchmarks/report/collect.sh --dir DIR
#
# Examples:
#   IMAGE=ghcr.io/tjefferies/metalog_jax/gpu-bench@sha256:...
#   benchmarks/gpu/run_docker.sh --image "$IMAGE"                        # 30 minutes
#   benchmarks/gpu/run_docker.sh --image "$IMAGE" --gpu h100-sxm --minutes 20
#   benchmarks/gpu/run_docker.sh --image "$IMAGE" --plan benchmarks/gpu/results/<run>/h100-sxm/plan.json
#
# Exit status:
#   0 when the benchmark ends (manifest.json's "state" says whether it finished
#   or failed), 64 for a usage error, and docker's status if it couldn't start.
set -euo pipefail

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
	echo "gpu/run_docker.sh: $*" >&2
	exit 64
}

image=
gpu=
device=
minutes=30
plan=
out=
dry_run=0
while [ $# -gt 0 ]; do
	case $1 in
	--image | --gpu | --device | --minutes | --plan | --out)
		[ $# -ge 2 ] || die "$1 needs a value"
		case $1 in
		--image) image=$2 ;;
		--gpu) gpu=$2 ;;
		--device) device=$2 ;;
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

[ -n "$image" ] || die "--image is required (see --help)"
case $image in
*:sha256-*) die "$image is the image's attestation, not the image; pass ${image%:sha256-*}@sha256:${image##*:sha256-}" ;;
esac
if [ -n "$gpu" ]; then
	case " $GPU_TYPES " in
	*" $gpu "*) ;;
	*) die "unknown GPU type $gpu; choose from: $GPU_TYPES, or leave --gpu out" ;;
	esac
fi
[[ $minutes =~ ^[0-9]+$ ]] && [ "$minutes" -gt 0 ] || die "--minutes should be a whole number, not $minutes"
[ -z "$device" ] || [[ $device =~ ^[0-9]+$ ]] || die "--device should be a GPU index, such as 0, not $device"
if [ -n "$plan" ]; then [ -f "$plan" ] || die "no plan at $plan"; fi
out=${out:-benchmarks/gpu/results/local-$(date -u +%Y%m%dT%H%M%SZ)/${gpu:-gpu}}
case $out in
/*) mount=$out ;;
*) mount=$PWD/$out ;;
esac

cmd=(docker run --rm --gpus "${device:+device=}${device:-all}"
	--user "$(id -u):$(id -g)"
	-e BENCH_LOCAL=1
	-e BENCH_TIME_BUDGET_S=$((minutes * 60))
	-e BENCH_IMAGE="$image")
if [ -n "$gpu" ]; then cmd+=(-e BENCH_GPU="$gpu"); fi
if [ -n "$plan" ]; then cmd+=(-e BENCH_PLAN_JSON="$(cat "$plan")"); fi
cmd+=(-v "$mount:/results" "$image")

if [ "$dry_run" = 1 ]; then
	printf '%q ' "${cmd[@]}"
	echo
	exit 0
fi
command -v docker >/dev/null || die "docker is required"
mkdir -p "$mount"
echo "Benchmarking for $minutes minutes; results go to $out" >&2
status=0
"${cmd[@]}" || status=$?
echo "Tables and charts: benchmarks/report/collect.sh --dir $out" >&2
exit "$status"

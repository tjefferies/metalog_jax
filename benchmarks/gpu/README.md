# GPU benchmark

Batch fitting and sampling on NVIDIA A100 SXM, H100 SXM and H200 SXM GPUs, rented from
[Runpod](https://docs.runpod.io/overview) for under $5 a run. Each run is
reproducible: the image is pinned by digest, and every pod records its setup in
`manifest.json`.

## What a run measures

`run.py` is the image's entrypoint. It gives each phase of [`plan.json`](plan.json) its
share of the time until the pod's deadline, and runs each measurement in its own
process (`measure.py`), so that peak memory starts from zero and running out of memory
ends only that measurement:

1. **Check:** the backend, the device and float64. The run stops if any is wrong.
2. **Correctness:** 1,024 fits on the GPU and on the CPU, compared.
3. **Single batches:** 10-term Feasible and OLS `fit_grid` calls, doubling from 1,024
   datasets until the GPU runs out of memory or the phase's time runs out.
4. **A chunked run:** chunks of the throughput knee's size (the smallest batch within
   5% of the best time per dataset) for about 3 minutes.
5. **Term counts:** Feasible at 4, 6, 8, 10 and 12 terms, then all five in one grid.
6. **Sampling:** a jitted `ppf(u)`, `rvs` with both PRNGs, `cdf` and `pdf(cdf(x))`,
   from 10⁶ to 10⁹ points.

Data are drawn on the device (`data.py`): the five families of
`examples/feasible_fits.py`, 200 samples each, summarized at the 105 `DEFAULT_Y`
quantiles. Dataset `j` is drawn from `fold_in(key(seed), j)`, so every GPU fits the same
datasets. Timings exclude drawing the data and compiling.

Each pod's results:

| File | Contents |
|---|---|
| `results.jsonl` | One record per measurement, in the CPU benchmark's schema (`mode`, `N`, `K`, `first_s`, `steady_s`, `per_dataset_ms`, `chunk`) plus `peak_gpu_bytes`, `bytes_limit`, `status`, compile time and solver iterations |
| `manifest.json` | The image, the software (every package's version), the GPU, the host, the environment (without secrets), the plan and its SHA-256, and timestamps |
| `nvidia-smi.xml` | `nvidia-smi -q -x` |
| `telemetry.csv` | GPU utilization, memory, power and clocks every 0.5 s |
| `measurements.log` | Each measurement's stderr |
| `pod.log` | The container's log, saved by the launcher |
| `launcher.json` | One level up: the pods' Runpod records, prices, states and charges |

## One-time setup

1. **Build the image:** run the
   [GPU benchmark image](../../.github/workflows/gpu-bench-image.yml) workflow. It
   pushes `ghcr.io/tjefferies/metalog_jax/gpu-bench:<version>-<sha>`, with a provenance
   attestation, and prints the digest to run.
2. **Make the package public,** in its settings on GitHub, so that Runpod, and anyone
   reproducing a run, can pull it without credentials.
3. **Add the Runpod API key:** as `RUNPOD_API_KEY` in your shell for local launches, or
   as a secret of the `runpod` environment for the
   [GPU benchmark](../../.github/workflows/gpu-bench.yml) workflow.

## Running

```bash
export RUNPOD_API_KEY=...
IMAGE=ghcr.io/tjefferies/metalog_jax/gpu-bench@sha256:...

# Print the GPUs, their live prices, the time cap and the estimated cost; create nothing
uv run python benchmarks/gpu/launch.py run --image "$IMAGE" --dry-run

# Run all three GPUs at once, within $5
uv run python benchmarks/gpu/launch.py run --image "$IMAGE" --budget 5.00 --cloud SECURE

# Rerun one GPU type, for example one that couldn't be placed
uv run python benchmarks/gpu/launch.py run --image "$IMAGE" --gpus h200-sxm

# Terminate any benchmark pod left over
uv run python benchmarks/gpu/launch.py cleanup
```

The launcher uses only the standard library. It gives every pod one time cap, so that
the sum of each GPU's price times the cap stays within 92% of `--budget`. At list prices
on 2026-10-30:

| | A100 SXM | H100 SXM | H200 SXM | All three |
|---|---|---|---|---|
| Secure Cloud, per hour | $1.79 | $3.99 | $5.29 | $11.07 |
| 24-minute cap | $0.72 | $1.60 | $2.12 | **$4.43** |
| Community Cloud, per hour | $1.39 | $2.69 | $3.59 | $7.67 |
| 35-minute cap | $0.81 | $1.57 | $2.09 | **$4.47** |

Results go to `benchmarks/gpu/results/<run id>/<gpu>/`, checked against each pod's
checksums.

**Cost guards:**

- **The pod:** at its deadline, the benchmark stops and serves its results. After the
  launcher acknowledges them, or the download window closes, the pod deletes itself
  with its own `RUNPOD_API_KEY`, or else exits, which stops compute billing.
- **The launcher:** it terminates each pod when its results are in or its cap ends, and
  every pod it created when it exits, on errors, Ctrl-C and SIGTERM included.
- **`cleanup`:** terminates any pod named `metalog-bench-*`.
- **The bill:** after a run, the launcher reads Runpod's charges into `launcher.json`,
  and exits with status 2 if they exceed the budget.

## Reproducing a run

Every published result names its image digest in `manifest.json`. To rerun it:

- **On Runpod:** `launch.py run --image <digest>`, with your own API key.
- **On your own NVIDIA GPU:**

  ```bash
  docker run --rm --gpus all --user "$(id -u):$(id -g)" -e BENCH_LOCAL=1 -e BENCH_GPU=h100-sxm \
      -e BENCH_TIME_BUDGET_S=1800 -v "$PWD/out:/results" \
      ghcr.io/tjefferies/metalog_jax/gpu-bench@sha256:...
  ```

  `BENCH_GPU` checks the device against a Runpod GPU type; leave it out for another
  card.
- **On the CPU,** from a checkout, which refreshes the M3 numbers:

  ```bash
  BENCH_LOCAL=1 BENCH_BACKEND=cpu JAX_PLATFORMS=cpu BENCH_RESULTS=out/m3 \
      BENCH_TIME_BUDGET_S=3600 uv run python -m benchmarks.gpu.run
  ```

## Charts and tables

```bash
# The README's chart for each device, and one comparing them
uv run --with matplotlib python benchmarks/feasible_scaling/plot.py \
    benchmarks/gpu/results/<run id>/*/ docs/source/_static

# The README's tables, in Markdown
uv run python benchmarks/gpu/report.py benchmarks/gpu/results/<run id>/*/
```

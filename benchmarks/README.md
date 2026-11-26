# Benchmarks

How fast metalog_jax fits and samples, measured on a laptop's CPU and on rented NVIDIA
A100, H100 and H200 GPUs. The README's charts and timing tables come from here.

## Layout

| Folder | What's in it |
|---|---|
| [`gpu/`](gpu/README.md) | The benchmark: `run.py` runs a time-boxed plan, and `measure.py` makes each measurement in its own process. Also the GPU plan, the image (`Dockerfile`), the Runpod launcher (`launch.py`) and the GPU scripts. The same program runs on a CPU. |
| [`cpu/`](cpu/) | The plan for a CPU, at sizes a laptop's memory holds, and the script that runs it |
| [`report/`](report/) | Tables (`tables.py`) and charts (`charts.py`) from any run's results, and `collect.sh`, which downloads a run and draws both |

A run writes a results directory: `results.jsonl` (one record per measurement),
`manifest.json` (the setup) and `plan.json`, among others. `gpu/README.md` lists every
file. GPU runs keep theirs in `gpu/results/<run id>/<gpu>/`, and CPU runs in
`cpu/results/<time>/`. The README's CPU chart and tables come from
[`cpu/results/20261126T035035Z`](cpu/results/20261126T035035Z), an M3's run.

## Scripts

Each script documents its options, costs, requirements and examples with `--help`.

| Script | What it does | Cost |
|---|---|---|
| [`gpu/build_image.sh`](gpu/build_image.sh) | Builds and pushes the benchmark's image on GitHub, then prints it by digest | — |
| [`gpu/run_runpod.sh`](gpu/run_runpod.sh) | Runs the benchmark on Runpod, one pod per GPU type. The launcher runs on GitHub, or here with `--local`. A dry run unless `--run`. | Up to `--budget` |
| [`gpu/run_docker.sh`](gpu/run_docker.sh) | Runs the image on your own NVIDIA GPU | — |
| [`cpu/run.sh`](cpu/run.sh) | Runs the CPU plan on this machine, from this checkout | — |
| [`report/collect.sh`](report/collect.sh) | Downloads a run's results, prints its tables and draws its charts | — |

On Runpod:

```bash
IMAGE=$(benchmarks/gpu/build_image.sh | tail -n 1)        # after changing the library or gpu/*.py
benchmarks/gpu/run_runpod.sh --image "$IMAGE"              # dry run: prices, time cap, cost
benchmarks/gpu/run_runpod.sh --image "$IMAGE" --run --watch
benchmarks/report/collect.sh                               # tables and charts
```

On this machine's CPU:

```bash
benchmarks/cpu/run.sh --out benchmarks/cpu/results/m3      # an hour
benchmarks/report/collect.sh --dir benchmarks/cpu/results/m3
```

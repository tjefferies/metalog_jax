"""Print the README's benchmark tables, in Markdown, from run results.

Usage: uv run python benchmarks/gpu/report.py RESULTS_DIR [RESULTS_DIR ...]

Each RESULTS_DIR is a pod's (or a local run's) directory from benchmarks/gpu/run.py,
such as ``benchmarks/gpu/results/<run id>/h100-sxm``. A ``launcher.json`` beside it
gives the pod's billed hourly price. Prints, per device, the tables the README has
for the M3 (by batch size and by term count), then one table across devices.
"""

import json
import sys
from pathlib import Path

GIB = 2**30
MISSING = "\N{EN DASH}"  # a table cell without a result


def load(directory: Path) -> dict:
    """A run's manifest, records and hourly price."""
    manifest = json.loads((directory / "manifest.json").read_text())
    records = [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text().splitlines()
    ]
    price = None
    launcher = directory.parent / "launcher.json"
    if launcher.exists():
        for pod in json.loads(launcher.read_text())["pods"]:
            if pod["gpu"] == manifest.get("gpu_type"):
                price = (pod.get("pod") or {}).get("cost") or pod.get("list_price")
    gpus = manifest.get("gpu", {}).get("gpus")
    name = gpus[0]["name"] if gpus else f"{manifest['host']['cpu_model']} (CPU)"
    return {"name": name, "on_gpu": bool(gpus), "records": records, "price": price}


def seconds(s: float) -> str:
    """A duration with sensible precision."""
    return f"{s:.2f} s" if s < 10 else f"{s:.1f} s" if s < 100 else f"{s:.0f} s"


def per_dataset(ms: float) -> str:
    """A time per dataset in ms or µs."""
    return f"{ms:.2f} ms" if ms >= 0.1 else f"{1e3 * ms:.1f} µs"


def memory(record: dict, on_gpu: bool) -> str:
    """A record's peak memory, the device's on a GPU and the host's on a CPU."""
    value = record.get("peak_gpu_bytes" if on_gpu else "host_peak_bytes")
    return MISSING if value is None else f"{value / GIB:.2f} GB"


def ok(records: list[dict], **match) -> list[dict]:
    """The successful records matching every key."""
    return [
        r
        for r in records
        if r["status"] == "ok" and all(r.get(k) == v for k, v in match.items())
    ]


def batch_table(run: dict) -> str:
    """Datasets, time, time per dataset and peak memory: single batches, then chunked."""
    rows = [
        "| Datasets (10 terms) | Time | Per dataset | Peak memory |",
        "|---|---|---|---|",
    ]
    for r in sorted(
        ok(run["records"], phase="batches", mode="feasible"), key=lambda r: r["N"]
    ):
        rows.append(
            f"| {r['N']:,} | {seconds(r['steady_s'])} | {per_dataset(r['per_dataset_ms'])} "
            f"| {memory(r, run['on_gpu'])} |"
        )
    for r in ok(run["records"], mode="feasible_chunked"):
        rows.append(
            f"| {r['N']:,} ({r['N'] // r['chunk']} chunks of {r['chunk']:,}) "
            f"| {seconds(r['fit_s_excl_compile'])} | {per_dataset(r['per_dataset_ms'])} "
            f"| {memory(r, run['on_gpu'])} |"
        )
    skipped = [
        r for r in run["records"] if r.get("phase") == "batches" and r["status"] != "ok"
    ]
    note = "".join(
        f"\n\n{r['N']:,} datasets ({r.get('method')}): {r['status']}" for r in skipped
    )
    return "\n".join(rows) + note


def terms_table(run: dict) -> str:
    """Time per dataset at each term count, and for one grid over all of them."""
    fits = sorted(
        ok(run["records"], phase="terms", mode="feasible"), key=lambda r: r["K"]
    )
    if not fits:
        return "(no term-count results)"
    n = fits[0]["N"]
    header = "| Terms | " + " | ".join(str(r["K"]) for r in fits) + " |"
    rule = "|---|" + "---|" * len(fits)
    row = (
        "| Per dataset | "
        + " | ".join(per_dataset(r["per_dataset_ms"]) for r in fits)
        + " |"
    )
    lines = [f"At {n:,} datasets:", "", header, rule, row]
    for r in ok(run["records"], phase="terms", mode="feasible_terms"):
        lines += [
            "",
            f"All {len(r['num_terms'])} term counts in one grid: {per_dataset(r['per_dataset_ms'])} per dataset.",
        ]
    return "\n".join(lines)


def throughput(records: list[dict], method: str) -> dict | None:
    """The best sustained record of a method: the chunked run, else the best batch."""
    chunked = ok(records, mode=f"{method}_chunked")
    if chunked:
        return chunked[0]
    mode = "feasible" if method == "feasible" else "ols_grid"
    batches = ok(records, phase="batches", mode=mode)
    return min(batches, key=lambda r: r["per_dataset_ms"]) if batches else None


def summary_table(runs: list[dict]) -> str:
    """Fits per hour, the largest batch, and the cost per million fits, by device."""
    rows = [
        "| Device | Feasible fits per hour | OLS fits per hour | Largest Feasible batch | Cost per million Feasible fits |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        cells = [run["name"]]
        for method in ("feasible", "ols"):
            best = throughput(run["records"], method)
            cells.append(
                f"{3.6e6 / best['per_dataset_ms'] / 1e6:,.1f} million"
                if best
                else MISSING
            )
        largest = ok(run["records"], phase="batches", mode="feasible")
        cells.append(f"{max(r['N'] for r in largest):,}" if largest else MISSING)
        best = throughput(run["records"], "feasible")
        if best and run["price"]:
            cells.append(f"${run['price'] * best['per_dataset_ms'] * 1e6 / 3.6e6:.2f}")
        else:
            cells.append(MISSING)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def sampling_table(runs: list[dict]) -> str:
    """Millions of points per second at each kind's largest successful size."""
    kinds = ("ppf_jit", "rvs_jax", "rvs_hdr", "cdf", "pdf_cdf")
    names = ("Jitted ppf(u)", "rvs, jax.random", "rvs, HDR", "cdf", "pdf(cdf(x))")
    rows = ["| Device | " + " | ".join(names) + " |", "|---|" + "---|" * len(kinds)]
    for run in runs:
        cells = [run["name"]]
        for kind in kinds:
            done = ok(run["records"], mode=kind)
            best = max(done, key=lambda r: r["N"]) if done else None
            cells.append(
                f"{best['points_per_s'] / 1e6:,.1f}M/s at {best['N']:.0e}"
                if best
                else MISSING
            )
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    """Print every table."""
    runs = [load(Path(a)) for a in argv[1:]]
    for run in runs:
        print(f"## {run['name']}\n\n{batch_table(run)}\n\n{terms_table(run)}\n")
    print(f"## Across devices\n\n{summary_table(runs)}\n\n{sampling_table(runs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

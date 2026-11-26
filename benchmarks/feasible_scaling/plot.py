"""Render the Feasible scaling charts (light and dark SVG) from benchmark results.

Usage:
    uv run --with matplotlib python benchmarks/feasible_scaling/plot.py INPUT... OUT_DIR [--png]

Each INPUT is either the CPU benchmark's ``results.json``, which gives
``feasible_scaling_{light,dark}.svg``, or a results directory of
``benchmarks/gpu/run.py`` (``results.jsonl`` and ``manifest.json``). A GPU's
directory gives ``gpu_scaling_<gpu>_{light,dark}.svg``, and a CPU run's gives
``feasible_scaling_{light,dark}.svg``. With more than one input, the inputs' Feasible
times per dataset are also drawn together, in ``gpu_scaling_comparison_*.svg``.

A run directory's Feasible points include the term-count phase's fits at the plan's
term count, and a line is projected only from MIN_PROJECTION_POINTS measured points:
for time, those from the throughput knee on.
"""

import dataclasses
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        feasible="#2a78d6",
        ols="#eb6834",
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        feasible="#3987e5",
        ols="#d95926",
    ),
}
# Lines for the comparison chart, one per device
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1f9d6b", "#8a5cd1", "#c43c5c")
KNEE_TOLERANCE = 0.05  # benchmarks/gpu/plan.json's chunked knee_tolerance
GIB = 2**30
# A line is projected from at least this many measured points; fewer are only joined
MIN_PROJECTION_POINTS = 3
# The time projections end at most this far up their log axis, so that their labels
# stay under the panel's title
END_FRACTION = 0.91


@dataclasses.dataclass
class Device:
    """One device's results, in the form the chart draws them.

    Times are in seconds and memory in GB. Single-batch records have ``N``,
    ``steady_s`` and ``memory_gb``; the chunked run also has ``chunk`` and
    ``fit_s_excl_compile``.
    """

    stem: str
    label: str
    machine: str
    feasible: list[dict]
    ols: list[dict]
    chunk: dict | None
    memory_limit_gb: float
    memory_name: str
    memory_title: str
    ols_label: str
    n_max: int
    fit_from: dict[str, int]
    oom: dict[str, int]


def compact(v, _pos=None):
    """1,000 -> 1K, 1,000,000 -> 1M."""
    if v >= 1e6:
        return f"{v / 1e6:.3g}M"
    if v >= 1e3:
        return f"{v / 1e3:.3g}K"
    return f"{v:.3g}"


def duration(v):
    """Round a duration in seconds to a readable label."""
    if v >= 3600:
        return f"{v / 3600:.1f} h"
    if v >= 60:
        return f"{v / 60:.0f} min"
    return f"{v:.0f} s"


def linfit(n, v):
    """Least-squares v = a + b * n."""
    b, a = np.polyfit(np.asarray(n, float), np.asarray(v, float), 1)
    return float(a), float(b)


def load_cpu_json(path: Path) -> Device:
    """The CPU benchmark's results.json, with its hand-written machine block."""
    res = json.loads(path.read_text())
    spec = res["machine"]
    fe = sorted(
        (r for r in res["runs"] if r["mode"] == "feasible" and r["K"] == 10),
        key=lambda r: r["N"],
    )
    ols = sorted(
        (r for r in res["runs"] if r["mode"] == "ols_grid"), key=lambda r: r["N"]
    )
    chunk = next(r for r in res["runs"] if r["mode"] == "feasible_chunked")
    for r in [*fe, *ols, chunk]:
        r["memory_gb"] = r["rss_peak_mb"] / 1024
    return Device(
        stem="feasible_scaling",
        label=f"{spec['cpu']} CPU",
        machine=(
            f"{spec['model']}, {spec['cpu']} ({spec['cores']}), {spec['ram_gb']:g} GB RAM, "
            f"{spec['os']}, JAX {spec['jax']} on {spec['backend']}"
        ),
        feasible=fe,
        ols=ols,
        chunk=chunk,
        memory_limit_gb=spec["ram_gb"],
        memory_name="Machine RAM",
        memory_title="Peak memory, single batch",
        ols_label="OLS (fit_grid; no validity check)",
        n_max=1_000_000,
        # Smaller batches are overhead-bound
        fit_from={"feasible": 500, "ols": 500},
        oom={},
    )


def _knee_n(records: list[dict]) -> int:
    """The smallest batch within KNEE_TOLERANCE of the best time per dataset."""
    best = min(r["steady_s"] / r["N"] for r in records)
    return min(
        r["N"] for r in records if r["steady_s"] / r["N"] <= best * (1 + KNEE_TOLERANCE)
    )


def load_run_dir(path: Path) -> Device:
    """A results directory of benchmarks/gpu/run.py, on a GPU or the CPU."""
    manifest = json.loads((path / "manifest.json").read_text())
    records = [
        json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()
    ]
    gpu_type = manifest.get("gpu_type")
    on_gpu = bool(manifest.get("gpu", {}).get("gpus"))
    memory_key = "peak_gpu_bytes" if on_gpu else "host_peak_bytes"
    num_terms = manifest["plan"]["num_terms"]

    def batches(mode):
        rows = [
            r
            for r in records
            if r.get("phase") == "batches"
            and r.get("mode") == mode
            and r["status"] == "ok"
        ]
        if mode == "feasible":  # the term-count phase's single batches, too
            sizes = {r["N"] for r in rows}
            rows += [
                r
                for r in records
                if r.get("phase") == "terms"
                and r.get("mode") == mode
                and r.get("K") == num_terms
                and not r.get("num_terms")
                and r["status"] == "ok"
                and r["N"] not in sizes
            ]
        for r in rows:
            r["memory_gb"] = r[memory_key] / GIB
        return sorted(rows, key=lambda r: r["N"])

    def first_oom(method):
        rows = [
            r
            for r in records
            if r.get("phase") == "batches" and r.get("method") == method
        ]
        return next((r["N"] for r in rows if r["status"] == "oom"), None)

    fe, ols = batches("feasible"), batches("ols_grid")
    chunk = next(
        (
            r
            for r in records
            if r.get("mode") == "feasible_chunked" and r["status"] == "ok"
        ),
        None,
    )
    if chunk:
        chunk["memory_gb"] = chunk[memory_key] / GIB
    packages = manifest["software"]["packages"]
    if on_gpu:
        card = manifest["gpu"]["gpus"][0]
        limit = max(r["bytes_limit"] for r in fe + ols) / GIB
        cloud = manifest["environment"].get("RUNPOD_DC_ID")
        machine = (
            f"{card['name']}, driver {manifest['gpu']['driver']}, "
            f"CUDA {manifest['gpu']['cuda']}, JAX {packages['jax']}"
            + (f", Runpod {cloud}" if cloud else "")
            + (
                f", image {manifest['image'].split(':')[-1][:12]}"
                if manifest.get("image")
                else ""
            )
        )
        label, name, title = card["name"], "GPU memory", "Peak GPU memory, single batch"
    else:
        mem_bytes = manifest["host"].get("mem_total_bytes")
        limit = (
            mem_bytes / GIB
            if mem_bytes
            else max(r["memory_gb"] for r in fe + ols) * 1.5
        )
        machine = (
            f"{manifest['host']['cpu_model']}, {manifest['host']['cpu_count']} cores, "
            f"{limit:.0f} GB RAM, {manifest['host']['platform']}, JAX {packages['jax']} on CPU"
        )
        label, name, title = (
            f"{manifest['host']['cpu_model']} CPU",
            "Machine RAM",
            "Peak memory, single batch",
        )
    return Device(
        stem=f"gpu_scaling_{gpu_type}" if gpu_type else "feasible_scaling",
        label=label,
        machine=machine,
        feasible=fe,
        ols=ols,
        chunk=chunk,
        memory_limit_gb=limit,
        memory_name=name,
        memory_title=title,
        ols_label="OLS (fit_grid)",
        n_max=100_000_000 if on_gpu else 1_000_000,
        fit_from={"feasible": _knee_n(fe), "ols": _knee_n(ols)},
        oom={
            k: v
            for k, v in (("feasible", first_oom("feasible")), ("ols", first_oom("ols")))
            if v
        },
    )


def models(d: Device) -> tuple[dict, dict]:
    """Time and memory, fitted as a + b * N on each method's linear range.

    Time is fitted from the throughput knee on, with the chunked run if its chunks
    are that large; memory, which has no knee, on at least the three largest batches.
    A method with fewer than MIN_PROJECTION_POINTS points gets None.
    """
    t_model, m_model = {}, {}
    for key, pts in (("feasible", d.feasible), ("ols", d.ols)):
        big = [r for r in pts if r["N"] >= d.fit_from[key]]
        n, t = [r["N"] for r in big], [r["steady_s"] for r in big]
        if key == "feasible" and d.chunk and d.chunk["chunk"] >= d.fit_from[key]:
            n, t = [*n, d.chunk["N"]], [*t, d.chunk["fit_s_excl_compile"]]
        t_model[key] = None
        if len(n) >= MIN_PROJECTION_POINTS:
            a, b = linfit(n, t)
            t_model[key] = {"a": a, "b": b}
        sizes = sorted(r["N"] for r in pts)
        start = min(d.fit_from[key], *sizes[-MIN_PROJECTION_POINTS:])
        fitted = [r for r in pts if r["N"] >= start]
        m_model[key] = None
        if len(fitted) >= MIN_PROJECTION_POINTS:
            c0, d0 = linfit(
                [r["N"] for r in fitted], [r["memory_gb"] * 1024 for r in fitted]
            )
            m_model[key] = {"c_mb": c0, "d_mb": d0, "from": start}
    return t_model, m_model


def style_axes(ax, c):
    """Recessive axes: hairline solid grid, no top/right spines, muted ticks."""
    ax.set_facecolor(c["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=c["muted"], labelsize=9, length=0, pad=6)
    ax.grid(True, which="major", color=c["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def _time_ticks(lo: float, hi: float) -> tuple[list[float], list[str]]:
    """Duration ticks from 10 ms to 10 h, within (lo, hi)."""
    ticks = [
        (0.01, "10 ms"),
        (0.1, "0.1 s"),
        (1, "1 s"),
        (10, "10 s"),
        (60, "1 min"),
        (600, "10 min"),
        (3600, "1 h"),
        (36000, "10 h"),
    ]
    kept = [(t, s) for t, s in ticks if lo <= t <= hi]
    return [t for t, _ in kept], [s for _, s in kept]


def render(d: Device, theme: str, out_dir: Path, t_model: dict, m_model: dict) -> Path:
    """Draw both panels in one theme and save the SVG (and optional PNG preview)."""
    c = THEMES[theme]
    plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none"})
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(11, 4.9), facecolor=c["surface"])
    fig.subplots_adjust(left=0.075, right=0.975, top=0.74, bottom=0.13, wspace=0.26)
    series = (
        ("ols", d.ols, c["ols"]),
        ("feasible", d.feasible, c["feasible"]),
    )  # Feasible on top
    n_min = min(r["N"] for r in d.feasible + d.ols)
    x_lo, x_text = n_min * 0.7, n_min * 0.9
    n_max = d.n_max
    limit = d.memory_limit_gb

    # ---- Panel 1: time ------------------------------------------------------------
    style_axes(ax_t, c)
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    for key, pts, col in series:
        if t_model[key] is None:  # too few points to project: join them
            ax_t.plot(
                [p["N"] for p in pts], [p["steady_s"] for p in pts], color=col, lw=2
            )
        else:
            a, b = t_model[key]["a"], t_model[key]["b"]
            n_hi = (
                d.chunk["N"]
                if key == "feasible" and d.chunk
                else max(p["N"] for p in pts)
            )
            n_solid = np.geomspace(d.fit_from[key], n_hi, 100)
            n_dash = np.geomspace(n_hi, n_max, 100)
            ax_t.plot(
                n_solid, a + b * n_solid, color=col, linewidth=2, solid_capstyle="round"
            )
            ax_t.plot(
                n_dash, a + b * n_dash, color=col, linewidth=2, linestyle=(0, (4, 3))
            )
        ax_t.scatter(
            [p["N"] for p in pts],
            [p["steady_s"] for p in pts],
            s=40,
            color=col,
            edgecolor=c["surface"],
            linewidth=2,
            zorder=3,
        )
    if d.chunk:
        ax_t.scatter(
            [d.chunk["N"]],
            [d.chunk["fit_s_excl_compile"]],
            s=46,
            facecolor=c["surface"],
            edgecolor=c["feasible"],
            linewidth=2,
            zorder=4,
        )
        ax_t.annotate(
            f"{compact(d.chunk['N'])} in chunks of {compact(d.chunk['chunk'])}",
            (d.chunk["N"], d.chunk["fit_s_excl_compile"]),
            xytext=(-10, 8),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=c["ink2"],
        )
    ends = []
    for key, name, dy in (("feasible", "Feasible", 8), ("ols", "OLS", -40)):
        if t_model[key] is None:
            continue
        y_end = t_model[key]["a"] + t_model[key]["b"] * n_max
        ends.append(y_end)
        ax_t.annotate(
            f"{name}\n~{duration(y_end)} for {compact(n_max)}",
            (n_max, y_end),
            xytext=(-2, dy),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=c["ink"],
        )
    y_values = [p["steady_s"] for p in d.feasible + d.ols]
    y_lo = 0.05 if min(y_values) > 0.05 else min(y_values) * 0.5
    y_hi = max(
        [3600 * 1.6, max(y_values) * 1.6]
        + [y_lo * (e / y_lo) ** (1 / END_FRACTION) for e in ends]
    )
    ax_t.set_yticks(*_time_ticks(y_lo, y_hi))
    ax_t.set_ylim(y_lo, y_hi)
    ax_t.set_xlim(x_lo, n_max * 1.3)
    ax_t.xaxis.set_major_formatter(FuncFormatter(compact))
    ax_t.set_xlabel("Datasets fitted", color=c["ink2"], fontsize=9.5)
    ax_t.set_title(
        "Time to fit, 10 terms (compilation excluded)",
        loc="left",
        color=c["ink"],
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    # ---- Panel 2: memory ---------------------------------------------------------
    style_axes(ax_m, c)
    ax_m.set_xscale("log")
    for key, pts, col in series:
        if m_model[key] is None:  # too few points to project: join them
            ax_m.plot(
                [p["N"] for p in pts], [p["memory_gb"] for p in pts], color=col, lw=2
            )
        else:
            cc, dd = m_model[key]["c_mb"], m_model[key]["d_mb"]
            n_hi = max(p["N"] for p in pts)
            n_end = (limit * 1024 - cc) / dd
            n_solid = np.geomspace(m_model[key]["from"], n_hi, 100)
            n_dash = np.geomspace(n_hi, n_end, 100)
            ax_m.plot(n_solid, (cc + dd * n_solid) / 1024, color=col, linewidth=2)
            ax_m.plot(
                n_dash,
                (cc + dd * n_dash) / 1024,
                color=col,
                linewidth=2,
                linestyle=(0, (4, 3)),
            )
        ax_m.scatter(
            [p["N"] for p in pts],
            [p["memory_gb"] for p in pts],
            s=40,
            color=col,
            edgecolor=c["surface"],
            linewidth=2,
            zorder=3,
        )
        if key in d.oom:
            ax_m.scatter(
                [d.oom[key]],
                [limit],
                s=60,
                marker="x",
                color=col,
                linewidth=2,
                zorder=4,
            )
    ax_m.axhline(limit, color=c["muted"], linewidth=1)
    ax_m.annotate(
        f"{d.memory_name} ({limit:.3g} GB)",
        (x_text, limit),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=8.5,
        color=c["ink2"],
    )
    if m_model["feasible"]:
        n_cap = (limit * 1024 - m_model["feasible"]["c_mb"]) / m_model["feasible"][
            "d_mb"
        ]
        ax_m.annotate(
            f"Feasible: one batch reaches {limit:.3g} GB\nnear {compact(round(n_cap, -3))} datasets",
            (x_text, 0.9125 * limit),
            fontsize=8.5,
            color=c["ink2"],
            va="top",
        )
    if d.chunk:
        ax_m.annotate(
            f"Chunking keeps memory flat:\n{compact(d.chunk['N'])} datasets in chunks of "
            f"{compact(d.chunk['chunk'])}\npeaked at {d.chunk['memory_gb']:.2g} GB",
            (x_text, 0.65 * limit),
            fontsize=8.5,
            color=c["ink2"],
            va="top",
        )
    ax_m.set_xlim(x_lo, n_max * 1.3)
    ax_m.set_ylim(0, limit * 1.12)
    ax_m.xaxis.set_major_formatter(FuncFormatter(compact))
    ax_m.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g} GB"))
    ax_m.set_xlabel("Datasets in one batch", color=c["ink2"], fontsize=9.5)
    ax_m.set_title(
        d.memory_title,
        loc="left",
        color=c["ink"],
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    # ---- Header: title, machine, legend ---------------------------------------------
    fig.text(
        0.075,
        0.955,
        "Batch fitting with MetalogFitMethod.Feasible",
        color=c["ink"],
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.905,
        f"{d.machine}. Dots are measured; dashed lines are projected.",
        color=c["ink2"],
        fontsize=9,
    )
    handles = [
        plt.Line2D(
            [],
            [],
            color=c["feasible"],
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=c["surface"],
            label="Feasible (fit_grid)",
        ),
        plt.Line2D(
            [],
            [],
            color=c["ols"],
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=c["surface"],
            label=d.ols_label,
        ),
    ]
    leg = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.068, 0.885),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.8,
    )
    for text in leg.get_texts():
        text.set_color(c["ink2"])

    out = out_dir / f"{d.stem}_{theme}.svg"
    fig.savefig(out, facecolor=c["surface"])
    if "--png" in sys.argv:
        fig.savefig(out.with_suffix(".png"), facecolor=c["surface"], dpi=110)
    plt.close(fig)
    return out


def render_comparison(devices: list[Device], theme: str, out_dir: Path) -> Path:
    """Draw each device's Feasible time per dataset against the batch size."""
    c = THEMES[theme]
    plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=c["surface"])
    fig.subplots_adjust(left=0.12, right=0.97, top=0.8, bottom=0.14)
    style_axes(ax, c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for d, color in zip(devices, SERIES_COLORS, strict=False):
        n = [r["N"] for r in d.feasible]
        ms = [1e3 * r["steady_s"] / r["N"] for r in d.feasible]
        ax.plot(
            n,
            ms,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=c["surface"],
            label=d.label,
        )
    ax.xaxis.set_major_formatter(FuncFormatter(compact))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g} ms"))
    ax.set_xlabel("Datasets in one batch", color=c["ink2"], fontsize=9.5)
    leg = ax.legend(frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(c["ink2"])
    fig.text(
        0.12,
        0.92,
        "Time per dataset, Feasible, 10 terms",
        color=c["ink"],
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(
        0.12,
        0.865,
        "Steady state, compilation excluded. Lower is faster.",
        color=c["ink2"],
        fontsize=9,
    )
    out = out_dir / f"gpu_scaling_comparison_{theme}.svg"
    fig.savefig(out, facecolor=c["surface"])
    if "--png" in sys.argv:
        fig.savefig(out.with_suffix(".png"), facecolor=c["surface"], dpi=110)
    plt.close(fig)
    return out


def load(path: Path) -> Device:
    """A device's results, from a results.json file or a run's directory."""
    return load_run_dir(path) if path.is_dir() else load_cpu_json(path)


def main(argv: list[str]) -> int:
    """Draw every input's chart, and a comparison of several."""
    paths = [Path(a) for a in argv[1:] if a != "--png"]
    *inputs, out_dir = paths
    devices = [load(p) for p in inputs]
    for d in devices:
        t_model, m_model = models(d)
        print(
            json.dumps(
                {"device": d.stem, "time_model": t_model, "memory_model": m_model},
                indent=1,
            )
        )
        for theme in THEMES:
            print(render(d, theme, out_dir, t_model, m_model))
    if len(devices) > 1:
        for theme in THEMES:
            print(render_comparison(devices, theme, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

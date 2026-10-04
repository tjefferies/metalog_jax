"""Render the Feasible scaling chart (light and dark SVG) from benchmark results.

Usage: uv run --with matplotlib python benchmarks/feasible_scaling/plot.py RESULTS_JSON OUT_DIR [--png]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

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

res = json.loads(Path(sys.argv[1]).read_text())
out_dir = Path(sys.argv[2])
spec = res["machine"]
fe = sorted(
    (r for r in res["runs"] if r["mode"] == "feasible" and r["K"] == 10),
    key=lambda r: r["N"],
)
ols = sorted((r for r in res["runs"] if r["mode"] == "ols_grid"), key=lambda r: r["N"])
chunk = next(r for r in res["runs"] if r["mode"] == "feasible_chunked")


def linfit(n, v):
    """Least-squares v = a + b * n."""
    b, a = np.polyfit(np.asarray(n, float), np.asarray(v, float), 1)
    return float(a), float(b)


# Projection models, fitted on batches of >= 500 datasets (smaller ones are overhead-bound)
big = [r for r in fe if r["N"] >= 500]
a_f, b_f = linfit(
    [r["N"] for r in big] + [chunk["N"]],
    [r["steady_s"] for r in big] + [chunk["fit_s_excl_compile"]],
)
a_o, b_o = linfit(
    [r["N"] for r in ols if r["N"] >= 500],
    [r["steady_s"] for r in ols if r["N"] >= 500],
)
t_model = {"feasible": {"a": a_f, "b": b_f}, "ols": {"a": a_o, "b": b_o}}
m_model = {}
for key, pts in (("feasible", big), ("ols", [r for r in ols if r["N"] >= 500])):
    c0, d0 = linfit([r["N"] for r in pts], [r["rss_peak_mb"] for r in pts])
    m_model[key] = {"c_mb": c0, "d_mb": d0}
print(json.dumps({"time_model": t_model, "memory_model": m_model}, indent=1))
N_MAX = 1_000_000
RAM_GB = spec["ram_gb"]


def compact(v, _pos=None):
    """1,000 -> 1K, 1,000,000 -> 1M."""
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}K"
    return f"{v:g}"


def duration(v):
    """Round a duration in seconds to a readable label."""
    if v >= 3600:
        return f"{v / 3600:.1f} h"
    if v >= 60:
        return f"{v / 60:.0f} min"
    return f"{v:.0f} s"


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


def render(theme: str) -> Path:
    """Draw both panels in one theme and save the SVG (and optional PNG preview)."""
    c = THEMES[theme]
    plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none"})
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(11, 4.9), facecolor=c["surface"])
    fig.subplots_adjust(left=0.075, right=0.975, top=0.74, bottom=0.13, wspace=0.26)
    n_fit_lo = 500  # models are fitted on batches of >= 500 datasets
    series = (
        ("ols", ols, c["ols"]),
        ("feasible", fe, c["feasible"]),
    )  # Feasible on top

    # ---- Panel 1: time ------------------------------------------------------------
    style_axes(ax_t, c)
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    for key, pts, col in series:
        a, b = t_model[key]["a"], t_model[key]["b"]
        n_hi = chunk["N"] if key == "feasible" else max(p["N"] for p in pts)
        n_solid = np.geomspace(n_fit_lo, n_hi, 100)
        n_dash = np.geomspace(n_hi, N_MAX, 100)
        ax_t.plot(
            n_solid, a + b * n_solid, color=col, linewidth=2, solid_capstyle="round"
        )
        ax_t.plot(n_dash, a + b * n_dash, color=col, linewidth=2, linestyle=(0, (4, 3)))
        ax_t.scatter(
            [p["N"] for p in pts],
            [p["steady_s"] for p in pts],
            s=40,
            color=col,
            edgecolor=c["surface"],
            linewidth=2,
            zorder=3,
        )
    ax_t.scatter(
        [chunk["N"]],
        [chunk["fit_s_excl_compile"]],
        s=46,
        facecolor=c["surface"],
        edgecolor=c["feasible"],
        linewidth=2,
        zorder=4,
    )
    ax_t.annotate(
        f"{compact(chunk['N'])} in chunks of {compact(chunk['chunk'])}",
        (chunk["N"], chunk["fit_s_excl_compile"]),
        xytext=(-10, 8),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=c["ink2"],
    )
    for key, name, dy in (("feasible", "Feasible", 8), ("ols", "OLS", -40)):
        y_end = t_model[key]["a"] + t_model[key]["b"] * N_MAX
        ax_t.annotate(
            f"{name}\n~{duration(y_end)} for 1M",
            (N_MAX, y_end),
            xytext=(-2, dy),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=c["ink"],
        )
    ticks = [0.1, 1, 10, 60, 600, 3600]
    ax_t.set_yticks(ticks, ["0.1 s", "1 s", "10 s", "1 min", "10 min", "1 h"])
    ax_t.set_ylim(0.05, 3600 * 1.6)
    ax_t.set_xlim(3.5, N_MAX * 1.3)
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
        cc, dd = m_model[key]["c_mb"], m_model[key]["d_mb"]
        n_hi = max(p["N"] for p in pts)
        n_end = (RAM_GB * 1024 - cc) / dd
        n_solid = np.geomspace(n_fit_lo, n_hi, 100)
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
            [p["rss_peak_mb"] / 1024 for p in pts],
            s=40,
            color=col,
            edgecolor=c["surface"],
            linewidth=2,
            zorder=3,
        )
    n_cap = (RAM_GB * 1024 - m_model["feasible"]["c_mb"]) / m_model["feasible"]["d_mb"]
    ax_m.axhline(RAM_GB, color=c["muted"], linewidth=1)
    ax_m.annotate(
        f"Machine RAM ({RAM_GB:g} GB)",
        (4.5, RAM_GB),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=8.5,
        color=c["ink2"],
    )
    ax_m.annotate(
        f"Feasible: one batch reaches {RAM_GB:g} GB\nnear {compact(round(n_cap, -3))} datasets",
        (4.5, 7.3),
        fontsize=8.5,
        color=c["ink2"],
        va="top",
    )
    ax_m.annotate(
        f"Chunking keeps memory flat:\n{compact(chunk['N'])} datasets in chunks of "
        f"{compact(chunk['chunk'])}\npeaked at {chunk['rss_peak_mb'] / 1024:.2g} GB",
        (4.5, 5.2),
        fontsize=8.5,
        color=c["ink2"],
        va="top",
    )
    ax_m.set_xlim(3.5, N_MAX * 1.3)
    ax_m.set_ylim(0, RAM_GB * 1.12)
    ax_m.xaxis.set_major_formatter(FuncFormatter(compact))
    ax_m.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g} GB"))
    ax_m.set_xlabel("Datasets in one batch", color=c["ink2"], fontsize=9.5)
    ax_m.set_title(
        "Peak memory, single batch",
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
        f"{spec['model']}, {spec['cpu']} ({spec['cores']}), {RAM_GB:g} GB RAM, "
        f"{spec['os']}, JAX {spec['jax']} on {spec['backend']}. Dots are measured; dashed "
        "lines are projected.",
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
            label="Feasible (fit_grid_datasets)",
        ),
        plt.Line2D(
            [],
            [],
            color=c["ols"],
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=c["surface"],
            label="OLS (fit_grid; no validity check)",
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

    out = out_dir / f"feasible_scaling_{theme}.svg"
    fig.savefig(out, facecolor=c["surface"])
    if "--png" in sys.argv:
        fig.savefig(out.with_suffix(".png"), facecolor=c["surface"], dpi=110)
    plt.close(fig)
    return out


for theme in THEMES:
    print(render(theme))

"""
Generate matplotlib PNG charts for the analysis report.

Reads the measured benchmark numbers (kept consistent with the tables in
report/analysis_report piyu.tex) and writes PNGs to ../report/figures/.

Run from the project root:
    python scripts/generate_charts.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Colour palette ───────────────────────────────────────────────────────────
COL = {
    "serial":  "#7F7F7F",
    "openmp":  "#1F77B4",
    "posix":   "#2CA02C",
    "mpi":     "#D62728",
    "cuda":    "#9467BD",
    "hybrid":  "#FF7F0E",
}

WORKERS = [1, 2, 4, 8]

# ─── Measured timings (seconds) ───────────────────────────────────────────────
TIME_BLUR = {
    "Serial": [79.78] * 4,
    "OpenMP": [78.68, 39.57, 19.96, 19.72],
    "POSIX":  [78.44, 39.80, 20.65, 20.53],
    "MPI":    [80.81, 40.39, 20.86, 20.16],
}
CUDA_BLUR    = 0.0513
HYBRID_BLUR  = {"1x4": 4.37, "2x2": 4.39, "4x1": 4.37,
                "1x8": 4.37, "2x4": 4.37, "4x2": 4.38}

TIME_EDGE = {
    "Serial": [2.169] * 4,
    "OpenMP": [2.21, 1.23, 0.734, 0.735],
    "POSIX":  [2.01, 1.06, 0.547, 0.562],
    "MPI":    [2.09, 1.05, 0.528, 0.543],
}
CUDA_EDGE = 0.0120

TIME_SHARPEN = {
    "Serial": [0.258] * 4,
    "OpenMP": [0.288, 0.158, 0.098, 0.098],
    "POSIX":  [0.258, 0.134, 0.085, 0.073],
    "MPI":    [0.258, 0.130, 0.066, 0.075],
}
CUDA_SHARPEN = 0.0045

RMSE = {
    "OpenMP":     {"blur": 0.0000, "edge": 0.0000, "sharpen": 0.0000},
    "POSIX":      {"blur": 0.0000, "edge": 0.0000, "sharpen": 0.0000},
    "MPI":        {"blur": 0.2447, "edge": 0.3815, "sharpen": 1.4036},
    "CUDA":       {"blur": 0.0016, "edge": 0.0000, "sharpen": 0.0000},
    "Hybrid 2x2": {"blur": 0.1153, "edge": 0.2019, "sharpen": 0.7896},
}


def _style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "legend.framealpha": 0.95,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def _save(fig, name):
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# ─── Grouped-bar execution time chart ────────────────────────────────────────
def chart_time_bars(filter_name: str, times: dict, cuda_t: float,
                    kernel_str: str, image_str: str, fname: str,
                    hybrid_best: float | None = None):
    impls    = ["OpenMP", "POSIX", "MPI"]
    impl_col = [COL["openmp"], COL["posix"], COL["mpi"]]
    serial   = times["Serial"][0]

    x = np.arange(len(WORKERS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    # Reference lines (each on its own y-value, no overlap with bars)
    ax.axhline(serial, color=COL["serial"], linestyle="--", linewidth=1.4,
               label=f"Serial baseline ({serial:.3f} s)")
    ax.axhline(cuda_t, color=COL["cuda"], linestyle=":", linewidth=1.6,
               label=f"CUDA ({cuda_t:.4f} s)")
    if hybrid_best is not None:
        ax.axhline(hybrid_best, color=COL["hybrid"], linestyle="-.",
                   linewidth=1.6, label=f"Hybrid best ({hybrid_best:.3f} s)")

    # Grouped bars: one cluster per worker count, one bar per implementation
    for i, (impl, c) in enumerate(zip(impls, impl_col)):
        vals = times[impl]
        offsets = x + (i - 1) * w
        bars = ax.bar(offsets, vals, w, color=c, label=impl,
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            txt = f"{v:.3f}" if v < 1 else f"{v:.2f}"
            ax.annotate(txt,
                        xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} worker{'s' if n > 1 else ''}" for n in WORKERS])
    ax.set_xlabel("Number of CPU workers (threads / processes)")
    ax.set_ylabel("Execution time (seconds, log scale)")
    ax.set_title(f"{filter_name} ({kernel_str}) on {image_str} — execution time")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)

    top = max(max(times[i]) for i in impls) * 3.5
    bottom = min(cuda_t, hybrid_best or cuda_t) * 0.4
    ax.set_ylim(bottom, top)

    _save(fig, fname)


# ─── Grouped-bar speedup chart ───────────────────────────────────────────────
def chart_speedup_bars(filter_name: str, times: dict, cuda_t: float,
                       fname: str, hybrid_best: float | None = None):
    impls    = ["OpenMP", "POSIX", "MPI"]
    impl_col = [COL["openmp"], COL["posix"], COL["mpi"]]
    serial   = times["Serial"][0]

    sp = {impl: [serial / t for t in times[impl]] for impl in impls}

    x = np.arange(len(WORKERS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    # Ideal-linear reference: faint wide bar behind each group
    ax.bar(x, WORKERS, width=0.86, color="#E8E8E8", edgecolor="#BBBBBB",
           label="Ideal linear ($S = p$)", zorder=0)

    for i, (impl, c) in enumerate(zip(impls, impl_col)):
        vals = sp[impl]
        offsets = x + (i - 1) * w
        bars = ax.bar(offsets, vals, w, color=c, label=impl,
                      edgecolor="black", linewidth=0.5, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}×",
                        xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5)

    cuda_sp = serial / cuda_t
    extra = f"CUDA ≈ {cuda_sp:.0f}× (off-chart)"
    if hybrid_best is not None:
        extra += f"   ·   Hybrid best ≈ {serial / hybrid_best:.1f}×"

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} worker{'s' if n > 1 else ''}" for n in WORKERS])
    ax.set_xlabel("Number of parallel workers $p$")
    ax.set_ylabel(r"Speedup $S(p) = T_1 / T_p$")
    ax.set_title(f"Speedup — {filter_name}\n{extra}", fontsize=12)
    ax.set_ylim(0, max(8, max(max(sp[i]) for i in impls)) * 1.15)
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    _save(fig, fname)


# ─── RMSE bar chart ──────────────────────────────────────────────────────────
def chart_rmse():
    impls   = list(RMSE.keys())
    filters = ["blur", "edge", "sharpen"]
    colours = [COL["openmp"], COL["posix"], COL["mpi"]]
    labels  = ["Gaussian Blur (21×21)", "Edge Detection (3×3)", "Sharpen (3×3)"]

    x = np.arange(len(impls))
    w = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.4))
    for i, (f, c, lab) in enumerate(zip(filters, colours, labels)):
        vals = [RMSE[impl][f] for impl in impls]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=c, label=lab,
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.4f}" if v > 0 else "0",
                        xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(impls)
    ax.set_ylabel("RMSE (pixel units)")
    ax.set_xlabel("Implementation")
    ax.set_title("RMSE vs. serial baseline (lower is better; 0 = bit-exact)")
    ax.set_ylim(0, max(1.6, max(RMSE[i]["sharpen"] for i in impls) * 1.15))
    ax.legend(fontsize=9, loc="upper left")

    _save(fig, "rmse_bar.png")


# ─── Hybrid configuration bar chart ──────────────────────────────────────────
def chart_hybrid():
    labels = ["Pure OMP-4", "Hyb 1×4", "Hyb 2×2", "Hyb 4×1",
              "Hyb 1×8", "Hyb 2×4", "Hyb 4×2", "Pure MPI-8"]
    vals = [20.61, HYBRID_BLUR["1x4"], HYBRID_BLUR["2x2"], HYBRID_BLUR["4x1"],
            HYBRID_BLUR["1x8"], HYBRID_BLUR["2x4"], HYBRID_BLUR["4x2"], 20.50]
    colours = [COL["openmp"]] + [COL["hybrid"]] * 6 + [COL["mpi"]]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    bars = ax.bar(labels, vals, color=colours, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f} s",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9.5)

    ax.set_ylabel("Execution time (seconds)")
    ax.set_xlabel("Configuration")
    ax.set_title("Gaussian Blur — pure vs. hybrid MPI+OpenMP configurations")
    ax.set_ylim(0, max(vals) * 1.18)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")

    _save(fig, "hybrid_bar.png")


# ─── CUDA vs serial bar chart ────────────────────────────────────────────────
def chart_cuda():
    filters  = ["Gaussian Blur\n(21×21, 4K)", "Edge Detection\n(3×3, 4K)",
                "Sharpen\n(3×3, 1.2K)"]
    serial_t = [81.43, 2.123, 0.263]
    cuda_t   = [CUDA_BLUR, CUDA_EDGE, CUDA_SHARPEN]
    speedup  = [s / c for s, c in zip(serial_t, cuda_t)]

    x = np.arange(len(filters))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.4))
    b1 = ax.bar(x - w / 2, serial_t, w, color=COL["serial"], label="Serial (CPU)",
                edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + w / 2, cuda_t, w, color=COL["cuda"], label="CUDA (Tesla T4)",
                edgecolor="black", linewidth=0.4)
    for bars, vals in [(b1, serial_t), (b2, cuda_t)]:
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.4f} s" if v < 1 else f"{v:.2f} s",
                        xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5)
    for xi, sp in zip(x, speedup):
        ax.annotate(f"{sp:.0f}× speedup",
                    xy=(xi, max(serial_t[int(xi)], 0.01) * 1.8),
                    ha="center", fontsize=10, color=COL["cuda"],
                    fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(filters)
    ax.set_ylabel("Execution time (seconds, log scale)")
    ax.set_title("CUDA vs. Serial across all three filters")
    ax.legend(loc="upper right", fontsize=9)

    _save(fig, "cuda_vs_serial.png")


def main():
    _style()
    print(f"Writing charts to {OUT}")

    chart_time_bars("Gaussian Blur", TIME_BLUR, CUDA_BLUR,
                    "21×21", "3840×2160", "blur_time.png",
                    hybrid_best=HYBRID_BLUR["4x2"])
    chart_speedup_bars("Gaussian Blur", TIME_BLUR, CUDA_BLUR,
                       "blur_speedup.png", hybrid_best=HYBRID_BLUR["4x2"])

    chart_time_bars("Edge Detection", TIME_EDGE, CUDA_EDGE,
                    "3×3", "3840×2160", "edge_time.png")
    chart_speedup_bars("Edge Detection", TIME_EDGE, CUDA_EDGE,
                       "edge_speedup.png")

    chart_time_bars("Sharpen", TIME_SHARPEN, CUDA_SHARPEN,
                    "3×3", "1252×896", "sharpen_time.png")
    chart_speedup_bars("Sharpen", TIME_SHARPEN, CUDA_SHARPEN,
                       "sharpen_speedup.png")

    chart_rmse()
    chart_hybrid()
    chart_cuda()

    print("Done.")


if __name__ == "__main__":
    main()

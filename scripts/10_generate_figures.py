"""
SIMETRIC figure generation.

Generates Figures 3–6 for the manuscript:
  - Figure 3: Analysis flow diagram (participant/attempt exclusion)
  - Figure 4: Bayesian posterior forest plot (fixation count IRR)
  - Figure 5: Temporal tercile fixation distributions by device
  - Figure 6: MST Seldinger phase fixation-rate profiles

Colour convention:
  CON = #FF9800 (orange)
  ATG = #2196F3 (blue)
  MST = #4CAF50 (green)

Required files:
  data/processed/model_dataset_attempt_level.csv
  data/processed/phase_analysis_temporal_thirds.csv
  data/processed/phase_analysis_mst_phases.csv
  results/bayesian/count_device_contrasts.csv  (from 08_bayesian_sensitivity.py)

Run:
  pip install pandas numpy matplotlib seaborn
  python scripts/10_generate_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────
DEVICE_COLORS = {
    "CON": "#FF9800",
    "ATG": "#2196F3",
    "MST": "#4CAF50",
}
DEVICE_ORDER = ["CON", "ATG", "MST"]

DPI = 300


def figure3_flow_diagram():
    """Figure 3: Analysis flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    box_style = dict(
        boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5
    )
    excl_style = dict(
        boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#E65100", linewidth=1
    )
    result_style = dict(
        boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.5
    )

    # Boxes (y from top to bottom)
    boxes = [
        (5, 13, "30 clinicians recruited\n438 total attempts\n(3 devices × 5 rounds × ~30)", box_style),
        (5, 11, "3 clinicians discontinued\n(P7: 5, P8: 10, P14: 9 attempts)\n−24 attempts", excl_style),
        (5, 9.5, "27 retained clinicians\n414 primary attempts", box_style),
        (5, 8, "11 mapping failures\n(zero AOI-mapped fixations)", excl_style),
        (5, 6.5, "403 gaze-valid attempts\n(whole-attempt analysis)", result_style),
        (2.5, 4.5, "Fixation count\nNB-GEE + Bayesian NB", result_style),
        (5, 4.5, "Fixation duration\nLMM + Bayesian log-normal", result_style),
        (7.5, 4.5, "Saccade analysis\nNB-GEE + Gaussian GEE", result_style),
        (5, 2.5, "MST phase analysis\n124 single-puncture attempts\n590 phase-level observations", result_style),
        (5, 1, "Temporal tercile analysis\n1,209 tercile-level observations", result_style),
    ]

    for x, y, text, style in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                bbox=style, fontfamily="sans-serif")

    # Arrows
    arrows = [
        (5, 12.3, 5, 11.7),
        (5, 10.3, 5, 10.2),
        (5, 8.8, 5, 8.7),
        (5, 7.3, 5, 7.2),
        (5, 5.8, 5, 5.5),   # to analysis row
        (5, 5.8, 2.5, 5.2),
        (5, 5.8, 7.5, 5.2),
        (5, 3.8, 5, 3.2),
        (5, 1.8, 5, 1.7),
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

    fig.suptitle("Figure 3: Analysis Flow Diagram", fontsize=12, fontweight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_flow_diagram.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  Saved: fig_flow_diagram.png")


def figure4_bayesian_forest():
    """Figure 4: Bayesian posterior forest plot for fixation count IRR."""
    # Try to load pre-computed Bayesian results
    bayesian_file = RESULTS_DIR / "bayesian" / "count_device_contrasts.csv"

    if bayesian_file.exists():
        bc = pd.read_csv(bayesian_file)
        contrasts = []
        for _, row in bc.iterrows():
            name = row["Contrast"].replace("Count: ", "")
            contrasts.append({
                "label": name,
                "median": row["Ratio_median"],
                "lo": row["HDI_2.5"],
                "hi": row["HDI_97.5"],
                "p_gt1": row["P_ratio_gt_1"],
            })
    else:
        # Fallback: manuscript values
        print("  Warning: Bayesian results not found; using manuscript values")
        contrasts = [
            {"label": "CON vs ATG", "median": 1.09, "lo": 0.93, "hi": 1.28,
             "p_gt1": 84.6},
            {"label": "MST vs ATG", "median": 0.80, "lo": 0.60, "hi": 1.06,
             "p_gt1": 6.1},
            {"label": "CON vs MST", "median": 1.37, "lo": 1.01, "hi": 1.85,
             "p_gt1": 97.8},
        ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    y_positions = list(range(len(contrasts)))
    for i, c in enumerate(contrasts):
        color = "#2196F3"
        ax.errorbar(
            c["median"], i,
            xerr=[[c["median"] - c["lo"]], [c["hi"] - c["median"]]],
            fmt="o", color=color, markersize=8, capsize=5, linewidth=2,
        )
        ax.text(
            c["hi"] + 0.05, i,
            f"P(IRR>1) = {c['p_gt1']:.1f}%",
            va="center", fontsize=9,
        )

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([c["label"] for c in contrasts], fontsize=10)
    ax.set_xlabel("Incidence Rate Ratio (IRR)", fontsize=11)
    ax.set_title("Figure 4: Bayesian Fixation-Count Contrasts\n(95% HDI)", fontsize=12)
    ax.invert_yaxis()
    sns.despine(ax=ax)

    fig.savefig(FIG_DIR / "fig_bayesian_forest.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  Saved: fig_bayesian_forest.png")


def figure5_temporal_tercile():
    """Figure 5: Temporal tercile fixation distributions by device."""
    data_file = DATA_DIR / "phase_analysis_temporal_thirds.csv"
    if not data_file.exists():
        print("  SKIP: phase_analysis_temporal_thirds.csv not found")
        return

    d = pd.read_csv(data_file)
    d = d[d["Fixation_count"].notna()].copy()
    d["Tercile"] = d["Tercile"].astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, device in zip(axes, DEVICE_ORDER):
        sub = d[d["Device_name"] == device]
        color = DEVICE_COLORS[device]

        for tercile in [1, 2, 3]:
            vals = sub[sub["Tercile"] == tercile]["Fixation_count"]
            alpha = 0.3 + 0.2 * tercile
            ax.boxplot(
                vals.dropna(),
                positions=[tercile],
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=alpha),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(markerfacecolor=color, marker="o", markersize=3, alpha=0.5),
            )

        ax.set_title(device, fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Temporal Tercile", fontsize=10)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["T1\n(early)", "T2\n(middle)", "T3\n(late)"])

    axes[0].set_ylabel("Fixation Count per Tercile", fontsize=10)

    fig.suptitle(
        "Figure 5: Fixation Count by Temporal Tercile and Device",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(FIG_DIR / "fig_temporal_tercile.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  Saved: fig_temporal_tercile.png")


def figure6_mst_phases():
    """Figure 6: MST Seldinger phase fixation-rate profiles."""
    data_file = DATA_DIR / "phase_analysis_mst_phases.csv"
    if not data_file.exists():
        print("  SKIP: phase_analysis_mst_phases.csv not found")
        return

    d = pd.read_csv(data_file)
    d = d[d["Fixation_rate"].notna()].copy()

    phase_order = [
        "Pre-needle",
        "Needle-to-Wire",
        "Wire-to-NeedleOut",
        "NeedleOut-to-CON",
        "CON-to-WireOut",
    ]
    phase_short = ["PN", "NtW", "WtNO", "NOtC", "CtWO"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Compute means and SEM per phase
    summary = (
        d.groupby("Phase")["Fixation_rate"]
        .agg(["mean", "std", "count"])
        .reindex(phase_order)
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])

    x = np.arange(len(phase_order))

    ax.bar(x, summary["mean"], yerr=summary["sem"], capsize=5,
           color=DEVICE_COLORS["MST"], alpha=0.7, edgecolor="white",
           error_kw=dict(linewidth=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(phase_short, fontsize=10)
    ax.set_ylabel("Fixation Rate (events/s)", fontsize=11)
    ax.set_xlabel("Seldinger Phase", fontsize=11)
    ax.set_title(
        "Figure 6: MST Phase-Level Fixation Rate\n"
        f"(N = {d['Attempt_ID'].nunique()} MST attempts, {len(d)} phase observations)",
        fontsize=12, fontweight="bold",
    )

    # Add full phase names as secondary x-labels
    for i, (short, full) in enumerate(zip(phase_short, phase_order)):
        ax.text(i, -0.08 * ax.get_ylim()[1], full, ha="center", va="top",
                fontsize=7, color="grey", rotation=15)

    sns.despine(ax=ax)
    fig.savefig(FIG_DIR / "fig_mst_phases.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  Saved: fig_mst_phases.png")


def main() -> None:
    print("Generating SIMETRIC figures...")
    print(f"Output directory: {FIG_DIR}")

    figure3_flow_diagram()
    figure4_bayesian_forest()
    figure5_temporal_tercile()
    figure6_mst_phases()

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()

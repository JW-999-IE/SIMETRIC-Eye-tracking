"""
SIMETRIC MST phase-level fixation-rate analysis.

Analyses fixation rate (count / phase duration) across the five Seldinger
phases of MST insertion attempts: Pre-needle, Needle-to-Wire, Wire-to-NeedleOut,
NeedleOut-to-CON, CON-to-WireOut.

Primary model: Poisson GEE with clinician clustering, log(phase duration) offset,
Pre-needle as reference. Ten pairwise contrasts with Holm correction.

Sensitivity: NB-GEE with identical specification.

Required file:
  data/processed/phase_analysis_mst_phases.csv

Run:
  pip install pandas numpy scipy statsmodels
  python scripts/07_mst_phase_analysis.py
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    SCRIPT_DIR.parent / "data" / "processed" / "phase_analysis_mst_phases.csv",
    SCRIPT_DIR / "phase_analysis_mst_phases.csv",
]
OUT_DIR = SCRIPT_DIR.parent / "results" / "mst_phase"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE_ORDER = [
    "Pre-needle",
    "Needle-to-Wire",
    "Wire-to-NeedleOut",
    "NeedleOut-to-CON",
    "CON-to-WireOut",
]


def find_data() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find phase_analysis_mst_phases.csv. "
        "Looked in: " + ", ".join(str(p) for p in DATA_CANDIDATES)
    )


def prepare_data(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    required = {
        "Attempt_ID", "Participant", "Expertise", "Success",
        "Phase", "Phase_duration_ms", "Fixation_count",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    d = d.copy()

    # Drop rows with zero or missing phase duration
    d = d[(d["Phase_duration_ms"] > 0) & d["Fixation_count"].notna()].copy()

    d["log_phase_seconds"] = np.log(d["Phase_duration_ms"] / 1000.0)

    # Categorical coding with Pre-needle as reference
    d["Phase"] = pd.Categorical(
        d["Phase"], categories=PHASE_ORDER, ordered=False
    )

    # Experience mapping
    exp_map = {
        "Expert": "Higher prior experience",
        "Intermediate": "Some prior experience",
        "Novice": "No prior experience",
    }
    d["Experience3"] = pd.Categorical(
        d["Expertise"].map(exp_map),
        categories=[
            "Higher prior experience",
            "Some prior experience",
            "No prior experience",
        ],
        ordered=False,
    )
    return d


def _contrast_result(
    result, name: str, weights: dict[str, float], exponentiate: bool
) -> dict:
    names = list(result.params.index)
    w = np.array([weights.get(n, 0.0) for n in names], dtype=float)
    beta = float(np.dot(w, result.params.to_numpy()))
    cov = np.asarray(result.cov_params())
    se = float(np.sqrt(np.dot(w, np.dot(cov, w))))
    z = beta / se if se > 0 else np.nan
    p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    lo = beta - 1.96 * se
    hi = beta + 1.96 * se
    if exponentiate:
        return {
            "Contrast": name, "IRR": math.exp(beta),
            "CI_low": math.exp(lo), "CI_high": math.exp(hi),
            "z": z, "p_unadjusted": p,
        }
    return {
        "Contrast": name, "Estimate": beta,
        "CI_low": lo, "CI_high": hi,
        "z": z, "p_unadjusted": p,
    }


def _pairwise_contrasts(result, phases: list[str]) -> pd.DataFrame:
    """All 10 pairwise phase contrasts with Holm correction."""
    rows = []
    for a, b in combinations(range(len(phases)), 2):
        phase_a, phase_b = phases[a], phases[b]
        weights = {}

        # Reference is Pre-needle (index 0).
        # For non-reference phases, the parameter is
        # C(Phase, Treatment(reference='Pre-needle'))[T.<phase>]
        if phase_a != "Pre-needle":
            param_a = f"C(Phase, Treatment(reference='Pre-needle'))[T.{phase_a}]"
            weights[param_a] = -1  # denominator
        if phase_b != "Pre-needle":
            param_b = f"C(Phase, Treatment(reference='Pre-needle'))[T.{phase_b}]"
            weights[param_b] = 1  # numerator

        label = f"{phase_b} vs {phase_a}"
        rows.append(_contrast_result(result, label, weights, True))

    out = pd.DataFrame(rows)
    valid = out["p_unadjusted"].notna()
    out["p_holm"] = np.nan
    if valid.any():
        out.loc[valid, "p_holm"] = multipletests(
            out.loc[valid, "p_unadjusted"].astype(float), method="holm"
        )[1]
    return out


def fit_poisson_gee(d: pd.DataFrame) -> None:
    """Primary model: Poisson GEE clustered by clinician."""
    print("  Fitting Poisson GEE...")

    formula = (
        "Fixation_count ~ C(Phase, Treatment(reference='Pre-needle')) "
        "+ C(Experience3) + C(Success)"
    )

    gee = smf.gee(
        formula,
        groups="Participant",
        data=d,
        family=sm.families.Poisson(),
        cov_struct=Exchangeable(),
        offset=d["log_phase_seconds"],
    )
    result = gee.fit(cov_type="bias_reduced", maxiter=200)

    (OUT_DIR / "poisson_gee_summary.txt").write_text(
        result.summary().as_text(), encoding="utf-8"
    )

    # Omnibus phase effect
    names = list(result.params.index)
    phase_idx = [i for i, n in enumerate(names) if "C(Phase" in n]
    if phase_idx:
        R = np.zeros((len(phase_idx), len(names)))
        for r, i in enumerate(phase_idx):
            R[r, i] = 1.0
        test = result.wald_test(R, scalar=True)
        print(f"  Phase omnibus: χ²({len(phase_idx)}) = {float(test.statistic):.2f}, "
              f"p = {float(test.pvalue):.4f}")
        wald_df = pd.DataFrame([{
            "Effect": "Phase (omnibus)",
            "chi2": float(test.statistic),
            "df": len(phase_idx),
            "p": float(test.pvalue),
        }])
        wald_df.to_csv(OUT_DIR / "poisson_omnibus_phase_test.csv", index=False)

    # Phase contrasts vs Pre-needle
    phase_vs_ref = []
    for phase in PHASE_ORDER[1:]:
        param = f"C(Phase, Treatment(reference='Pre-needle'))[T.{phase}]"
        phase_vs_ref.append(
            _contrast_result(result, f"{phase} vs Pre-needle", {param: 1}, True)
        )
    pd.DataFrame(phase_vs_ref).to_csv(
        OUT_DIR / "poisson_phase_vs_reference.csv", index=False
    )

    # All 10 pairwise contrasts with Holm
    pw = _pairwise_contrasts(result, PHASE_ORDER)
    pw.to_csv(OUT_DIR / "poisson_pairwise_contrasts_holm.csv", index=False)

    print("  Pairwise contrasts (Holm-corrected):")
    for _, row in pw.iterrows():
        sig = "*" if row["p_holm"] < 0.05 else ""
        print(f"    {row['Contrast']}: IRR = {row['IRR']:.2f} "
              f"[{row['CI_low']:.2f}, {row['CI_high']:.2f}], "
              f"p_holm = {row['p_holm']:.3f}{sig}")


def fit_nb_gee_sensitivity(d: pd.DataFrame) -> None:
    """Sensitivity model: NB-GEE clustered by clinician."""
    print("\n  Fitting NB-GEE sensitivity...")

    formula = (
        "Fixation_count ~ C(Phase, Treatment(reference='Pre-needle')) "
        "+ C(Experience3) + C(Success)"
    )

    # Two-step alpha estimation
    try:
        nb = smf.negativebinomial(
            formula, data=d, offset=d["log_phase_seconds"]
        ).fit(disp=False, maxiter=1000)
        alpha = float(nb.params.get("alpha", 1.0))
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 1.0
    except Exception:
        alpha = 1.0

    print(f"  Estimated NB alpha: {alpha:.6f}")

    gee = smf.gee(
        formula,
        groups="Participant",
        data=d,
        family=sm.families.NegativeBinomial(alpha=alpha),
        cov_struct=Exchangeable(),
        offset=d["log_phase_seconds"],
    )
    result = gee.fit(cov_type="bias_reduced", maxiter=200)

    (OUT_DIR / "nb_gee_sensitivity_summary.txt").write_text(
        result.summary().as_text() + f"\n\nNB alpha fixed at {alpha:.6f}\n",
        encoding="utf-8",
    )

    # All 10 pairwise contrasts with Holm
    pw = _pairwise_contrasts(result, PHASE_ORDER)
    pw.to_csv(OUT_DIR / "nb_gee_pairwise_contrasts_holm.csv", index=False)

    print("  NB-GEE pairwise contrasts (Holm-corrected):")
    for _, row in pw.iterrows():
        sig = "*" if row["p_holm"] < 0.05 else ""
        print(f"    {row['Contrast']}: IRR = {row['IRR']:.2f} "
              f"[{row['CI_low']:.2f}, {row['CI_high']:.2f}], "
              f"p_holm = {row['p_holm']:.3f}{sig}")


def descriptive_table(d: pd.DataFrame) -> None:
    """Phase-level descriptive statistics."""
    desc = (
        d.groupby("Phase", observed=True)
        .agg(
            n_obs=("Fixation_count", "size"),
            n_attempts=("Attempt_ID", "nunique"),
            mean_fixation_count=("Fixation_count", "mean"),
            sd_fixation_count=("Fixation_count", "std"),
            median_fixation_count=("Fixation_count", "median"),
            mean_rate=("Fixation_rate", "mean"),
            sd_rate=("Fixation_rate", "std"),
            mean_phase_duration_ms=("Phase_duration_ms", "mean"),
        )
        .reindex(PHASE_ORDER)
    )
    desc.to_csv(OUT_DIR / "descriptive_statistics.csv")
    print("\n  Descriptive statistics:")
    print(desc.to_string())


def main() -> None:
    path = find_data()
    print(f"Data: {path}")
    d = prepare_data(path)
    print(f"Phase-level observations: {len(d)}")
    print(f"Unique MST attempts: {d['Attempt_ID'].nunique()}")
    print(f"Participants: {d['Participant'].nunique()}")

    d.to_csv(OUT_DIR / "analysis_dataset_used.csv", index=False)

    descriptive_table(d)

    print("\n── Primary: Poisson GEE ──")
    fit_poisson_gee(d)

    print("\n── Sensitivity: NB-GEE ──")
    fit_nb_gee_sensitivity(d)

    readme = (
        "MST phase analysis results\n\n"
        f"Phase-level observations: {len(d)}\n"
        f"Unique MST attempts: {d['Attempt_ID'].nunique()}\n"
        f"Participants: {d['Participant'].nunique()}\n"
        f"Phases: {', '.join(PHASE_ORDER)}\n\n"
        "Models:\n"
        "- Poisson GEE (primary): clinician clustering, exchangeable correlation,\n"
        "  bias-reduced SE, log(phase duration) offset, Pre-needle reference\n"
        "- NB-GEE (sensitivity): same specification with NB dispersion\n\n"
        "Contrasts: 10 pairwise phase comparisons, Holm-corrected.\n"
    )
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")
    print(f"\nResults written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

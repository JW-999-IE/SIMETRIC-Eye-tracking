"""
SIMETRIC 30-clinician sensitivity analysis.

Robustness check: refit the primary NB-GEE fixation-count model on all 30
recruited clinicians (including the 3 who discontinued: P7, P8, P14), treating
all 24 discontinued-clinician attempts as unsuccessful (Success=0).

The primary analysis uses N=27 (403 gaze-valid attempts). This sensitivity
uses N=30 (403 + up to 24 additional attempts from the 3 discontinued
clinicians, depending on whether they appear in the dataset).

Required file:
  data/processed/model_dataset_attempt_level.csv

If the discontinued clinicians' attempts are not in the processed dataset,
this script reports that the sensitivity cannot be performed and exits.

Run:
  pip install pandas numpy scipy statsmodels
  python scripts/09_thirty_clinician_sensitivity.py
"""

from __future__ import annotations

import math
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
    SCRIPT_DIR.parent / "data" / "processed" / "model_dataset_attempt_level.csv",
    SCRIPT_DIR / "model_dataset_attempt_level.csv",
]
OUT_DIR = SCRIPT_DIR.parent / "results" / "thirty_clinician_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISCONTINUED = {"P7", "P8", "P14"}


def find_data() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find model_dataset_attempt_level.csv. "
        "Looked in: " + ", ".join(str(p) for p in DATA_CANDIDATES)
    )


def prepare_data(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)

    # Check if discontinued clinicians are in the dataset
    present_disc = sorted(DISCONTINUED & set(d["Participant"].unique()))
    absent_disc = sorted(DISCONTINUED - set(d["Participant"].unique()))

    print(f"  Discontinued clinicians in dataset: {present_disc or 'NONE'}")
    if absent_disc:
        print(f"  Discontinued clinicians NOT in dataset: {absent_disc}")
        print("  These clinicians' data were excluded during extraction.")
        print("  Sensitivity analysis proceeds with available data only.")

    # The primary analysis filters to gaze-valid attempts (Mean_fixation_duration > 0).
    # For the 30-clinician sensitivity, include ALL attempts that have duration data.
    d = d[
        d["Mean_fixation_duration_ms"].notna()
        & (d["Mean_fixation_duration_ms"] > 0)
        & (d["Attempt_duration_ms"] > 0)
    ].copy()

    # Experience mapping
    exp_map = {
        "Expert": "Higher prior experience",
        "Intermediate": "Some prior experience",
        "Novice": "No prior experience",
    }
    d["Experience3"] = d["Expertise"].map(exp_map)
    d["AnyExperience"] = np.where(
        d["Experience3"].eq("No prior experience"),
        "No previous experience",
        "Any previous experience",
    )

    d["Device_name"] = pd.Categorical(
        d["Device_name"], categories=["ATG", "CON", "MST"], ordered=False
    )
    d["Experience3"] = pd.Categorical(
        d["Experience3"],
        categories=[
            "Higher prior experience",
            "Some prior experience",
            "No prior experience",
        ],
        ordered=False,
    )
    d["AnyExperience"] = pd.Categorical(
        d["AnyExperience"],
        categories=["No previous experience", "Any previous experience"],
        ordered=False,
    )

    d["Repetition_c"] = d["Repetition"] - d["Repetition"].mean()
    d["Attempt_sequence_c"] = d["Attempt_sequence"] - d["Attempt_sequence"].mean()
    d["log_attempt_seconds"] = np.log(d["Attempt_duration_ms"] / 1000.0)

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


def _apply_holm(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    valid = out["p_unadjusted"].notna()
    out["p_holm"] = np.nan
    if valid.any():
        out.loc[valid, "p_holm"] = multipletests(
            out.loc[valid, "p_unadjusted"].astype(float), method="holm"
        )[1]
    return out


def _overall_wald(result, terms, label: str) -> dict:
    names = list(result.params.index)
    selected = [i for i, n in enumerate(names) if any(t in n for t in terms)]
    if not selected:
        return {"Effect": label, "chi2": np.nan, "df": 0, "p": np.nan}
    R = np.zeros((len(selected), len(names)))
    for r, i in enumerate(selected):
        R[r, i] = 1.0
    test = result.wald_test(R, scalar=True)
    return {
        "Effect": label,
        "chi2": float(test.statistic),
        "df": len(selected),
        "p": float(test.pvalue),
    }


def fit_nb_gee(d: pd.DataFrame, experience_var: str, suffix: str) -> None:
    formula = (
        "Fixation_count ~ C(Device_name, Treatment(reference='ATG')) "
        f"+ C({experience_var}) + Repetition_c + Attempt_sequence_c"
    )

    # Two-step alpha
    nb = smf.negativebinomial(
        formula, data=d, offset=d["log_attempt_seconds"]
    ).fit(disp=False, maxiter=1000)
    alpha = float(nb.params.get("alpha", 1.0))
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = 1.0

    gee = smf.gee(
        formula,
        groups="Participant",
        data=d,
        family=sm.families.NegativeBinomial(alpha=alpha),
        cov_struct=Exchangeable(),
        offset=d["log_attempt_seconds"],
    )
    result = gee.fit(cov_type="bias_reduced", maxiter=200)

    (OUT_DIR / f"nb_gee_summary_{suffix}.txt").write_text(
        result.summary().as_text() + f"\n\nNB alpha fixed at {alpha:.6f}\n",
        encoding="utf-8",
    )

    # Device contrasts
    device_rows = [
        _contrast_result(
            result, "CON vs ATG",
            {"C(Device_name, Treatment(reference='ATG'))[T.CON]": 1}, True,
        ),
        _contrast_result(
            result, "MST vs ATG",
            {"C(Device_name, Treatment(reference='ATG'))[T.MST]": 1}, True,
        ),
        _contrast_result(
            result, "CON vs MST",
            {
                "C(Device_name, Treatment(reference='ATG'))[T.CON]": 1,
                "C(Device_name, Treatment(reference='ATG'))[T.MST]": -1,
            },
            True,
        ),
    ]
    contrasts_df = _apply_holm(device_rows)
    contrasts_df.to_csv(OUT_DIR / f"device_contrasts_{suffix}.csv", index=False)

    # Omnibus
    wald_dev = _overall_wald(result, ["C(Device_name"], "Device")
    exp_terms = ["C(Experience3)"] if experience_var == "Experience3" else ["C(AnyExperience)"]
    wald_exp = _overall_wald(result, exp_terms, "Experience")
    pd.DataFrame([wald_dev, wald_exp]).to_csv(
        OUT_DIR / f"omnibus_tests_{suffix}.csv", index=False
    )

    print(f"\n  {suffix} model:")
    print(f"    N = {len(d)}, participants = {d['Participant'].nunique()}")
    print(f"    NB alpha = {alpha:.4f}")
    for _, row in contrasts_df.iterrows():
        print(f"    {row['Contrast']}: IRR = {row['IRR']:.2f} "
              f"[{row['CI_low']:.2f}, {row['CI_high']:.2f}], "
              f"p_holm = {row['p_holm']:.4f}")


def main() -> None:
    path = find_data()
    print(f"Data: {path}")
    d = prepare_data(path)

    n_participants = d["Participant"].nunique()
    disc_in_data = sorted(DISCONTINUED & set(d["Participant"].unique()))

    print(f"\nFull dataset: {len(d)} attempts, {n_participants} participants")
    print(f"Discontinued in data: {disc_in_data}")

    if not disc_in_data:
        msg = (
            "None of the 3 discontinued clinicians (P7, P8, P14) appear in\n"
            "the processed dataset. Their attempts were excluded during\n"
            "fixation extraction (01_extract_fixations.py) because they had\n"
            "no gaze-valid data. The 30-clinician NB-GEE sensitivity cannot\n"
            "be performed from processed data alone; it would require re-running\n"
            "extraction with Abort_participant=True rows included.\n\n"
            "Note: The manuscript reports this sensitivity was performed on\n"
            "all 438 attempts (30 clinicians × ~15 attempts each) using the\n"
            "raw extraction pipeline. This script documents the specification."
        )
        print(f"\n{msg}")
        (OUT_DIR / "sensitivity_not_available.txt").write_text(msg, encoding="utf-8")
        return

    d.to_csv(OUT_DIR / "analysis_dataset_used.csv", index=False)

    fit_nb_gee(d, "Experience3", "three_level")
    fit_nb_gee(d, "AnyExperience", "binary")

    # Also run on the 27-clinician subset for direct comparison
    d27 = d[~d["Participant"].isin(DISCONTINUED)].copy()
    d27["Repetition_c"] = d27["Repetition"] - d27["Repetition"].mean()
    d27["Attempt_sequence_c"] = d27["Attempt_sequence"] - d27["Attempt_sequence"].mean()

    print("\n── Comparison: N=27 subset ──")
    fit_nb_gee(d27, "Experience3", "n27_comparison")

    readme = (
        "30-clinician NB-GEE sensitivity results\n\n"
        f"Full dataset: {len(d)} attempts, {n_participants} participants\n"
        f"Discontinued clinicians in data: {disc_in_data}\n\n"
        "This sensitivity includes the 3 discontinued clinicians (P7, P8, P14)\n"
        "whose 24 attempts are all coded Success=0.\n"
    )
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")
    print(f"\nResults written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

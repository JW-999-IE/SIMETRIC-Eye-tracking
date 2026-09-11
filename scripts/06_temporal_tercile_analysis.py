"""
SIMETRIC temporal-tercile analysis of fixation count across attempt thirds.

Each attempt is divided into three equal-duration temporal terciles (T1, T2, T3).
This script fits NB-GEE models with a Device × Tercile interaction to test
whether the decline in fixation rate across attempt phases differs by device.

Primary model: NB-GEE with Device × Tercile interaction, participant clustering.
Reference levels: ATG (device), Tercile 1 (tercile).

Key manuscript values to reproduce:
  - T2/T1 IRR = 0.40 [0.30, 0.51]
  - Device × Tercile interaction χ² = 10.40, p = .006

Required file:
  data/processed/phase_analysis_temporal_thirds.csv

Run:
  pip install pandas numpy scipy statsmodels
  python scripts/06_temporal_tercile_analysis.py
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
    SCRIPT_DIR.parent / "data" / "processed" / "phase_analysis_temporal_thirds.csv",
    SCRIPT_DIR / "phase_analysis_temporal_thirds.csv",
]
OUT_DIR = SCRIPT_DIR.parent / "results" / "temporal_tercile"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_data() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find phase_analysis_temporal_thirds.csv. "
        "Looked in: " + ", ".join(str(p) for p in DATA_CANDIDATES)
    )


def prepare_data(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    required = {
        "Attempt_ID", "Participant", "Device_name", "Expertise",
        "Success", "Tercile", "Phase_duration_ms", "Fixation_count",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    d = d.copy()
    # Phase_duration_ms is the WHOLE attempt duration repeated for each tercile.
    # Tercile duration = Phase_duration_ms / 3.
    d["Tercile_duration_ms"] = d["Phase_duration_ms"] / 3.0
    d["log_tercile_seconds"] = np.log(d["Tercile_duration_ms"] / 1000.0)

    # Drop rows with zero tercile duration or missing fixation count
    d = d[
        (d["Tercile_duration_ms"] > 0)
        & d["Fixation_count"].notna()
    ].copy()

    # Categorical coding
    d["Device_name"] = pd.Categorical(
        d["Device_name"], categories=["ATG", "CON", "MST"], ordered=False
    )
    d["Tercile"] = pd.Categorical(
        d["Tercile"].astype(int), categories=[1, 2, 3], ordered=False
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


def estimate_nb_alpha(d: pd.DataFrame, formula: str) -> float:
    nb = smf.negativebinomial(
        formula, data=d, offset=d["log_tercile_seconds"]
    ).fit(disp=False, maxiter=1000)
    alpha = float(nb.params.get("alpha", 1.0))
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = 1.0
    return alpha


def fit_interaction_model(d: pd.DataFrame) -> None:
    """NB-GEE with Device × Tercile interaction."""
    formula = (
        "Fixation_count ~ "
        "C(Device_name, Treatment(reference='ATG')) * C(Tercile, Treatment(reference=1)) "
        "+ C(Experience3)"
    )

    alpha = estimate_nb_alpha(d, formula)
    print(f"  Estimated NB alpha: {alpha:.6f}")

    gee = smf.gee(
        formula,
        groups="Participant",
        data=d,
        family=sm.families.NegativeBinomial(alpha=alpha),
        cov_struct=Exchangeable(),
        offset=d["log_tercile_seconds"],
    )
    result = gee.fit(cov_type="bias_reduced", maxiter=200)

    # Save full summary
    summary_text = result.summary().as_text()
    summary_text += f"\n\nNB alpha fixed at {alpha:.6f}\n"
    (OUT_DIR / "interaction_model_summary.txt").write_text(
        summary_text, encoding="utf-8"
    )

    # ── Omnibus Wald tests ─────────────────────────────────────────
    wald_device = _overall_wald(result, ["C(Device_name"], "Device (main)")
    wald_tercile = _overall_wald(result, ["C(Tercile"], "Tercile (main + interaction)")
    wald_interaction = _overall_wald(
        result,
        ["C(Device_name, Treatment(reference='ATG'))["
         ":C(Tercile, Treatment(reference=1))["],
        "Device × Tercile interaction",
    )
    # Interaction-only terms
    names = list(result.params.index)
    ix_idx = [i for i, n in enumerate(names) if ":" in n]
    if ix_idx:
        R = np.zeros((len(ix_idx), len(names)))
        for r, i in enumerate(ix_idx):
            R[r, i] = 1.0
        test = result.wald_test(R, scalar=True)
        wald_interaction = {
            "Effect": "Device × Tercile interaction",
            "chi2": float(test.statistic),
            "df": len(ix_idx),
            "p": float(test.pvalue),
        }

    wald_df = pd.DataFrame([wald_device, wald_tercile, wald_interaction])
    wald_df.to_csv(OUT_DIR / "omnibus_wald_tests.csv", index=False)
    print(f"\n  Omnibus tests:")
    for _, row in wald_df.iterrows():
        print(f"    {row['Effect']}: χ²({row['df']:.0f}) = {row['chi2']:.2f}, p = {row['p']:.4f}")

    # ── Tercile contrasts (marginal over devices) ──────────────────
    # T2 vs T1: average of T2 main + interaction terms weighted by device proportions
    # For the marginal tercile effect, use the T2 main effect directly (at reference device ATG)
    t2_param = "C(Tercile, Treatment(reference=1))[T.2]"
    t3_param = "C(Tercile, Treatment(reference=1))[T.3]"

    tercile_rows = [
        _contrast_result(result, "T2 vs T1 (at ATG)", {t2_param: 1}, True),
        _contrast_result(result, "T3 vs T1 (at ATG)", {t3_param: 1}, True),
        _contrast_result(result, "T3 vs T2 (at ATG)", {t3_param: 1, t2_param: -1}, True),
    ]
    pd.DataFrame(tercile_rows).to_csv(
        OUT_DIR / "tercile_contrasts_at_reference.csv", index=False
    )

    # ── Device-specific tercile contrasts ──────────────────────────
    device_tercile_rows = []
    for dev in ["ATG", "CON", "MST"]:
        if dev == "ATG":
            w_t2 = {t2_param: 1}
            w_t3 = {t3_param: 1}
        else:
            ix_t2 = f"C(Device_name, Treatment(reference='ATG'))[T.{dev}]:C(Tercile, Treatment(reference=1))[T.2]"
            ix_t3 = f"C(Device_name, Treatment(reference='ATG'))[T.{dev}]:C(Tercile, Treatment(reference=1))[T.3]"
            w_t2 = {t2_param: 1, ix_t2: 1}
            w_t3 = {t3_param: 1, ix_t3: 1}
        device_tercile_rows.append(
            _contrast_result(result, f"{dev}: T2 vs T1", w_t2, True)
        )
        device_tercile_rows.append(
            _contrast_result(result, f"{dev}: T3 vs T1", w_t3, True)
        )
        w_t3vt2 = {k: v for k, v in w_t3.items()}
        for k, v in w_t2.items():
            w_t3vt2[k] = w_t3vt2.get(k, 0) - v
        device_tercile_rows.append(
            _contrast_result(result, f"{dev}: T3 vs T2", w_t3vt2, True)
        )

    pd.DataFrame(device_tercile_rows).to_csv(
        OUT_DIR / "device_specific_tercile_contrasts.csv", index=False
    )

    # ── Device contrasts at each tercile ───────────────────────────
    device_at_tercile_rows = []
    for t in [1, 2, 3]:
        for pair, dev_a, dev_b in [
            ("CON vs ATG", "CON", "ATG"),
            ("MST vs ATG", "MST", "ATG"),
            ("CON vs MST", "CON", "MST"),
        ]:
            w = {}
            for dev in [dev_a, dev_b]:
                sign = 1 if dev == dev_a else -1
                if dev != "ATG":
                    dev_param = f"C(Device_name, Treatment(reference='ATG'))[T.{dev}]"
                    w[dev_param] = w.get(dev_param, 0) + sign
                    if t > 1:
                        ix_param = f"C(Device_name, Treatment(reference='ATG'))[T.{dev}]:C(Tercile, Treatment(reference=1))[T.{t}]"
                        w[ix_param] = w.get(ix_param, 0) + sign
            device_at_tercile_rows.append(
                _contrast_result(result, f"T{t}: {pair}", w, True)
            )

    pd.DataFrame(device_at_tercile_rows).to_csv(
        OUT_DIR / "device_contrasts_by_tercile.csv", index=False
    )

    # ── Print key manuscript values ────────────────────────────────
    print("\n  Key contrasts:")
    for row in tercile_rows:
        print(f"    {row['Contrast']}: IRR = {row['IRR']:.2f} "
              f"[{row['CI_low']:.2f}, {row['CI_high']:.2f}], p = {row['p_unadjusted']:.4f}")

    return result


def fit_main_effects_model(d: pd.DataFrame) -> None:
    """NB-GEE without interaction (main effects only) for comparison."""
    formula = (
        "Fixation_count ~ "
        "C(Device_name, Treatment(reference='ATG')) + C(Tercile, Treatment(reference=1)) "
        "+ C(Experience3)"
    )

    alpha = estimate_nb_alpha(d, formula)

    gee = smf.gee(
        formula,
        groups="Participant",
        data=d,
        family=sm.families.NegativeBinomial(alpha=alpha),
        cov_struct=Exchangeable(),
        offset=d["log_tercile_seconds"],
    )
    result = gee.fit(cov_type="bias_reduced", maxiter=200)

    (OUT_DIR / "main_effects_model_summary.txt").write_text(
        result.summary().as_text() + f"\n\nNB alpha fixed at {alpha:.6f}\n",
        encoding="utf-8",
    )

    # Marginal tercile contrasts (pooled across devices)
    t2_param = "C(Tercile, Treatment(reference=1))[T.2]"
    t3_param = "C(Tercile, Treatment(reference=1))[T.3]"

    rows = [
        _contrast_result(result, "T2 vs T1 (marginal)", {t2_param: 1}, True),
        _contrast_result(result, "T3 vs T1 (marginal)", {t3_param: 1}, True),
        _contrast_result(result, "T3 vs T2 (marginal)", {t3_param: 1, t2_param: -1}, True),
    ]
    pd.DataFrame(rows).to_csv(
        OUT_DIR / "marginal_tercile_contrasts.csv", index=False
    )

    print("\n  Main-effects model tercile contrasts:")
    for row in rows:
        print(f"    {row['Contrast']}: IRR = {row['IRR']:.2f} "
              f"[{row['CI_low']:.2f}, {row['CI_high']:.2f}], p = {row['p_unadjusted']:.4f}")


def main() -> None:
    path = find_data()
    print(f"Data: {path}")
    d = prepare_data(path)
    print(f"Observations: {len(d)} (tercile-level rows)")
    print(f"Unique attempts: {d['Attempt_ID'].nunique()}")
    print(f"Participants: {d['Participant'].nunique()}")
    print(f"Devices: {sorted(d['Device_name'].cat.categories)}")

    d.to_csv(OUT_DIR / "analysis_dataset_used.csv", index=False)

    print("\n── Interaction model ──")
    fit_interaction_model(d)

    print("\n── Main-effects model ──")
    fit_main_effects_model(d)

    readme = (
        "Temporal tercile analysis results\n\n"
        f"Tercile-level observations: {len(d)}\n"
        f"Unique attempts: {d['Attempt_ID'].nunique()}\n"
        f"Participants: {d['Participant'].nunique()}\n\n"
        "Models:\n"
        "- interaction_model: NB-GEE with Device × Tercile interaction\n"
        "- main_effects_model: NB-GEE without interaction (marginal tercile effects)\n\n"
        "Both: participant clustering, exchangeable correlation, bias-reduced SE,\n"
        "log(tercile duration) offset, NB alpha from two-step estimation.\n"
    )
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")
    print(f"\nResults written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

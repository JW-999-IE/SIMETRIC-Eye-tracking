from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests

DATA_FILE = Path(__file__).with_name("model_dataset_attempt_level.csv")
OUT_DIR = Path(__file__).with_name("python_confirmatory_fixation_results")
OUT_DIR.mkdir(exist_ok=True)


def prepare_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.name} next to this script. "
            "Copy model_dataset_attempt_level.csv into the same folder."
        )
    d = pd.read_csv(path)
    required = {
        "Participant", "Device_name", "Expertise", "Repetition",
        "Attempt_sequence", "Mean_fixation_duration_ms", "Fixation_count",
        "Attempt_duration_ms",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    d = d.copy()
    d = d[
        d["Mean_fixation_duration_ms"].notna()
        & (d["Mean_fixation_duration_ms"] > 0)
        & d["Fixation_count"].notna()
        & (d["Attempt_duration_ms"] > 0)
    ].copy()

    exp_map = {
        "Expert": "Higher prior experience",
        "Intermediate": "Some prior experience",
        "Novice": "No prior experience",
    }
    d["Experience3"] = d["Expertise"].map(exp_map)
    if d["Experience3"].isna().any():
        bad = sorted(d.loc[d["Experience3"].isna(), "Expertise"].astype(str).unique())
        raise ValueError(f"Unrecognised Expertise values: {bad}")

    d["AnyExperience"] = np.where(
        d["Experience3"].eq("No prior experience"),
        "No previous experience",
        "Any previous experience",
    )

    # Explicit reference levels.
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

    # Centre continuous predictors so the intercept is interpretable.
    d["Repetition_c"] = d["Repetition"] - d["Repetition"].mean()
    d["Attempt_sequence_c"] = d["Attempt_sequence"] - d["Attempt_sequence"].mean()
    d["log_mean_fixation_duration"] = np.log(d["Mean_fixation_duration_ms"])
    d["log_attempt_seconds"] = np.log(d["Attempt_duration_ms"] / 1000.0)
    return d


def _contrast_result(
    result,
    name: str,
    weights: dict[str, float],
    exponentiate: bool,
) -> dict[str, float | str]:
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
            "Contrast": name,
            "Estimate": math.exp(beta),
            "CI_low": math.exp(lo),
            "CI_high": math.exp(hi),
            "p_unadjusted": p,
        }
    return {
        "Contrast": name,
        "Estimate": beta,
        "CI_low": lo,
        "CI_high": hi,
        "p_unadjusted": p,
    }


def _apply_holm(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    valid = out["p_unadjusted"].notna()
    out["p_holm"] = np.nan
    if valid.any():
        out.loc[valid, "p_holm"] = multipletests(
            out.loc[valid, "p_unadjusted"].astype(float), method="holm"
        )[1]
    return out


def _overall_wald(result, terms: Iterable[str], label: str) -> pd.DataFrame:
    names = list(result.params.index)
    selected = [i for i, n in enumerate(names) if any(t in n for t in terms)]
    if not selected:
        return pd.DataFrame([{"Effect": label, "chi2": np.nan, "df": 0, "p": np.nan}])
    R = np.zeros((len(selected), len(names)))
    for r, i in enumerate(selected):
        R[r, i] = 1.0
    test = result.wald_test(R, scalar=True)
    return pd.DataFrame(
        [{
            "Effect": label,
            "chi2": float(test.statistic),
            "df": len(selected),
            "p": float(test.pvalue),
        }]
    )


def fit_duration_model(d: pd.DataFrame, experience_var: str, suffix: str) -> None:
    formula = (
        "log_mean_fixation_duration ~ C(Device_name, Treatment(reference='ATG')) "
        f"+ C({experience_var}) + Repetition_c + Attempt_sequence_c"
    )
    model = smf.mixedlm(formula, d, groups=d["Participant"], re_formula="1")
    result = model.fit(reml=False, method="lbfgs", maxiter=2000, disp=False)

    (OUT_DIR / f"duration_mixedlm_{suffix}.txt").write_text(
        result.summary().as_text(), encoding="utf-8"
    )

    device_rows = [
        _contrast_result(
            result,
            "CON vs ATG",
            {"C(Device_name, Treatment(reference='ATG'))[T.CON]": 1},
            True,
        ),
        _contrast_result(
            result,
            "MST vs ATG",
            {"C(Device_name, Treatment(reference='ATG'))[T.MST]": 1},
            True,
        ),
        _contrast_result(
            result,
            "CON vs MST",
            {
                "C(Device_name, Treatment(reference='ATG'))[T.CON]": 1,
                "C(Device_name, Treatment(reference='ATG'))[T.MST]": -1,
            },
            True,
        ),
    ]
    _apply_holm(device_rows).to_csv(
        OUT_DIR / f"duration_device_contrasts_{suffix}.csv", index=False
    )

    if experience_var == "Experience3":
        some = "C(Experience3)[T.Some prior experience]"
        no = "C(Experience3)[T.No prior experience]"
        exp_rows = [
            _contrast_result(result, "Some vs higher", {some: 1}, True),
            _contrast_result(result, "No vs higher", {no: 1}, True),
            _contrast_result(result, "No vs some", {no: 1, some: -1}, True),
        ]
        exp_terms = ["C(Experience3)"]
    else:
        any_term = "C(AnyExperience)[T.Any previous experience]"
        exp_rows = [
            _contrast_result(
                result, "Any vs no previous experience", {any_term: 1}, True
            )
        ]
        exp_terms = ["C(AnyExperience)"]

    _apply_holm(exp_rows).to_csv(
        OUT_DIR / f"duration_experience_contrasts_{suffix}.csv", index=False
    )
    pd.concat(
        [
            _overall_wald(result, ["C(Device_name"], "Device"),
            _overall_wald(result, exp_terms, "Experience"),
        ],
        ignore_index=True,
    ).to_csv(OUT_DIR / f"duration_overall_tests_{suffix}.csv", index=False)


def estimate_nb_alpha(d: pd.DataFrame, experience_var: str):
    formula = (
        "Fixation_count ~ C(Device_name, Treatment(reference='ATG')) "
        f"+ C({experience_var}) + Repetition_c + Attempt_sequence_c"
    )
    nb = smf.negativebinomial(
        formula,
        data=d,
        offset=d["log_attempt_seconds"],
    ).fit(disp=False, maxiter=1000)
    alpha = float(nb.params.get("alpha", 1.0))
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = 1.0
    return alpha, nb


def fit_count_model(d: pd.DataFrame, experience_var: str, suffix: str) -> None:
    formula = (
        "Fixation_count ~ C(Device_name, Treatment(reference='ATG')) "
        f"+ C({experience_var}) + Repetition_c + Attempt_sequence_c"
    )
    alpha, nb_independent = estimate_nb_alpha(d, experience_var)
    (OUT_DIR / f"count_alpha_estimation_{suffix}.txt").write_text(
        nb_independent.summary().as_text() + f"\n\nEstimated alpha used in GEE: {alpha:.6f}\n",
        encoding="utf-8",
    )

    gee = smf.gee(
        formula,
        groups="Participant",
        data=d,
        family=sm.families.NegativeBinomial(alpha=alpha),
        cov_struct=Exchangeable(),
        offset=d["log_attempt_seconds"],
    )
    result = gee.fit(cov_type="bias_reduced", maxiter=200)
    (OUT_DIR / f"count_negative_binomial_gee_{suffix}.txt").write_text(
        result.summary().as_text() + f"\n\nNB alpha fixed at {alpha:.6f}\n",
        encoding="utf-8",
    )

    device_rows = [
        _contrast_result(
            result,
            "CON vs ATG",
            {"C(Device_name, Treatment(reference='ATG'))[T.CON]": 1},
            True,
        ),
        _contrast_result(
            result,
            "MST vs ATG",
            {"C(Device_name, Treatment(reference='ATG'))[T.MST]": 1},
            True,
        ),
        _contrast_result(
            result,
            "CON vs MST",
            {
                "C(Device_name, Treatment(reference='ATG'))[T.CON]": 1,
                "C(Device_name, Treatment(reference='ATG'))[T.MST]": -1,
            },
            True,
        ),
    ]
    _apply_holm(device_rows).to_csv(
        OUT_DIR / f"count_device_contrasts_{suffix}.csv", index=False
    )

    if experience_var == "Experience3":
        some = "C(Experience3)[T.Some prior experience]"
        no = "C(Experience3)[T.No prior experience]"
        exp_rows = [
            _contrast_result(result, "Some vs higher", {some: 1}, True),
            _contrast_result(result, "No vs higher", {no: 1}, True),
            _contrast_result(result, "No vs some", {no: 1, some: -1}, True),
        ]
        exp_terms = ["C(Experience3)"]
    else:
        any_term = "C(AnyExperience)[T.Any previous experience]"
        exp_rows = [
            _contrast_result(
                result, "Any vs no previous experience", {any_term: 1}, True
            )
        ]
        exp_terms = ["C(AnyExperience)"]

    _apply_holm(exp_rows).to_csv(
        OUT_DIR / f"count_experience_contrasts_{suffix}.csv", index=False
    )
    pd.concat(
        [
            _overall_wald(result, ["C(Device_name"], "Device"),
            _overall_wald(result, exp_terms, "Experience"),
        ],
        ignore_index=True,
    ).to_csv(OUT_DIR / f"count_overall_tests_{suffix}.csv", index=False)


def main() -> None:
    d = prepare_data(DATA_FILE)
    d.to_csv(OUT_DIR / "analysis_dataset_used.csv", index=False)

    fit_duration_model(d, "Experience3", "three_level")
    fit_duration_model(d, "AnyExperience", "binary")
    fit_count_model(d, "Experience3", "three_level")
    fit_count_model(d, "AnyExperience", "binary")

    readme = f"""Python-only confirmatory fixation analyses\n\nRows analysed: {len(d)}\nParticipants: {d['Participant'].nunique()}\n\nDuration model:\n- Linear mixed-effects model on log mean fixation duration\n- Participant random intercept\n- ML estimation\n\nCount model:\n- Negative-binomial GEE clustered by participant\n- Exchangeable working correlation\n- Bias-reduced covariance\n- Log attempt duration offset\n- Alpha estimated from a negative-binomial regression and then fixed in GEE\n\nImportant: the count model is a population-averaged repeated-measures model, not a subject-specific random-intercept GLMM. Describe it as GEE in the manuscript.\n"""
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")
    print(f"Completed. Results written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

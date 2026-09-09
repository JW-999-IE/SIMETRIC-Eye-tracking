# save_as: procedural_success_gee.py
from __future__ import annotations

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial

warnings.filterwarnings("default")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "procedural_success_gee_results"

# The script searches these locations in order.
DATA_CANDIDATES = [
    SCRIPT_DIR / "attempt_intervals_cleaned.csv",
    SCRIPT_DIR / "corrected_fixation_results_v3" / "attempt_intervals_cleaned.csv",
    SCRIPT_DIR / "corrected_fixation_results_v2" / "attempt_intervals_cleaned.csv",
]

# Primary analysis excludes attempts marked Abort_participant=True.
#
# Set this to True ONLY if all 24 aborted attempts represent inability to
# complete the procedure and should legitimately be classified as failures.
# Leave False for technical, administrative, safety-related, or otherwise
# non-procedural discontinuations.
RUN_ABORT_AS_FAILURE_SENSITIVITY = False

# The original cleaned file may not contain experience categories for the
# discontinued participants. To run the optional 438-attempt sensitivity model,
# enter only verified categories here, for example:
# ABORT_EXPERIENCE_MAP = {
#     "P7": "No prior experience",
#     "P8": "Some prior experience",
#     "P14": "Higher prior experience",
# }
ABORT_EXPERIENCE_MAP: dict[str, str] = {}

DEVICE_LEVELS = ["ATG", "CON", "MST"]
EXPERIENCE3_LEVELS = [
    "Higher prior experience",
    "Some prior experience",
    "No prior experience",
]
ANY_EXPERIENCE_LEVELS = ["No previous experience", "Any previous experience"]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def find_data_file() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    tried = "\n".join(f"  - {p}" for p in DATA_CANDIDATES)
    raise FileNotFoundError(
        "Could not find attempt_intervals_cleaned.csv.\n"
        "Copy it next to this script or into corrected_fixation_results_v3.\n"
        f"Locations checked:\n{tried}"
    )


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True, "1": True, "yes": True, "y": True,
            "false": False, "0": False, "no": False, "n": False,
        })
        .fillna(False)
        .astype(bool)
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise multiplicity adjustment."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted_sorted[rank] = running_max

    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted.tolist()


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    return "<.001" if p < 0.001 else f"{p:.3f}"


def write_model_text(result, path: Path, note: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(result.summary().as_text())
        f.write("\n\n")
        f.write(note.strip())
        f.write("\n")


def wald_block(result, term_names: list[str], label: str) -> dict:
    term_names = [t for t in term_names if t in result.params.index]
    if not term_names:
        raise KeyError(f"No terms found for overall test: {label}")

    beta = result.params.loc[term_names].to_numpy(dtype=float)
    cov = result.cov_params().loc[term_names, term_names].to_numpy(dtype=float)
    inv_cov = np.linalg.pinv(cov)
    statistic = float(beta.T @ inv_cov @ beta)
    df = int(np.linalg.matrix_rank(cov))
    p_value = float(chi2.sf(statistic, df))

    return {
        "Effect": label,
        "Wald_chi_square": statistic,
        "df": df,
        "p_value": p_value,
        "p_formatted": format_p(p_value),
    }


def coefficient_vector(
    result,
    factor_prefix: str,
    level: str,
    reference: str,
) -> np.ndarray:
    vector = np.zeros(len(result.params), dtype=float)
    if level == reference:
        return vector

    candidates = [
        i for i, name in enumerate(result.params.index)
        if name.startswith(factor_prefix) and name.endswith(f"[T.{level}]")
    ]
    if len(candidates) != 1:
        raise KeyError(
            f"Could not uniquely find coefficient for {level!r}. "
            f"Candidates: {[result.params.index[i] for i in candidates]}"
        )
    vector[candidates[0]] = 1.0
    return vector


def pairwise_odds_ratios(
    result,
    factor_prefix: str,
    levels: list[str],
    reference: str,
    comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    params = result.params.to_numpy(dtype=float)
    cov = result.cov_params().to_numpy(dtype=float)
    rows = []

    for first, second in comparisons:
        c = (
            coefficient_vector(result, factor_prefix, first, reference)
            - coefficient_vector(result, factor_prefix, second, reference)
        )
        estimate = float(c @ params)
        variance = float(c @ cov @ c)
        se = math.sqrt(max(variance, 0.0))
        z_value = estimate / se if se > 0 else np.nan
        p_value = float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
        lower = estimate - 1.96 * se
        upper = estimate + 1.96 * se

        rows.append({
            "Comparison": f"{first} vs {second}",
            "Log_odds_difference": estimate,
            "SE": se,
            "z_value": z_value,
            "p_unadjusted": p_value,
            "Odds_ratio": math.exp(estimate),
            "CI_low": math.exp(lower),
            "CI_high": math.exp(upper),
        })

    adjusted = holm_adjust([row["p_unadjusted"] for row in rows])
    for row, p_adj in zip(rows, adjusted):
        row["p_Holm"] = p_adj
        row["p_Holm_formatted"] = format_p(p_adj)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
def load_and_prepare(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)

    required = {
        "Participant", "Device_name", "Expertise",
        "Repetition", "Attempt_sequence", "Abort_participant",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")

    if "Success" not in d.columns:
        if "FTIS" not in d.columns:
            raise KeyError("Neither Success nor FTIS is present.")
        d["Success"] = (
            d["FTIS"].astype(str).str.strip().str.upper().map({"Y": 1, "N": 0})
        )

    d["Abort_participant"] = parse_bool(d["Abort_participant"])
    d["Success"] = pd.to_numeric(d["Success"], errors="coerce")
    d["Repetition"] = pd.to_numeric(d["Repetition"], errors="coerce")
    d["Attempt_sequence"] = pd.to_numeric(d["Attempt_sequence"], errors="coerce")
    d["Device_name"] = d["Device_name"].astype(str).str.strip().str.upper()
    d["Participant"] = d["Participant"].astype(str).str.strip()

    experience_map = {
        "Expert": "Higher prior experience",
        "Intermediate": "Some prior experience",
        "Novice": "No prior experience",
        "Higher prior experience": "Higher prior experience",
        "Some prior experience": "Some prior experience",
        "No prior experience": "No prior experience",
    }
    d["Experience3"] = d["Expertise"].map(experience_map)
    d["AnyExperience"] = np.where(
        d["Experience3"].eq("No prior experience"),
        "No previous experience",
        "Any previous experience",
    )

    return d


def finalize_model_data(d: pd.DataFrame, label: str) -> pd.DataFrame:
    d = d.copy()

    d["Device_name"] = pd.Categorical(
        d["Device_name"], categories=DEVICE_LEVELS, ordered=False
    )
    d["Experience3"] = pd.Categorical(
        d["Experience3"], categories=EXPERIENCE3_LEVELS, ordered=False
    )
    d["AnyExperience"] = pd.Categorical(
        d["AnyExperience"], categories=ANY_EXPERIENCE_LEVELS, ordered=False
    )

    before = len(d)
    d = d.dropna(
        subset=[
            "Participant", "Device_name", "Experience3",
            "Success", "Repetition", "Attempt_sequence",
        ]
    ).copy()
    d = d[d["Success"].isin([0, 1])].copy()
    d["Success"] = d["Success"].astype(int)

    if len(d) != before:
        print(
            f"WARNING: {label}: dropped {before - len(d)} rows "
            "with missing/invalid model fields."
        )
    return d


def center_covariates(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["Repetition_c"] = d["Repetition"] - d["Repetition"].mean()
    d["Attempt_sequence_c"] = (
        d["Attempt_sequence"] - d["Attempt_sequence"].mean()
    )
    return d


def descriptive_table(d: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = (
        d.groupby(group_col, observed=False)["Success"]
        .agg(Successes="sum", Total="count")
        .reset_index()
    )
    out["Failures"] = out["Total"] - out["Successes"]
    out["Success_rate"] = out["Successes"] / out["Total"]
    out["Success_percent"] = 100 * out["Success_rate"]
    return out


# ---------------------------------------------------------------------
# GEE models
# ---------------------------------------------------------------------
def fit_success_gee(d: pd.DataFrame, experience_var: str):
    if experience_var == "Experience3":
        exp_term = (
            "C(Experience3, "
            "Treatment(reference='Higher prior experience'))"
        )
    elif experience_var == "AnyExperience":
        exp_term = (
            "C(AnyExperience, "
            "Treatment(reference='No previous experience'))"
        )
    else:
        raise ValueError(experience_var)

    formula = (
        "Success ~ "
        "C(Device_name, Treatment(reference='ATG')) + "
        f"{exp_term} + "
        "Repetition_c + Attempt_sequence_c"
    )

    model = smf.gee(
        formula=formula,
        groups="Participant",
        data=d,
        cov_struct=Exchangeable(),
        family=Binomial(),
    )
    return model.fit(
        cov_type="bias_reduced",
        maxiter=500,
    )


def run_analysis_set(d: pd.DataFrame, output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    d = center_covariates(d)

    d.to_csv(output_dir / "analysis_dataset_used.csv", index=False)
    descriptive_table(d, "Device_name").to_csv(
        output_dir / "success_descriptives_by_device.csv", index=False
    )
    descriptive_table(d, "Experience3").to_csv(
        output_dir / "success_descriptives_by_experience3.csv", index=False
    )
    descriptive_table(d, "AnyExperience").to_csv(
        output_dir / "success_descriptives_by_any_experience.csv", index=False
    )

    readme_lines = [
        f"Analysis set: {label}",
        f"Rows analysed: {len(d)}",
        f"Participants: {d['Participant'].nunique()}",
        "",
        "Primary model:",
        "- Binomial GEE with logit link",
        "- Participant clustering",
        "- Exchangeable working correlation",
        "- Bias-reduced covariance",
        "- Device, experience, repetition, and chronological attempt sequence",
        "- Holm-adjusted pairwise comparisons",
        "",
        "Interpretation:",
        "- Odds ratios are population-averaged GEE estimates.",
        "- The analysis accounts for repeated attempts from each participant.",
    ]
    (output_dir / "README.txt").write_text(
        "\n".join(readme_lines), encoding="utf-8"
    )

    # Three-level experience analysis
    result3 = fit_success_gee(d, "Experience3")
    write_model_text(
        result3,
        output_dir / "success_gee_three_level.txt",
        note=(
            "Method: binomial GEE with logit link; participant cluster; "
            "exchangeable working correlation; covariance=bias_reduced."
        ),
    )

    device_terms = [
        term for term in result3.params.index
        if term.startswith("C(Device_name")
    ]
    exp3_terms = [
        term for term in result3.params.index
        if term.startswith("C(Experience3")
    ]
    overall3 = pd.DataFrame([
        wald_block(result3, device_terms, "Device"),
        wald_block(result3, exp3_terms, "Experience3"),
    ])
    overall3.to_csv(
        output_dir / "success_overall_tests_three_level.csv", index=False
    )

    device_contrasts3 = pairwise_odds_ratios(
        result3,
        factor_prefix="C(Device_name",
        levels=DEVICE_LEVELS,
        reference="ATG",
        comparisons=[
            ("CON", "ATG"),
            ("MST", "ATG"),
            ("CON", "MST"),
        ],
    )
    device_contrasts3.to_csv(
        output_dir / "success_device_contrasts_three_level.csv", index=False
    )

    experience_contrasts3 = pairwise_odds_ratios(
        result3,
        factor_prefix="C(Experience3",
        levels=EXPERIENCE3_LEVELS,
        reference="Higher prior experience",
        comparisons=[
            ("Some prior experience", "Higher prior experience"),
            ("No prior experience", "Higher prior experience"),
            ("No prior experience", "Some prior experience"),
        ],
    )
    experience_contrasts3.to_csv(
        output_dir / "success_experience_contrasts_three_level.csv", index=False
    )

    # Binary experience sensitivity analysis
    result_binary = fit_success_gee(d, "AnyExperience")
    write_model_text(
        result_binary,
        output_dir / "success_gee_binary_experience.txt",
        note=(
            "Method: binomial GEE with logit link; participant cluster; "
            "exchangeable working correlation; covariance=bias_reduced."
        ),
    )

    device_terms_b = [
        term for term in result_binary.params.index
        if term.startswith("C(Device_name")
    ]
    exp_b_terms = [
        term for term in result_binary.params.index
        if term.startswith("C(AnyExperience")
    ]
    overall_b = pd.DataFrame([
        wald_block(result_binary, device_terms_b, "Device"),
        wald_block(result_binary, exp_b_terms, "AnyExperience"),
    ])
    overall_b.to_csv(
        output_dir / "success_overall_tests_binary_experience.csv", index=False
    )

    pairwise_odds_ratios(
        result_binary,
        factor_prefix="C(Device_name",
        levels=DEVICE_LEVELS,
        reference="ATG",
        comparisons=[
            ("CON", "ATG"),
            ("MST", "ATG"),
            ("CON", "MST"),
        ],
    ).to_csv(
        output_dir / "success_device_contrasts_binary_experience.csv", index=False
    )

    pairwise_odds_ratios(
        result_binary,
        factor_prefix="C(AnyExperience",
        levels=ANY_EXPERIENCE_LEVELS,
        reference="No previous experience",
        comparisons=[
            ("Any previous experience", "No previous experience"),
        ],
    ).to_csv(
        output_dir / "success_experience_contrast_binary.csv", index=False
    )

    print(f"INFO: Completed {label}")
    print(f"INFO: Rows analysed: {len(d)}")
    print(f"INFO: Participants: {d['Participant'].nunique()}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    data_file = find_data_file()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"INFO: Reading {data_file}")
    all_attempts = load_and_prepare(data_file)

    # Always export aborted attempts for manual classification.
    aborted = all_attempts[all_attempts["Abort_participant"]].copy()
    aborted.to_csv(OUTPUT_DIR / "aborted_attempts_review.csv", index=False)

    # Primary: exclude all attempts marked as aborted.
    primary = all_attempts[~all_attempts["Abort_participant"]].copy()
    primary = finalize_model_data(primary, "Primary non-aborted analysis")
    run_analysis_set(
        primary,
        OUTPUT_DIR / "primary_nonabort_414",
        label="Primary non-aborted analysis",
    )

    # Optional sensitivity: classify every marked abort as procedural failure.
    if RUN_ABORT_AS_FAILURE_SENSITIVITY:
        sensitivity = all_attempts.copy()

        # Fill verified experience categories for discontinued participants.
        for participant, category in ABORT_EXPERIENCE_MAP.items():
            sensitivity.loc[
                sensitivity["Participant"].eq(participant), "Experience3"
            ] = category

        sensitivity["AnyExperience"] = np.where(
            sensitivity["Experience3"].eq("No prior experience"),
            "No previous experience",
            np.where(
                sensitivity["Experience3"].notna(),
                "Any previous experience",
                np.nan,
            ),
        )
        sensitivity.loc[
            sensitivity["Abort_participant"], "Success"
        ] = 0

        missing_abort_experience = sorted(
            sensitivity.loc[
                sensitivity["Abort_participant"]
                & sensitivity["Experience3"].isna(),
                "Participant",
            ].unique()
        )

        if missing_abort_experience:
            message = (
                "The 438-attempt abort-as-failure sensitivity model was not run "
                "because verified experience categories are missing for: "
                + ", ".join(missing_abort_experience)
                + ". Add verified categories to ABORT_EXPERIENCE_MAP, or keep "
                  "the primary 414-attempt analysis only."
            )
            print("WARNING:", message)
            (
                OUTPUT_DIR / "sensitivity_not_run_missing_abort_experience.txt"
            ).write_text(message + "\n", encoding="utf-8")
        else:
            sensitivity = finalize_model_data(
                sensitivity,
                "Abort-as-failure sensitivity analysis",
            )
            run_analysis_set(
                sensitivity,
                OUTPUT_DIR / "sensitivity_all_aborts_as_failures_438",
                label=(
                    "Sensitivity analysis: all marked aborts classified as failures"
                ),
            )
    else:
        print(
            "INFO: Abort-as-failure sensitivity was not run. "
            "This is correct unless all marked aborts were genuine procedural failures."
        )

    print(f"INFO: Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

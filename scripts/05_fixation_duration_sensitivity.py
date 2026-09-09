# save_as: fixation_duration_sensitivity.py
"""
SIMETRIC fixation-duration sensitivity analyses.

This script performs two sensitivity analyses requested in response to reviewer
concerns about highly skewed fixation-duration distributions:

1. Attempt-level median fixation duration
   - Outcome: log(attempt-level median fixation duration in ms)
   - Model: linear mixed-effects model with participant random intercept

2. Mean fixation duration after excluding individual fixation events >10 seconds
   - Unique mapped fixation events are reassigned to attempts using the same
     event-midpoint rule as the corrected fixation extraction.
   - Outcome: log(attempt-level mean duration of retained events)
   - Model: linear mixed-effects model with participant random intercept

Both analyses adjust for:
- device;
- procedural-experience stratum;
- repetition;
- chronological attempt sequence.

Three-level experience is the principal exploratory specification. A binary
any-versus-no-prior-experience sensitivity model is also produced.

Required files:
- model_dataset_attempt_level.csv
- fixation_events_unique_all_recordings.csv

Place both files next to this script, or leave them inside
corrected_fixation_results_v3 beneath the script folder.
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "fixation_duration_sensitivity_results"

ATTEMPT_CANDIDATES = [
    SCRIPT_DIR / "model_dataset_attempt_level.csv",
    SCRIPT_DIR / "corrected_fixation_results_v3" / "model_dataset_attempt_level.csv",
    SCRIPT_DIR / "corrected_fixation_results_v2" / "model_dataset_attempt_level.csv",
]

EVENT_CANDIDATES = [
    SCRIPT_DIR / "fixation_events_unique_all_recordings.csv",
    SCRIPT_DIR / "corrected_fixation_results_v3" / "fixation_events_unique_all_recordings.csv",
    SCRIPT_DIR / "corrected_fixation_results_v2" / "fixation_events_unique_all_recordings.csv",
]

DEVICE_LEVELS = ["ATG", "CON", "MST"]
EXPERIENCE3_LEVELS = [
    "Higher prior experience",
    "Some prior experience",
    "No prior experience",
]
ANY_EXPERIENCE_LEVELS = [
    "No previous experience",
    "Any previous experience",
]
LONG_FIXATION_THRESHOLD_MS = 10_000.0


def locate_file(candidates: list[Path], required: set[str], label: str) -> Path:
    checked: list[str] = []
    for path in candidates:
        checked.append(str(path))
        if not path.exists():
            continue
        cols = set(pd.read_csv(path, nrows=2).columns)
        if required.issubset(cols):
            return path
        checked[-1] += f" [missing {sorted(required - cols)}]"

    for name in {p.name for p in candidates}:
        for path in SCRIPT_DIR.rglob(name):
            if OUTPUT_DIR in path.parents:
                continue
            checked.append(str(path))
            try:
                cols = set(pd.read_csv(path, nrows=2).columns)
            except Exception as exc:
                checked[-1] += f" [read error: {exc}]"
                continue
            if required.issubset(cols):
                return path
            checked[-1] += f" [missing {sorted(required - cols)}]"

    raise FileNotFoundError(
        f"Could not find a usable {label} file.\n"
        + "\n".join(f"  - {x}" for x in checked)
    )


def prepare_attempt_data(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)

    required = {
        "Participant",
        "File",
        "Attempt_ID",
        "Attempt_start_ms",
        "Attempt_end_ms",
        "Device_name",
        "Expertise",
        "Repetition",
        "Attempt_sequence",
        "Fixation_count",
        "Mean_fixation_duration_ms",
        "Median_fixation_duration_ms",
        "Q1_fixation_duration_ms",
        "Q3_fixation_duration_ms",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise KeyError(f"Attempt dataset is missing: {missing}")

    numeric = [
        "Attempt_start_ms",
        "Attempt_end_ms",
        "Repetition",
        "Attempt_sequence",
        "Fixation_count",
        "Mean_fixation_duration_ms",
        "Median_fixation_duration_ms",
        "Q1_fixation_duration_ms",
        "Q3_fixation_duration_ms",
    ]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    experience_map = {
        "Expert": "Higher prior experience",
        "Intermediate": "Some prior experience",
        "Novice": "No prior experience",
        "Higher prior experience": "Higher prior experience",
        "Some prior experience": "Some prior experience",
        "No prior experience": "No prior experience",
    }
    d["Experience3"] = d["Expertise"].map(experience_map)
    d["AnyExperience"] = pd.Series(pd.NA, index=d.index, dtype="object")
    d.loc[
        d["Experience3"].eq("No prior experience"),
        "AnyExperience",
    ] = "No previous experience"
    d.loc[
        d["Experience3"].notna()
        & ~d["Experience3"].eq("No prior experience"),
        "AnyExperience",
    ] = "Any previous experience"

    d["Participant"] = d["Participant"].astype(str).str.strip()
    d["File"] = d["File"].astype(str).str.strip()
    d["Attempt_ID"] = d["Attempt_ID"].astype(str).str.strip()
    d["Device_name"] = d["Device_name"].astype(str).str.strip().str.upper()

    d = d.dropna(
        subset=[
            "Participant",
            "File",
            "Attempt_ID",
            "Attempt_start_ms",
            "Attempt_end_ms",
            "Device_name",
            "Experience3",
            "AnyExperience",
            "Repetition",
            "Attempt_sequence",
            "Median_fixation_duration_ms",
        ]
    ).copy()
    d = d[
        d["Device_name"].isin(DEVICE_LEVELS)
        & d["Median_fixation_duration_ms"].gt(0)
        & d["Attempt_end_ms"].gt(d["Attempt_start_ms"])
    ].copy()

    d["Device_name"] = pd.Categorical(
        d["Device_name"], categories=DEVICE_LEVELS
    )
    d["Experience3"] = pd.Categorical(
        d["Experience3"], categories=EXPERIENCE3_LEVELS
    )
    d["AnyExperience"] = pd.Categorical(
        d["AnyExperience"], categories=ANY_EXPERIENCE_LEVELS
    )

    d["Repetition_c"] = d["Repetition"] - d["Repetition"].mean()
    d["Attempt_sequence_c"] = (
        d["Attempt_sequence"] - d["Attempt_sequence"].mean()
    )
    d["log_median_fixation_duration"] = np.log(
        d["Median_fixation_duration_ms"]
    )

    if d["Attempt_ID"].duplicated().any():
        duplicates = d.loc[
            d["Attempt_ID"].duplicated(keep=False),
            ["Attempt_ID", "Participant", "File"],
        ]
        raise ValueError(
            "Attempt_ID is not unique.\n" + duplicates.head(20).to_string(index=False)
        )
    return d


def prepare_event_data(path: Path) -> pd.DataFrame:
    e = pd.read_csv(path, low_memory=False)
    required = {
        "Participant",
        "File",
        "Fixation_midpoint_ms",
        "Fixation_duration_ms",
        "Fixation_index",
    }
    missing = sorted(required - set(e.columns))
    if missing:
        raise KeyError(f"Event dataset is missing: {missing}")

    e["Participant"] = e["Participant"].astype(str).str.strip()
    e["File"] = e["File"].astype(str).str.strip()
    e["Fixation_midpoint_ms"] = pd.to_numeric(
        e["Fixation_midpoint_ms"], errors="coerce"
    )
    e["Fixation_duration_ms"] = pd.to_numeric(
        e["Fixation_duration_ms"], errors="coerce"
    )
    e = e.dropna(
        subset=[
            "Participant",
            "File",
            "Fixation_midpoint_ms",
            "Fixation_duration_ms",
            "Fixation_index",
        ]
    ).copy()
    e = e[e["Fixation_duration_ms"].gt(0)].copy()
    return e


def assign_events_to_attempts(
    attempts: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempt_rows: list[dict] = []
    event_rows: list[pd.DataFrame] = []

    grouped_events = {
        key: group.sort_values("Fixation_midpoint_ms")
        for key, group in events.groupby(["Participant", "File"], sort=False)
    }

    for (participant, file_name), attempt_group in attempts.groupby(
        ["Participant", "File"], sort=False, observed=True
    ):
        event_group = grouped_events.get((participant, file_name))
        if event_group is None:
            event_group = events.iloc[0:0].copy()

        for _, attempt in attempt_group.iterrows():
            selected = event_group[
                event_group["Fixation_midpoint_ms"].between(
                    attempt["Attempt_start_ms"],
                    attempt["Attempt_end_ms"],
                    inclusive="both",
                )
            ].copy()

            selected["Attempt_ID"] = attempt["Attempt_ID"]
            selected["Device_name"] = str(attempt["Device_name"])
            selected["Experience3"] = str(attempt["Experience3"])
            selected["Excluded_gt_10_seconds"] = (
                selected["Fixation_duration_ms"] > LONG_FIXATION_THRESHOLD_MS
            )
            event_rows.append(selected)

            retained = selected[
                selected["Fixation_duration_ms"] <= LONG_FIXATION_THRESHOLD_MS
            ]

            attempt_rows.append(
                {
                    "Attempt_ID": attempt["Attempt_ID"],
                    "Recalculated_fixation_count": len(selected),
                    "Retained_fixation_count_le_10s": len(retained),
                    "Excluded_fixation_count_gt_10s": int(
                        (
                            selected["Fixation_duration_ms"]
                            > LONG_FIXATION_THRESHOLD_MS
                        ).sum()
                    ),
                    "Trimmed_mean_fixation_duration_ms": (
                        retained["Fixation_duration_ms"].mean()
                        if len(retained)
                        else np.nan
                    ),
                    "Trimmed_median_fixation_duration_ms": (
                        retained["Fixation_duration_ms"].median()
                        if len(retained)
                        else np.nan
                    ),
                    "Maximum_assigned_fixation_duration_ms": (
                        selected["Fixation_duration_ms"].max()
                        if len(selected)
                        else np.nan
                    ),
                }
            )

    attempt_metrics = pd.DataFrame(attempt_rows)
    assigned_events = (
        pd.concat(event_rows, ignore_index=True)
        if event_rows
        else events.iloc[0:0].copy()
    )

    merged_check = attempts[
        ["Attempt_ID", "Fixation_count"]
    ].merge(
        attempt_metrics[
            ["Attempt_ID", "Recalculated_fixation_count"]
        ],
        on="Attempt_ID",
        how="left",
        validate="one_to_one",
    )
    mismatch = merged_check[
        merged_check["Fixation_count"].ne(
            merged_check["Recalculated_fixation_count"]
        )
    ]
    if not mismatch.empty:
        raise ValueError(
            "Recalculated event counts did not match the corrected attempt-level "
            "fixation counts. Examples:\n"
            + mismatch.head(20).to_string(index=False)
        )

    return attempt_metrics, assigned_events


def descriptive_table(d: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    def summarise(g: pd.DataFrame) -> pd.Series:
        values = g["Median_fixation_duration_ms"].dropna()
        return pd.Series(
            {
                "Participants": g["Participant"].nunique(),
                "Attempts": len(g),
                "Median_attempt_median_fixation_ms": values.median(),
                "Q1_attempt_median_fixation_ms": values.quantile(0.25),
                "Q3_attempt_median_fixation_ms": values.quantile(0.75),
                "IQR_attempt_median_fixation_ms": (
                    values.quantile(0.75) - values.quantile(0.25)
                ),
                "Median_attempt_mean_fixation_ms": (
                    g["Mean_fixation_duration_ms"].median()
                ),
                "Q1_attempt_mean_fixation_ms": (
                    g["Mean_fixation_duration_ms"].quantile(0.25)
                ),
                "Q3_attempt_mean_fixation_ms": (
                    g["Mean_fixation_duration_ms"].quantile(0.75)
                ),
            }
        )

    return (
        d.groupby(groups, observed=False, sort=False)
        .apply(summarise, include_groups=False)
        .reset_index()
    )


def pooled_event_descriptives(
    events: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    def summarise(g: pd.DataFrame) -> pd.Series:
        values = g["Fixation_duration_ms"].dropna()
        return pd.Series(
            {
                "Unique_fixation_events": len(values),
                "Median_event_duration_ms": values.median(),
                "Q1_event_duration_ms": values.quantile(0.25),
                "Q3_event_duration_ms": values.quantile(0.75),
                "IQR_event_duration_ms": (
                    values.quantile(0.75) - values.quantile(0.25)
                ),
                "Events_gt_10_seconds": int(
                    (values > LONG_FIXATION_THRESHOLD_MS).sum()
                ),
            }
        )

    return (
        events.groupby(groups, observed=False, sort=False)
        .apply(summarise, include_groups=False)
        .reset_index()
    )


def fit_mixedlm_with_fallback(
    formula: str,
    d: pd.DataFrame,
):
    failures: list[str] = []
    # Powell is used first because L-BFGS can falsely collapse the random
    # intercept variance to the boundary for these skew-robust outcomes.
    for method in ["powell", "bfgs", "nm", "cg"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    d,
                    groups=d["Participant"],
                    re_formula="1",
                )
                result = model.fit(
                    reml=False,
                    method=method,
                    maxiter=5000,
                    disp=False,
                )
            covariance = float(result.cov_re.iloc[0, 0])
            if (
                bool(result.converged)
                and np.isfinite(np.asarray(result.params)).all()
                and np.isfinite(np.asarray(result.cov_params())).all()
                and np.isfinite(covariance)
                and covariance >= 0
            ):
                warning_text = " | ".join(str(x.message) for x in caught)
                return result, method, warning_text
            failures.append(
                f"{method}: converged={result.converged}, "
                f"random-intercept variance={covariance}"
            )
        except Exception as exc:
            failures.append(f"{method}: {type(exc).__name__}: {exc}")

    raise RuntimeError("All MixedLM optimisers failed: " + " | ".join(failures))


def holm_adjust(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    valid = out["p_unadjusted"].notna()
    out["p_Holm"] = np.nan
    if valid.any():
        out.loc[valid, "p_Holm"] = multipletests(
            out.loc[valid, "p_unadjusted"].astype(float),
            method="holm",
        )[1]
    return out


def contrast(
    result,
    label: str,
    weights: dict[str, float],
) -> dict:
    names = list(result.params.index)
    w = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    beta = float(w @ result.params.to_numpy(dtype=float))
    covariance = np.asarray(result.cov_params(), dtype=float)
    variance = float(w @ covariance @ w)
    se = math.sqrt(max(variance, 0.0))
    z_value = beta / se if se > 0 else np.nan
    p_value = (
        float(2 * norm.sf(abs(z_value)))
        if np.isfinite(z_value)
        else np.nan
    )
    lower = beta - 1.96 * se
    upper = beta + 1.96 * se
    return {
        "Comparison": label,
        "Log_ratio": beta,
        "SE": se,
        "z_value": z_value,
        "p_unadjusted": p_value,
        "Geometric_mean_ratio": math.exp(beta),
        "CI_low": math.exp(lower),
        "CI_high": math.exp(upper),
    }


def overall_wald(
    result,
    term_prefixes: list[str],
    label: str,
) -> dict:
    names = list(result.params.index)
    selected = [
        i for i, name in enumerate(names)
        if any(name.startswith(prefix) for prefix in term_prefixes)
    ]
    if not selected:
        raise KeyError(f"No model terms found for {label}.")

    beta = result.params.to_numpy(dtype=float)[selected]
    covariance = np.asarray(result.cov_params(), dtype=float)[
        np.ix_(selected, selected)
    ]
    statistic = float(beta.T @ np.linalg.pinv(covariance) @ beta)
    df = int(np.linalg.matrix_rank(covariance))
    p_value = float(chi2.sf(statistic, df))
    return {
        "Effect": label,
        "Wald_chi_square": statistic,
        "df": df,
        "p_value": p_value,
        "p_formatted": (
            "<.001" if p_value < 0.001 else f"{p_value:.3f}"
        ),
    }


def run_model(
    d: pd.DataFrame,
    outcome_column: str,
    outcome_label: str,
    experience_var: str,
    suffix: str,
) -> None:
    if experience_var == "Experience3":
        experience_term = (
            "C(Experience3, "
            "Treatment(reference='Higher prior experience'))"
        )
        experience_prefix = "C(Experience3"
    else:
        experience_term = (
            "C(AnyExperience, "
            "Treatment(reference='No previous experience'))"
        )
        experience_prefix = "C(AnyExperience"

    formula = (
        f"{outcome_column} ~ "
        "C(Device_name, Treatment(reference='ATG')) + "
        f"{experience_term} + Repetition_c + Attempt_sequence_c"
    )
    result, optimiser, warnings_text = fit_mixedlm_with_fallback(formula, d)

    random_variance = float(result.cov_re.iloc[0, 0])
    summary_text = result.summary().as_text()
    summary_text += (
        "\n\nOutcome: " + outcome_label
        + "\nTransformation: natural logarithm."
        + "\nMethod: linear mixed-effects model with participant random intercept."
        + f"\nOptimiser: {optimiser}."
        + f"\nEstimated participant random-intercept variance: {random_variance:.6f}."
        + "\nExponentiated estimates are geometric mean ratios."
    )
    if warnings_text:
        summary_text += "\nWarnings captured: " + warnings_text
    summary_text += "\n"

    (
        OUTPUT_DIR / f"{suffix}_model.txt"
    ).write_text(summary_text, encoding="utf-8")

    device_rows = [
        contrast(
            result,
            "CON vs ATG",
            {
                "C(Device_name, Treatment(reference='ATG'))[T.CON]": 1.0
            },
        ),
        contrast(
            result,
            "MST vs ATG",
            {
                "C(Device_name, Treatment(reference='ATG'))[T.MST]": 1.0
            },
        ),
        contrast(
            result,
            "CON vs MST",
            {
                "C(Device_name, Treatment(reference='ATG'))[T.CON]": 1.0,
                "C(Device_name, Treatment(reference='ATG'))[T.MST]": -1.0,
            },
        ),
    ]
    holm_adjust(device_rows).to_csv(
        OUTPUT_DIR / f"{suffix}_device_contrasts.csv",
        index=False,
    )

    if experience_var == "Experience3":
        some = (
            "C(Experience3, "
            "Treatment(reference='Higher prior experience'))"
            "[T.Some prior experience]"
        )
        none = (
            "C(Experience3, "
            "Treatment(reference='Higher prior experience'))"
            "[T.No prior experience]"
        )
        experience_rows = [
            contrast(result, "Some vs higher", {some: 1.0}),
            contrast(result, "No vs higher", {none: 1.0}),
            contrast(
                result,
                "No vs some",
                {none: 1.0, some: -1.0},
            ),
        ]
    else:
        any_term = (
            "C(AnyExperience, "
            "Treatment(reference='No previous experience'))"
            "[T.Any previous experience]"
        )
        experience_rows = [
            contrast(
                result,
                "Any vs no previous experience",
                {any_term: 1.0},
            )
        ]

    holm_adjust(experience_rows).to_csv(
        OUTPUT_DIR / f"{suffix}_experience_contrasts.csv",
        index=False,
    )

    overall = pd.DataFrame(
        [
            overall_wald(
                result,
                ["C(Device_name"],
                "Device",
            ),
            overall_wald(
                result,
                [experience_prefix],
                experience_var,
            ),
        ]
    )
    overall.to_csv(
        OUTPUT_DIR / f"{suffix}_overall_tests.csv",
        index=False,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    attempt_path = locate_file(
        ATTEMPT_CANDIDATES,
        {
            "Participant",
            "Attempt_ID",
            "Median_fixation_duration_ms",
            "Fixation_count",
        },
        "attempt-level fixation dataset",
    )
    event_path = locate_file(
        EVENT_CANDIDATES,
        {
            "Participant",
            "File",
            "Fixation_midpoint_ms",
            "Fixation_duration_ms",
        },
        "unique fixation-event dataset",
    )

    print(f"INFO: Attempt file: {attempt_path}")
    print(f"INFO: Event file: {event_path}")

    attempts = prepare_attempt_data(attempt_path)
    events = prepare_event_data(event_path)

    attempt_trim, assigned_events = assign_events_to_attempts(
        attempts,
        events,
    )
    analysis = attempts.merge(
        attempt_trim,
        on="Attempt_ID",
        how="left",
        validate="one_to_one",
    )

    assigned_events.to_csv(
        OUTPUT_DIR / "assigned_unique_fixation_events.csv",
        index=False,
    )
    analysis.to_csv(
        OUTPUT_DIR / "attempt_level_sensitivity_dataset.csv",
        index=False,
    )

    descriptive_table(
        analysis,
        ["Device_name"],
    ).to_csv(
        OUTPUT_DIR / "fixation_duration_descriptives_by_device.csv",
        index=False,
    )
    descriptive_table(
        analysis,
        ["Experience3", "Device_name"],
    ).to_csv(
        OUTPUT_DIR
        / "fixation_duration_descriptives_by_experience_device.csv",
        index=False,
    )
    pooled_event_descriptives(
        assigned_events,
        ["Device_name"],
    ).to_csv(
        OUTPUT_DIR / "pooled_unique_event_descriptives_by_device.csv",
        index=False,
    )
    pooled_event_descriptives(
        assigned_events,
        ["Experience3", "Device_name"],
    ).to_csv(
        OUTPUT_DIR
        / "pooled_unique_event_descriptives_by_experience_device.csv",
        index=False,
    )

    affected = analysis[
        analysis["Excluded_fixation_count_gt_10s"].gt(0)
    ].copy()
    affected.to_csv(
        OUTPUT_DIR / "attempts_affected_by_fixations_gt_10_seconds.csv",
        index=False,
    )

    trimmed = analysis[
        analysis["Trimmed_mean_fixation_duration_ms"].gt(0)
    ].copy()
    trimmed["log_trimmed_mean_fixation_duration"] = np.log(
        trimmed["Trimmed_mean_fixation_duration_ms"]
    )

    flow = pd.DataFrame(
        [
            {
                "Analysis": "Attempt-level median fixation duration",
                "Attempts": len(analysis),
                "Participants": analysis["Participant"].nunique(),
                "Unique_fixation_events_initial": int(
                    analysis["Recalculated_fixation_count"].sum()
                ),
                "Events_excluded_gt_10_seconds": 0,
                "Attempts_affected_by_gt_10_seconds": 0,
                "Attempts_excluded_no_retained_events": 0,
            },
            {
                "Analysis": "Mean fixation duration after excluding events >10 seconds",
                "Attempts": len(trimmed),
                "Participants": trimmed["Participant"].nunique(),
                "Unique_fixation_events_initial": int(
                    analysis["Recalculated_fixation_count"].sum()
                ),
                "Events_excluded_gt_10_seconds": int(
                    analysis["Excluded_fixation_count_gt_10s"].sum()
                ),
                "Attempts_affected_by_gt_10_seconds": int(
                    analysis["Excluded_fixation_count_gt_10s"].gt(0).sum()
                ),
                "Attempts_excluded_no_retained_events": int(
                    analysis["Trimmed_mean_fixation_duration_ms"].isna().sum()
                ),
            },
        ]
    )
    flow.to_csv(
        OUTPUT_DIR / "fixation_duration_sensitivity_analysis_flow.csv",
        index=False,
    )

    run_model(
        analysis,
        "log_median_fixation_duration",
        "log attempt-level median fixation duration in milliseconds",
        "Experience3",
        "median_duration_three_level",
    )
    run_model(
        analysis,
        "log_median_fixation_duration",
        "log attempt-level median fixation duration in milliseconds",
        "AnyExperience",
        "median_duration_binary_experience",
    )

    run_model(
        trimmed,
        "log_trimmed_mean_fixation_duration",
        "log attempt-level mean fixation duration after excluding events >10 seconds",
        "Experience3",
        "trimmed_mean_duration_three_level",
    )
    run_model(
        trimmed,
        "log_trimmed_mean_fixation_duration",
        "log attempt-level mean fixation duration after excluding events >10 seconds",
        "AnyExperience",
        "trimmed_mean_duration_binary_experience",
    )

    readme = f"""SIMETRIC fixation-duration sensitivity analysis

Attempt-level input:
{attempt_path}

Unique-event input:
{event_path}

Sensitivity analysis 1:
Log attempt-level median fixation duration.
Attempts: {len(analysis)}
Participants: {analysis['Participant'].nunique()}

Sensitivity analysis 2:
Log attempt-level mean fixation duration after excluding individual mapped
fixation events longer than 10 seconds.
Initial assigned unique fixation events:
{int(analysis['Recalculated_fixation_count'].sum())}

Events excluded (>10 seconds):
{int(analysis['Excluded_fixation_count_gt_10s'].sum())}

Attempts containing at least one excluded event:
{int(analysis['Excluded_fixation_count_gt_10s'].gt(0).sum())}

Attempts excluded because no event remained:
{int(analysis['Trimmed_mean_fixation_duration_ms'].isna().sum())}

Final trimmed-mean attempts:
{len(trimmed)}

Model:
Linear mixed-effects regression on the natural-log outcome, with participant
random intercept and fixed effects for device, experience, repetition, and
chronological attempt sequence. Exponentiated estimates are geometric mean
ratios. Pairwise contrasts use Holm adjustment.

Interpretation:
These are sensitivity analyses. They test whether the principal null device
finding changes when the skewed outcome is represented by the attempt-level
median or when unusually long individual fixation events are removed.
"""
    (OUTPUT_DIR / "README.txt").write_text(readme, encoding="utf-8")

    print("INFO: Sensitivity analyses completed.")
    print(f"INFO: Attempt-level median model: {len(analysis)} attempts.")
    print(
        "INFO: Events >10 seconds excluded:",
        int(analysis["Excluded_fixation_count_gt_10s"].sum()),
    )
    print(
        "INFO: Attempts affected:",
        int(analysis["Excluded_fixation_count_gt_10s"].gt(0).sum()),
    )
    print(f"INFO: Trimmed-mean model: {len(trimmed)} attempts.")
    print(f"INFO: Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

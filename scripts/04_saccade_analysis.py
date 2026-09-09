"""
SIMETRIC reanalysis of ultrasound-mapped saccade-event rate and saccadic amplitude.

Place this script in the folder containing the raw Tobii TSV files. It reuses:
  corrected_fixation_results_v3/attempt_intervals_cleaned.csv
  corrected_fixation_results_v3/session_recording_map.csv

Primary outcomes
----------------
1. Ultrasound-mapped saccade-event rate:
   unique mapped saccade events / whole-attempt duration in seconds.
   Analysed using negative-binomial GEE clustered by participant, with log
   attempt duration as an offset.

2. Mean saccadic amplitude:
   If a native amplitude column is present, the event-level native value is
   used. Otherwise amplitude is reconstructed as the angular separation in
   degrees between the binocular gaze-direction centroids of the immediately
   preceding and following ultrasound-mapped fixation events. This derived
   definition is explicitly recorded in all output files.
   Attempt-level mean amplitude is log-transformed and analysed using Gaussian
   GEE clustered by participant.

Both models include device, prior procedural-experience stratum, repetition,
and chronological attempt sequence. Three-level experience is primary
exploratory; any-versus-none experience is a sensitivity analysis.

Run:
  py -m pip install pandas numpy scipy statsmodels
  py saccade_analysis_revised.py
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

warnings.filterwarnings("default")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "corrected_saccade_results"

ATTEMPT_CANDIDATES = [
    SCRIPT_DIR / "attempt_intervals_cleaned.csv",
    SCRIPT_DIR / "corrected_fixation_results_v3" / "attempt_intervals_cleaned.csv",
    SCRIPT_DIR / "corrected_fixation_results_v2" / "attempt_intervals_cleaned.csv",
]
SESSION_MAP_CANDIDATES = [
    SCRIPT_DIR / "session_recording_map.csv",
    SCRIPT_DIR / "corrected_fixation_results_v3" / "session_recording_map.csv",
    SCRIPT_DIR / "corrected_fixation_results_v2" / "session_recording_map.csv",
]

TIMESTAMP_COL = "Recording timestamp"
SENSOR_COL = "Sensor"
PARTICIPANT_COL = "Participant name"
RECORDING_COL = "Recording name"
MAPPED_TYPE_COL = "Mapped eye movement type [US]"
MAPPED_INDEX_COL = "Mapped eye movement type index [US]"
VALIDITY_LEFT_COL = "Validity left"
VALIDITY_RIGHT_COL = "Validity right"

DIRECTION_COLS = [
    "Gaze direction left X", "Gaze direction left Y", "Gaze direction left Z",
    "Gaze direction right X", "Gaze direction right Y", "Gaze direction right Z",
]

# These names are checked case-insensitively. Current uploaded TSVs do not
# contain a native amplitude field, so the adjacent-fixation angular method is
# expected to be used unless the data are re-exported with an amplitude metric.
NATIVE_AMPLITUDE_CANDIDATES = [
    "Saccadic amplitude",
    "Saccade amplitude",
    "Saccadic amplitude [deg]",
    "Saccade amplitude [deg]",
    "Eye movement event amplitude",
    "Eye movement event amplitude [deg]",
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
EXPERIENCE_MAP = {
    "Expert": "Higher prior experience",
    "Intermediate": "Some prior experience",
    "Novice": "No prior experience",
    "Higher prior experience": "Higher prior experience",
    "Some prior experience": "Some prior experience",
    "No prior experience": "No prior experience",
}

# Prevent pairing a saccade with fixation centroids separated by a long period
# of unmapped/invalid gaze. This is a QC rule, not a physiological threshold.
MAX_FIXATION_LINK_GAP_MS = 1000.0
CHUNK_SIZE = 200_000

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def find_existing(candidates: Iterable[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{checked}")


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype(str).str.strip().str.casefold().map({
            "true": True, "1": True, "yes": True, "y": True,
            "false": False, "0": False, "no": False, "n": False,
        }).fillna(False).astype(bool)
    )


def recording_number(value: object) -> int | None:
    match = re.search(r"(\d+)\s*$", str(value))
    return int(match.group(1)) if match else None


def build_tsv_map() -> dict[int, Path]:
    files = sorted(SCRIPT_DIR.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No TSV files found in {SCRIPT_DIR}")
    mapping: dict[int, Path] = {}
    duplicates: list[tuple[int, str, str]] = []
    for path in files:
        number = recording_number(path.stem)
        if number is None:
            continue
        if number in mapping:
            duplicates.append((number, str(mapping[number]), str(path)))
        else:
            mapping[number] = path
    if duplicates:
        pd.DataFrame(
            duplicates, columns=["Recording_number", "First_file", "Duplicate_file"]
        ).to_csv(OUTPUT_DIR / "duplicate_recording_number_files.csv", index=False)
        raise RuntimeError(
            "More than one TSV had the same trailing recording number. "
            "See duplicate_recording_number_files.csv."
        )
    return mapping


def locate_native_amplitude(header: list[str]) -> str | None:
    lookup = {c.casefold(): c for c in header}
    for candidate in NATIVE_AMPLITUDE_CANDIDATES:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    # Flexible fallback for a column containing both amplitude and saccade/event.
    matches = [
        c for c in header
        if "amplitude" in c.casefold()
        and ("sacc" in c.casefold() or "eye movement" in c.casefold())
    ]
    return matches[0] if len(matches) == 1 else None


def inspect_columns(path: Path) -> tuple[list[str], str | None, bool]:
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    required = [
        TIMESTAMP_COL, SENSOR_COL, PARTICIPANT_COL,
        MAPPED_TYPE_COL, MAPPED_INDEX_COL,
        VALIDITY_LEFT_COL, VALIDITY_RIGHT_COL,
    ]
    missing = [c for c in required if c not in header]
    if missing:
        raise KeyError(f"Missing required Tobii columns: {missing}")

    native = locate_native_amplitude(header)
    has_direction = all(c in header for c in DIRECTION_COLS)
    optional = [RECORDING_COL]
    if has_direction:
        optional.extend(DIRECTION_COLS)
    if native:
        optional.append(native)
    usecols = required + [c for c in optional if c not in required]
    return usecols, native, has_direction


def read_eye_rows(path: Path, usecols: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        eye = chunk[chunk[SENSOR_COL].astype(str).str.casefold().eq("eye tracker")]
        if not eye.empty:
            pieces.append(eye)
    if not pieces:
        return pd.DataFrame(columns=usecols)
    d = pd.concat(pieces, ignore_index=True)
    d[TIMESTAMP_COL] = pd.to_numeric(d[TIMESTAMP_COL], errors="coerce")
    d[MAPPED_INDEX_COL] = pd.to_numeric(d[MAPPED_INDEX_COL], errors="coerce")
    d = d.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return d


def valid_eye_mask(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.casefold().eq("valid").to_numpy()


def normalized_rows(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(array, axis=1)
    valid = np.isfinite(array).all(axis=1) & np.isfinite(norms) & (norms > 0)
    out = np.full_like(array, np.nan, dtype=float)
    out[valid] = array[valid] / norms[valid, None]
    return out, valid


def add_binocular_direction(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if not all(c in d.columns for c in DIRECTION_COLS):
        d[["_gx", "_gy", "_gz"]] = np.nan
        return d

    for col in DIRECTION_COLS:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    left = d[DIRECTION_COLS[:3]].to_numpy(dtype=float)
    right = d[DIRECTION_COLS[3:]].to_numpy(dtype=float)
    left_n, left_vector_ok = normalized_rows(left)
    right_n, right_vector_ok = normalized_rows(right)
    left_ok = left_vector_ok & valid_eye_mask(d[VALIDITY_LEFT_COL])
    right_ok = right_vector_ok & valid_eye_mask(d[VALIDITY_RIGHT_COL])

    combined = np.full((len(d), 3), np.nan, dtype=float)
    only_left = left_ok & ~right_ok
    only_right = right_ok & ~left_ok
    both = left_ok & right_ok
    combined[only_left] = left_n[only_left]
    combined[only_right] = right_n[only_right]
    if both.any():
        mean_vec = left_n[both] + right_n[both]
        norms = np.linalg.norm(mean_vec, axis=1)
        good = norms > 0
        rows = np.flatnonzero(both)
        combined[rows[good]] = mean_vec[good] / norms[good, None]

    d[["_gx", "_gy", "_gz"]] = combined
    return d


def centroid_vector(group: pd.DataFrame) -> pd.Series:
    values = group[["_gx", "_gy", "_gz"]].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return pd.Series({
            "Centroid_x": np.nan, "Centroid_y": np.nan, "Centroid_z": np.nan,
            "Valid_direction_samples": 0,
        })
    vector = values.mean(axis=0)
    norm_value = np.linalg.norm(vector)
    if not np.isfinite(norm_value) or norm_value <= 0:
        vector[:] = np.nan
    else:
        vector = vector / norm_value
    return pd.Series({
        "Centroid_x": vector[0], "Centroid_y": vector[1], "Centroid_z": vector[2],
        "Valid_direction_samples": len(values),
    })


def extract_events(
    eye: pd.DataFrame,
    participant: str,
    recording: str,
    file_name: str,
    native_amplitude_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    d = add_binocular_direction(eye)
    mapped_type = d[MAPPED_TYPE_COL].astype(str).str.strip().str.casefold()

    fx = d[mapped_type.eq("fixation") & d[MAPPED_INDEX_COL].notna()].copy()
    sc = d[mapped_type.eq("saccade") & d[MAPPED_INDEX_COL].notna()].copy()

    if fx.empty:
        fixations = pd.DataFrame(columns=[
            "Fixation_index", "Fixation_start_ms", "Fixation_end_ms",
            "Fixation_midpoint_ms", "Centroid_x", "Centroid_y", "Centroid_z",
            "Valid_direction_samples",
        ])
    else:
        base = (
            fx.groupby(MAPPED_INDEX_COL, sort=False)
            .agg(
                Fixation_start_ms=(TIMESTAMP_COL, "min"),
                Fixation_end_ms=(TIMESTAMP_COL, "max"),
                Fixation_eye_samples=(TIMESTAMP_COL, "size"),
            )
            .reset_index()
            .rename(columns={MAPPED_INDEX_COL: "Fixation_index"})
        )
        vectors = (
            fx.groupby(MAPPED_INDEX_COL, sort=False)
            .apply(centroid_vector, include_groups=False)
            .reset_index()
            .rename(columns={MAPPED_INDEX_COL: "Fixation_index"})
        )
        fixations = base.merge(vectors, on="Fixation_index", how="left")
        fixations["Fixation_midpoint_ms"] = (
            fixations["Fixation_start_ms"] + fixations["Fixation_end_ms"]
        ) / 2.0
        fixations = fixations.sort_values("Fixation_midpoint_ms").reset_index(drop=True)

    if sc.empty:
        return fixations, pd.DataFrame(), (
            "native_event_amplitude" if native_amplitude_col else
            "adjacent_mapped_fixation_centroid_angle"
        )

    aggregations: dict[str, tuple[str, str]] = {
        "Saccade_start_ms": (TIMESTAMP_COL, "min"),
        "Saccade_end_ms": (TIMESTAMP_COL, "max"),
        "Saccade_eye_samples": (TIMESTAMP_COL, "size"),
    }
    if native_amplitude_col:
        sc[native_amplitude_col] = pd.to_numeric(sc[native_amplitude_col], errors="coerce")
        aggregations["Native_amplitude_deg"] = (native_amplitude_col, "median")

    saccades = (
        sc.groupby(MAPPED_INDEX_COL, sort=False)
        .agg(**aggregations)
        .reset_index()
        .rename(columns={MAPPED_INDEX_COL: "Saccade_index"})
    )
    saccades["Saccade_midpoint_ms"] = (
        saccades["Saccade_start_ms"] + saccades["Saccade_end_ms"]
    ) / 2.0
    saccades["Participant"] = participant
    saccades["Recording"] = recording
    saccades["File"] = file_name

    if native_amplitude_col:
        amplitude = pd.to_numeric(saccades["Native_amplitude_deg"], errors="coerce")
        saccades["Saccade_amplitude_deg"] = amplitude.where(
            amplitude.gt(0) & amplitude.le(180)
        )
        saccades["Amplitude_method"] = "native_event_amplitude"
        saccades["Amplitude_QC_flag"] = np.where(
            saccades["Saccade_amplitude_deg"].notna(), "OK", "Missing_or_invalid_native_amplitude"
        )
        return fixations, saccades, "native_event_amplitude"

    saccades["Previous_fixation_index"] = np.nan
    saccades["Next_fixation_index"] = np.nan
    saccades["Previous_gap_ms"] = np.nan
    saccades["Next_gap_ms"] = np.nan
    saccades["Saccade_amplitude_deg"] = np.nan
    saccades["Amplitude_method"] = "adjacent_mapped_fixation_centroid_angle"
    saccades["Amplitude_QC_flag"] = "No_adjacent_valid_fixations"

    if fixations.empty:
        return fixations, saccades, "adjacent_mapped_fixation_centroid_angle"

    times = fixations["Fixation_midpoint_ms"].to_numpy(dtype=float)
    for idx, row in saccades.iterrows():
        midpoint = float(row["Saccade_midpoint_ms"])
        insertion = int(np.searchsorted(times, midpoint, side="left"))
        prev_pos = insertion - 1
        next_pos = insertion
        if prev_pos < 0 or next_pos >= len(fixations):
            continue
        previous = fixations.iloc[prev_pos]
        following = fixations.iloc[next_pos]
        previous_gap = midpoint - float(previous["Fixation_end_ms"])
        next_gap = float(following["Fixation_start_ms"]) - midpoint
        saccades.at[idx, "Previous_fixation_index"] = previous["Fixation_index"]
        saccades.at[idx, "Next_fixation_index"] = following["Fixation_index"]
        saccades.at[idx, "Previous_gap_ms"] = previous_gap
        saccades.at[idx, "Next_gap_ms"] = next_gap

        vectors = np.array([
            previous["Centroid_x"], previous["Centroid_y"], previous["Centroid_z"],
            following["Centroid_x"], following["Centroid_y"], following["Centroid_z"],
        ], dtype=float)
        if not np.isfinite(vectors).all():
            saccades.at[idx, "Amplitude_QC_flag"] = "Missing_fixation_direction"
            continue
        if previous_gap > MAX_FIXATION_LINK_GAP_MS or next_gap > MAX_FIXATION_LINK_GAP_MS:
            saccades.at[idx, "Amplitude_QC_flag"] = "Fixation_link_gap_over_1000ms"
            continue

        v1 = vectors[:3]
        v2 = vectors[3:]
        dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(dot)))
        if not np.isfinite(angle) or angle <= 0 or angle > 180:
            saccades.at[idx, "Amplitude_QC_flag"] = "Invalid_derived_angle"
            continue
        saccades.at[idx, "Saccade_amplitude_deg"] = angle
        saccades.at[idx, "Amplitude_QC_flag"] = "OK"

    return fixations, saccades, "adjacent_mapped_fixation_centroid_angle"


def attempt_metrics(
    eye: pd.DataFrame,
    fixations: pd.DataFrame,
    saccades: pd.DataFrame,
    attempts: pd.DataFrame,
    recording: str,
    file_name: str,
    amplitude_method: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, attempt in attempts.iterrows():
        start = float(attempt["Attempt_start_ms"])
        end = float(attempt["Attempt_end_ms"])
        duration_ms = end - start
        eye_count = int(eye[TIMESTAMP_COL].between(start, end, inclusive="both").sum())
        attempt_fixations = fixations[
            fixations["Fixation_midpoint_ms"].between(start, end, inclusive="both")
        ] if not fixations.empty else fixations
        attempt_saccades = saccades[
            saccades["Saccade_midpoint_ms"].between(start, end, inclusive="both")
        ] if not saccades.empty else saccades
        amplitudes = pd.to_numeric(
            attempt_saccades.get("Saccade_amplitude_deg", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()

        row = attempt.to_dict()
        row.update({
            "Recording": recording,
            "File": file_name,
            "Eye_sample_count": eye_count,
            "Mapped_fixation_count": int(len(attempt_fixations)),
            "Saccade_count": int(len(attempt_saccades)),
            "Saccade_rate_per_second": (
                len(attempt_saccades) / (duration_ms / 1000.0)
                if duration_ms > 0 else np.nan
            ),
            "Amplitude_saccade_count": int(len(amplitudes)),
            "Amplitude_valid_proportion": (
                len(amplitudes) / len(attempt_saccades)
                if len(attempt_saccades) > 0 else np.nan
            ),
            "Mean_saccade_amplitude_deg": (
                float(amplitudes.mean()) if not amplitudes.empty else np.nan
            ),
            "Median_saccade_amplitude_deg": (
                float(amplitudes.median()) if not amplitudes.empty else np.nan
            ),
            "Q1_saccade_amplitude_deg": (
                float(amplitudes.quantile(0.25)) if not amplitudes.empty else np.nan
            ),
            "Q3_saccade_amplitude_deg": (
                float(amplitudes.quantile(0.75)) if not amplitudes.empty else np.nan
            ),
            "Max_saccade_amplitude_deg": (
                float(amplitudes.max()) if not amplitudes.empty else np.nan
            ),
            "Amplitude_method": amplitude_method,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    sorted_adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (m - rank) * p[original_index])
        running = max(running, candidate)
        sorted_adjusted[rank] = running
    adjusted = np.empty(m, dtype=float)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = sorted_adjusted[rank]
    return adjusted.tolist()


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    return "<.001" if p < 0.001 else f"{p:.3f}"


def wald_block(result, term_names: list[str], label: str) -> dict:
    terms = [t for t in term_names if t in result.params.index]
    if not terms:
        raise KeyError(f"No terms found for {label}")
    beta = result.params.loc[terms].to_numpy(dtype=float)
    covariance = result.cov_params().loc[terms, terms].to_numpy(dtype=float)
    statistic = float(beta.T @ np.linalg.pinv(covariance) @ beta)
    df = int(np.linalg.matrix_rank(covariance))
    p = float(chi2.sf(statistic, df))
    return {
        "Effect": label,
        "Wald_chi_square": statistic,
        "df": df,
        "p_value": p,
        "p_formatted": format_p(p),
    }


def coefficient_vector(result, factor_prefix: str, level: str, reference: str) -> np.ndarray:
    vector = np.zeros(len(result.params), dtype=float)
    if level == reference:
        return vector
    matches = [
        i for i, term in enumerate(result.params.index)
        if term.startswith(factor_prefix) and term.endswith(f"[T.{level}]")
    ]
    if len(matches) != 1:
        raise KeyError(f"Could not identify coefficient for {level}: {matches}")
    vector[matches[0]] = 1.0
    return vector



def safe_exp(value: float) -> float:
    """Exponentiate while avoiding OverflowError in sparse diagnostic runs."""
    return float(np.exp(np.clip(value, -700, 700)))


def pairwise_ratios(
    result,
    factor_prefix: str,
    reference: str,
    comparisons: list[tuple[str, str]],
    ratio_name: str,
) -> pd.DataFrame:
    params = result.params.to_numpy(dtype=float)
    covariance = result.cov_params().to_numpy(dtype=float)
    rows = []
    for first, second in comparisons:
        c = (
            coefficient_vector(result, factor_prefix, first, reference)
            - coefficient_vector(result, factor_prefix, second, reference)
        )
        estimate = float(c @ params)
        variance = float(c @ covariance @ c)
        se = math.sqrt(max(variance, 0.0))
        z = estimate / se if se > 0 else np.nan
        p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
        rows.append({
            "Comparison": f"{first} vs {second}",
            "Log_ratio": estimate,
            "SE": se,
            "z_value": z,
            "p_unadjusted": p,
            ratio_name: safe_exp(estimate),
            "CI_low": safe_exp(estimate - 1.96 * se),
            "CI_high": safe_exp(estimate + 1.96 * se),
        })
    adjusted = holm_adjust([r["p_unadjusted"] for r in rows])
    for row, value in zip(rows, adjusted):
        row["p_Holm"] = value
        row["p_Holm_formatted"] = format_p(value)
    return pd.DataFrame(rows)


def fit_gee_with_fallback(formula: str, d: pd.DataFrame, family, offset=None):
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable, Independence

    attempts = [
        (Exchangeable(), "bias_reduced"),
        (Exchangeable(), "robust"),
        (Independence(), "robust"),
    ]
    errors = []
    for structure, covariance in attempts:
        try:
            model = smf.gee(
                formula=formula,
                groups="Participant",
                data=d,
                family=family,
                cov_struct=structure,
                offset=offset,
            )
            result = model.fit(cov_type=covariance, maxiter=500)
            if (
                bool(getattr(result, "converged", True))
                and np.isfinite(np.asarray(result.params)).all()
                and np.isfinite(np.asarray(result.cov_params())).all()
            ):
                return result, type(structure).__name__, covariance
            errors.append(f"{type(structure).__name__}/{covariance}: non-finite result")
        except Exception as exc:
            errors.append(f"{type(structure).__name__}/{covariance}: {exc}")
    raise RuntimeError("All GEE fitting strategies failed: " + " | ".join(errors))


def estimate_nb_alpha(formula: str, d: pd.DataFrame, offset: pd.Series) -> float:
    try:
        from statsmodels.discrete.discrete_model import NegativeBinomial
        model = NegativeBinomial.from_formula(formula, data=d, offset=offset)
        result = model.fit(disp=0, maxiter=500)
        if "alpha" in result.params.index:
            alpha = float(result.params["alpha"])
        else:
            alpha = 1.0
        if np.isfinite(alpha) and alpha > 0:
            return alpha
    except Exception as exc:
        LOGGER.warning("Independent NB alpha estimation failed; using alpha=1: %s", exc)
    return 1.0


def formulas(experience_var: str) -> tuple[str, str]:
    if experience_var == "Experience3":
        experience_term = (
            "C(Experience3, Treatment(reference='Higher prior experience'))"
        )
    else:
        experience_term = (
            "C(AnyExperience, Treatment(reference='No previous experience'))"
        )
    common = (
        "C(Device_name, Treatment(reference='ATG')) + "
        f"{experience_term} + Repetition_c + Attempt_sequence_c"
    )
    return f"Saccade_count ~ {common}", f"log_mean_saccade_amplitude ~ {common}"


def run_models(primary: pd.DataFrame) -> None:
    from statsmodels.genmod.families import Gaussian, NegativeBinomial, Poisson

    count_data = primary[
        primary["Attempt_duration_ms"].gt(0)
        & primary["Mapped_fixation_count"].gt(0)
        & primary["Saccade_count"].ge(0)
    ].copy()
    count_data["log_attempt_seconds"] = np.log(count_data["Attempt_duration_ms"] / 1000.0)

    amplitude_data = count_data[
        count_data["Mean_saccade_amplitude_deg"].gt(0)
        & count_data["Amplitude_saccade_count"].gt(0)
    ].copy()
    amplitude_data["log_mean_saccade_amplitude"] = np.log(
        amplitude_data["Mean_saccade_amplitude_deg"]
    )

    count_data.to_csv(OUTPUT_DIR / "model_dataset_saccade_rate.csv", index=False)
    amplitude_data.to_csv(OUTPUT_DIR / "model_dataset_saccade_amplitude.csv", index=False)

    model_flow = pd.DataFrame([
        {
            "Model": "Saccade-event rate",
            "Attempts": len(count_data),
            "Participants": count_data["Participant"].nunique(),
        },
        {
            "Model": "Mean saccadic amplitude",
            "Attempts": len(amplitude_data),
            "Participants": amplitude_data["Participant"].nunique(),
        },
    ])
    model_flow.to_csv(OUTPUT_DIR / "model_analysis_flow.csv", index=False)

    if count_data["Participant"].nunique() < 3:
        LOGGER.warning(
            "Fewer than three participant clusters are available; extraction "
            "completed, but inferential models were skipped."
        )
        return

    for experience_var, suffix in [
        ("Experience3", "three_level"),
        ("AnyExperience", "binary_experience"),
    ]:
        count_formula, amplitude_formula = formulas(experience_var)

        alpha = estimate_nb_alpha(
            count_formula, count_data, count_data["log_attempt_seconds"]
        )
        count_family_name = "NegativeBinomial"
        try:
            count_result, count_structure, count_cov = fit_gee_with_fallback(
                count_formula,
                count_data,
                NegativeBinomial(alpha=alpha),
                offset=count_data["log_attempt_seconds"],
            )
        except Exception as exc:
            LOGGER.warning("NB GEE failed; using Poisson GEE fallback: %s", exc)
            count_family_name = "Poisson_fallback"
            count_result, count_structure, count_cov = fit_gee_with_fallback(
                count_formula,
                count_data,
                Poisson(),
                offset=count_data["log_attempt_seconds"],
            )

        count_text = count_result.summary().as_text() + (
            f"\n\nCount family: {count_family_name}.\n"
            f"Alpha estimated by independent NB likelihood: {alpha:.6f}\n"
            "Method: clustered count GEE; participant cluster; "
            f"working correlation={count_structure}; covariance={count_cov}; "
            "whole-attempt duration offset.\n"
        )
        (OUTPUT_DIR / f"saccade_rate_{suffix}_gee.txt").write_text(
            count_text, encoding="utf-8"
        )

        device_terms = [
            x for x in count_result.params.index if x.startswith("C(Device_name")
        ]
        experience_prefix = "C(Experience3" if experience_var == "Experience3" else "C(AnyExperience"
        experience_terms = [
            x for x in count_result.params.index if x.startswith(experience_prefix)
        ]
        pd.DataFrame([
            wald_block(count_result, device_terms, "Device"),
            wald_block(count_result, experience_terms, experience_var),
        ]).to_csv(OUTPUT_DIR / f"saccade_rate_{suffix}_overall_tests.csv", index=False)

        pairwise_ratios(
            count_result,
            "C(Device_name",
            "ATG",
            [("CON", "ATG"), ("MST", "ATG"), ("CON", "MST")],
            "Incidence_rate_ratio",
        ).to_csv(OUTPUT_DIR / f"saccade_rate_{suffix}_device_contrasts.csv", index=False)

        if experience_var == "Experience3":
            exp_comparisons = [
                ("Some prior experience", "Higher prior experience"),
                ("No prior experience", "Higher prior experience"),
                ("No prior experience", "Some prior experience"),
            ]
            exp_reference = "Higher prior experience"
        else:
            exp_comparisons = [
                ("Any previous experience", "No previous experience")
            ]
            exp_reference = "No previous experience"
        pairwise_ratios(
            count_result,
            experience_prefix,
            exp_reference,
            exp_comparisons,
            "Incidence_rate_ratio",
        ).to_csv(OUTPUT_DIR / f"saccade_rate_{suffix}_experience_contrasts.csv", index=False)

        if amplitude_data.empty or amplitude_data["Participant"].nunique() < 3:
            LOGGER.warning("Insufficient amplitude data; amplitude models skipped.")
            continue

        amplitude_result, amplitude_structure, amplitude_cov = fit_gee_with_fallback(
            amplitude_formula, amplitude_data, Gaussian()
        )
        amplitude_text = amplitude_result.summary().as_text() + (
            "\n\nOutcome: log attempt-level mean saccadic amplitude.\n"
            "Exponentiated estimates are geometric mean ratios.\n"
            "Method: Gaussian GEE; participant cluster; "
            f"working correlation={amplitude_structure}; covariance={amplitude_cov}.\n"
        )
        (OUTPUT_DIR / f"saccade_amplitude_{suffix}_gee.txt").write_text(
            amplitude_text, encoding="utf-8"
        )

        amplitude_device_terms = [
            x for x in amplitude_result.params.index if x.startswith("C(Device_name")
        ]
        amplitude_experience_terms = [
            x for x in amplitude_result.params.index if x.startswith(experience_prefix)
        ]
        pd.DataFrame([
            wald_block(amplitude_result, amplitude_device_terms, "Device"),
            wald_block(amplitude_result, amplitude_experience_terms, experience_var),
        ]).to_csv(OUTPUT_DIR / f"saccade_amplitude_{suffix}_overall_tests.csv", index=False)

        pairwise_ratios(
            amplitude_result,
            "C(Device_name",
            "ATG",
            [("CON", "ATG"), ("MST", "ATG"), ("CON", "MST")],
            "Geometric_mean_ratio",
        ).to_csv(OUTPUT_DIR / f"saccade_amplitude_{suffix}_device_contrasts.csv", index=False)
        pairwise_ratios(
            amplitude_result,
            experience_prefix,
            exp_reference,
            exp_comparisons,
            "Geometric_mean_ratio",
        ).to_csv(OUTPUT_DIR / f"saccade_amplitude_{suffix}_experience_contrasts.csv", index=False)


def make_descriptives(primary: pd.DataFrame) -> pd.DataFrame:
    def q1(x):
        return x.quantile(0.25)
    def q3(x):
        return x.quantile(0.75)

    return (
        primary.groupby(["Experience3", "Device_name"], observed=False)
        .agg(
            Participants=("Participant", "nunique"),
            Attempts=("Attempt_ID", "size"),
            Median_saccade_rate_per_second=("Saccade_rate_per_second", "median"),
            Q1_saccade_rate_per_second=("Saccade_rate_per_second", q1),
            Q3_saccade_rate_per_second=("Saccade_rate_per_second", q3),
            Mean_saccade_rate_per_second=("Saccade_rate_per_second", "mean"),
            SD_saccade_rate_per_second=("Saccade_rate_per_second", "std"),
            Attempts_with_amplitude=("Mean_saccade_amplitude_deg", lambda x: x.notna().sum()),
            Median_attempt_mean_amplitude_deg=("Mean_saccade_amplitude_deg", "median"),
            Q1_attempt_mean_amplitude_deg=("Mean_saccade_amplitude_deg", q1),
            Q3_attempt_mean_amplitude_deg=("Mean_saccade_amplitude_deg", q3),
            Mean_attempt_mean_amplitude_deg=("Mean_saccade_amplitude_deg", "mean"),
            SD_attempt_mean_amplitude_deg=("Mean_saccade_amplitude_deg", "std"),
        )
        .reset_index()
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    attempt_path = find_existing(ATTEMPT_CANDIDATES, "attempt_intervals_cleaned.csv")
    session_map_path = find_existing(SESSION_MAP_CANDIDATES, "session_recording_map.csv")
    attempts = pd.read_csv(attempt_path, low_memory=False)
    session_map = pd.read_csv(session_map_path, low_memory=False)

    required_attempt = {
        "Participant", "Session_index", "Attempt_ID", "Attempt_start_ms",
        "Attempt_end_ms", "Attempt_duration_ms", "Device_name", "Expertise",
        "Abort_participant", "Repetition", "Attempt_sequence",
    }
    missing = sorted(required_attempt - set(attempts.columns))
    if missing:
        raise KeyError(f"Attempt file is missing columns: {missing}")
    required_map = {"Participant", "Session_index", "Recording"}
    missing_map = sorted(required_map - set(session_map.columns))
    if missing_map:
        raise KeyError(f"Session map is missing columns: {missing_map}")

    attempts["Abort_participant"] = parse_bool(attempts["Abort_participant"])
    attempts["Experience3"] = attempts["Expertise"].map(EXPERIENCE_MAP)
    attempts["AnyExperience"] = pd.Series(pd.NA, index=attempts.index, dtype="object")
    attempts.loc[
        attempts["Experience3"].eq("No prior experience"), "AnyExperience"
    ] = "No previous experience"
    attempts.loc[
        attempts["Experience3"].notna()
        & ~attempts["Experience3"].eq("No prior experience"),
        "AnyExperience",
    ] = "Any previous experience"

    tsv_map = build_tsv_map()
    assigned = session_map[session_map["Session_index"].notna()].copy()
    assigned["Session_index"] = assigned["Session_index"].astype(int)

    all_metrics: list[pd.DataFrame] = []
    all_saccades: list[pd.DataFrame] = []
    qc_rows: list[dict] = []

    for i, mapping_row in enumerate(assigned.itertuples(index=False), start=1):
        participant = str(mapping_row.Participant)
        session_index = int(mapping_row.Session_index)
        recording = str(mapping_row.Recording)
        number = recording_number(recording)
        file_path = tsv_map.get(number) if number is not None else None
        LOGGER.info(
            "[%d/%d] %s session %d -> %s",
            i, len(assigned), participant, session_index, recording,
        )
        session_attempts = attempts[
            attempts["Participant"].eq(participant)
            & attempts["Session_index"].eq(session_index)
        ].copy()

        if file_path is None or not file_path.exists():
            qc_rows.append({
                "Participant": participant,
                "Session_index": session_index,
                "Recording": recording,
                "Status": "TSV file not found",
            })
            continue

        try:
            usecols, native_col, has_direction = inspect_columns(file_path)
            eye = read_eye_rows(file_path, usecols)
            if eye.empty:
                raise ValueError("No Eye Tracker rows")
            fixations, saccades, amplitude_method = extract_events(
                eye, participant, recording, file_path.name, native_col
            )
            metrics = attempt_metrics(
                eye, fixations, saccades, session_attempts,
                recording, file_path.name, amplitude_method,
            )
            all_metrics.append(metrics)
            if not saccades.empty:
                all_saccades.append(saccades)
            qc_rows.append({
                "Participant": participant,
                "Session_index": session_index,
                "Recording": recording,
                "File": file_path.name,
                "Eye_tracker_rows": len(eye),
                "Unique_mapped_fixations": len(fixations),
                "Unique_mapped_saccades": len(saccades),
                "Saccades_with_amplitude": (
                    int(saccades["Saccade_amplitude_deg"].notna().sum())
                    if not saccades.empty else 0
                ),
                "Native_amplitude_column": native_col or "",
                "Direction_columns_available": has_direction,
                "Amplitude_method": amplitude_method,
                "Attempt_rows": len(metrics),
                "Status": "OK",
            })
        except Exception as exc:
            LOGGER.exception("Failed %s", file_path.name)
            qc_rows.append({
                "Participant": participant,
                "Session_index": session_index,
                "Recording": recording,
                "File": file_path.name,
                "Status": f"ERROR: {exc}",
            })

    pd.DataFrame(qc_rows).to_csv(OUTPUT_DIR / "file_session_QC.csv", index=False)
    if not all_metrics:
        raise RuntimeError("No attempt-level saccade metrics were created.")

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics = metrics.sort_values(["Participant", "Attempt_sequence"]).reset_index(drop=True)
    metrics.to_csv(OUTPUT_DIR / "attempt_level_saccade_metrics_all.csv", index=False)
    if all_saccades:
        saccade_events = pd.concat(all_saccades, ignore_index=True)
        saccade_events.to_csv(OUTPUT_DIR / "saccade_events_unique_all_recordings.csv", index=False)
        (
            saccade_events["Amplitude_QC_flag"].value_counts(dropna=False)
            .rename_axis("Amplitude_QC_flag")
            .reset_index(name="Saccade_events")
            .to_csv(OUTPUT_DIR / "amplitude_QC_summary.csv", index=False)
        )

    expected = set(attempts["Attempt_ID"])
    observed = set(metrics["Attempt_ID"])
    attempts[attempts["Attempt_ID"].isin(expected - observed)].to_csv(
        OUTPUT_DIR / "attempts_without_saccade_metrics.csv", index=False
    )

    primary = metrics[
        ~metrics["Abort_participant"]
        & metrics["Experience3"].notna()
        & metrics["Device_name"].isin(DEVICE_LEVELS)
        & metrics["Attempt_duration_ms"].gt(0)
        & metrics["Eye_sample_count"].gt(0)
        & metrics["Mapped_fixation_count"].gt(0)
    ].copy()
    primary["Device_name"] = pd.Categorical(primary["Device_name"], categories=DEVICE_LEVELS)
    primary["Experience3"] = pd.Categorical(primary["Experience3"], categories=EXPERIENCE3_LEVELS)
    primary["AnyExperience"] = pd.Categorical(primary["AnyExperience"], categories=ANY_EXPERIENCE_LEVELS)
    primary["Repetition_c"] = primary["Repetition"] - primary["Repetition"].mean()
    primary["Attempt_sequence_c"] = (
        primary["Attempt_sequence"] - primary["Attempt_sequence"].mean()
    )
    primary.to_csv(OUTPUT_DIR / "attempt_level_saccade_metrics_primary_nonabort.csv", index=False)
    make_descriptives(primary).to_csv(
        OUTPUT_DIR / "descriptives_by_experience_device.csv", index=False
    )

    run_models(primary)

    methods = sorted(primary["Amplitude_method"].dropna().astype(str).unique().tolist())
    readme = f"""SIMETRIC corrected saccade analysis

Attempt intervals: {attempt_path}
Session map: {session_map_path}
Assigned recording sessions processed: {len(assigned)}
Attempt rows extracted: {len(metrics)}
Primary non-abort mapped-gaze attempts: {len(primary)}
Primary participants: {primary['Participant'].nunique()}
Amplitude method(s): {', '.join(methods) if methods else 'none'}

Saccade rate is the number of unique ultrasound-mapped saccade events divided
by whole-attempt duration in seconds. It is not a count of TSV sample rows.

When no native amplitude column is available, amplitude is reconstructed as
the angular separation between binocular gaze-direction centroids of the
immediately preceding and following ultrasound-mapped fixation events. This
derived measure should be described explicitly and should not be presented as
a native Tobii Pro Lab amplitude export.

Primary models:
- Negative-binomial GEE for saccade-event count with whole-attempt duration offset.
- Gaussian GEE for log attempt-level mean saccadic amplitude.
- Participant clustering and exchangeable working correlation.
- Device, experience, repetition and chronological attempt sequence covariates.
- Holm-adjusted pairwise contrasts.
"""
    (OUTPUT_DIR / "README.txt").write_text(readme, encoding="utf-8")

    LOGGER.info("Completed. Results written to %s", OUTPUT_DIR)
    LOGGER.info("Primary attempts: %d; participants: %d", len(primary), primary["Participant"].nunique())


if __name__ == "__main__":
    main()

"""
SIMETRIC fixation extraction using insertion.xlsx attempt intervals (session-aware v3).

This version is for Tobii TSV exports that do NOT contain a TOI/Attempt column.
It uses Start time (ms) and End Time (ms) from insertion.xlsx to assign each
unique ultrasound-mapped fixation event to an insertion attempt.

Key corrections:
- counts unique mapped fixation events, not TSV rows;
- keeps one row per insertion attempt;
- reports skew-robust descriptives (median/IQR);
- retains abort rows in audit outputs but excludes them from the primary model;
- uses participant-clustered/repeated-measures models.

Put this script and insertion.xlsx in the same folder as the TSV files, then run:
    py fixation_analysis_revised_v3.py
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
SCRIPT_FOLDER = Path(__file__).resolve().parent
INPUT_FOLDER = SCRIPT_FOLDER
INTERVAL_FILE = SCRIPT_FOLDER / "insertion.xlsx"
OUTPUT_FOLDER = SCRIPT_FOLDER / "corrected_fixation_results_v3"

# Device coding used in insertion.xlsx.
DEVICE_MAP = {
    "DEVICE1": "MST",
    "DEVICE2": "ATG",
    "DEVICE3": "CON",
}

# Primary analysis excludes the 24 rows belonging to the three aborted
# participants, while retaining them in audit outputs.
ABORT_PARTICIPANTS = {"P7", "P8", "P14"}

EXPERTISE = {
    "P1": "Expert", "P3": "Expert", "P9": "Expert", "P12": "Expert", "P29": "Expert",
    "P2": "Intermediate", "P4": "Intermediate", "P5": "Intermediate", "P6": "Intermediate",
    "P10": "Intermediate", "P11": "Intermediate", "P16": "Intermediate", "P24": "Intermediate",
    "P28": "Intermediate",
    "P13": "Novice", "P15": "Novice", "P17": "Novice", "P18": "Novice", "P19": "Novice",
    "P20": "Novice", "P21": "Novice", "P22": "Novice", "P23": "Novice", "P25": "Novice",
    "P26": "Novice", "P27": "Novice", "P30": "Novice",
}

# Tobii columns. These match the uploaded Recording 15 export.
PARTICIPANT_COL = "Participant name"
RECORDING_COL = "Recording name"
TIMESTAMP_COL = "Recording timestamp"
SENSOR_COL = "Sensor"
FIXATION_TYPE_COL = "Mapped eye movement type [US]"
FIXATION_INDEX_COL = "Mapped eye movement type index [US]"
FIXATION_X_COL = "Mapped fixation X [US]"
FIXATION_Y_COL = "Mapped fixation Y [US]"
VALIDITY_LEFT_COL = "Validity left"
VALIDITY_RIGHT_COL = "Validity right"

# Review rather than automatically delete very long fixation events.
LONG_FIXATION_REVIEW_MS = 10_000

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 2. HELPERS
# -----------------------------------------------------------------------------
def normalize_participant(value: object) -> str:
    """Convert P7-abort or P7-1 to P7."""
    text = str(value).strip()
    match = re.search(r"\bP\s*(\d+)\b", text, flags=re.IGNORECASE)
    return f"P{int(match.group(1))}" if match else text


def clean_device_code(value: object) -> tuple[Optional[str], Optional[int], str]:
    """DEVICE2-3 -> (ATG, 3, DEVICE2-3)."""
    text = str(value).strip().upper().replace(" ", "")
    match = re.match(r"^(DEVICE[123])[-_]?([0-9]+)?$", text)
    if not match:
        return None, None, text
    base = match.group(1)
    repetition = int(match.group(2)) if match.group(2) else None
    return DEVICE_MAP.get(base), repetition, text


def at_least_one_eye_valid(left: pd.Series, right: pd.Series) -> pd.Series:
    left_valid = left.astype(str).str.casefold().eq("valid")
    right_valid = right.astype(str).str.casefold().eq("valid")
    return left_valid | right_valid


def load_attempt_intervals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Attempt interval file not found: {path}\n"
            "Copy insertion.xlsx into the same folder as this script, or edit INTERVAL_FILE."
        )

    d = pd.read_excel(path)
    required = {
        "Id", "Round", "Time", "Device", "FTIS",
        "Start time (ms)", "End Time (ms)", "Duration (ms)",
    }
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"insertion.xlsx is missing columns: {sorted(missing)}")

    d = d.copy()
    d["Participant_raw"] = d["Id"].astype(str)
    d["Participant"] = d["Id"].map(normalize_participant)
    d["Expertise"] = d["Participant"].map(EXPERTISE)
    d["Abort_participant"] = d["Participant"].isin(ABORT_PARTICIPANTS)

    parsed = d["Device"].map(clean_device_code)
    d[["Device_name", "Repetition", "Device_code_clean"]] = pd.DataFrame(
        parsed.tolist(), index=d.index
    )

    d["Attempt_start_ms"] = pd.to_numeric(d["Start time (ms)"], errors="coerce")
    d["Attempt_end_ms"] = pd.to_numeric(d["End Time (ms)"], errors="coerce")
    d["Attempt_duration_ms"] = d["Attempt_end_ms"] - d["Attempt_start_ms"]
    d["FTIS"] = d["FTIS"].astype(str).str.strip().str.upper()
    d["Success"] = d["FTIS"].map({"Y": 1, "N": 0})

    # Time is the within-participant chronological insertion sequence.
    d["Attempt_sequence"] = pd.to_numeric(d["Time"], errors="coerce")
    d["Round"] = pd.to_numeric(d["Round"], errors="coerce")

    # A unique row key remains unique even when a device-repetition label was repeated.
    d["Attempt_ID"] = [
        f"{p}_T{int(t) if pd.notna(t) else 'NA'}_{dev}_{i+1}"
        for i, (p, t, dev) in enumerate(
            zip(d["Participant"], d["Attempt_sequence"], d["Device_code_clean"])
        )
    ]

    # Attempt timestamps restart whenever Tobii recording was restarted.
    # Split each participant's chronologically ordered attempts into recording
    # sessions whenever the next attempt starts before the previous one ended.
    d = d.sort_values(["Participant", "Attempt_sequence"], kind="stable").copy()
    session_values = pd.Series(index=d.index, dtype="Int64")
    for participant, group in d.groupby("Participant", sort=False):
        session = 1
        previous_end = None
        for idx, row in group.iterrows():
            start = row["Attempt_start_ms"]
            end = row["Attempt_end_ms"]
            if previous_end is not None and start < previous_end:
                session += 1
            session_values.loc[idx] = session
            previous_end = end
    d["Session_index"] = session_values.astype(int)

    invalid = d[
        d["Participant"].eq("")
        | d["Attempt_start_ms"].isna()
        | d["Attempt_end_ms"].isna()
        | d["Attempt_end_ms"].le(d["Attempt_start_ms"])
        | d["Device_name"].isna()
    ]
    if not invalid.empty:
        invalid.to_csv(OUTPUT_FOLDER / "invalid_attempt_interval_rows.csv", index=False)
        raise ValueError(
            f"Found {len(invalid)} invalid attempt interval rows. "
            "See invalid_attempt_interval_rows.csv."
        )

    return d


def inspect_required_tsv_columns(file_path: Path) -> list[str]:
    header = pd.read_csv(file_path, sep="\t", nrows=0).columns.tolist()
    required = [
        PARTICIPANT_COL, TIMESTAMP_COL, SENSOR_COL,
        FIXATION_TYPE_COL, FIXATION_INDEX_COL,
        VALIDITY_LEFT_COL, VALIDITY_RIGHT_COL,
    ]
    optional = [RECORDING_COL, FIXATION_X_COL, FIXATION_Y_COL]
    missing = [c for c in required if c not in header]
    if missing:
        raise KeyError(f"Missing required Tobii columns: {missing}")
    return required + [c for c in optional if c in header]


def read_eye_tracker_rows(file_path: Path, usecols: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        usecols=usecols,
        chunksize=200_000,
        low_memory=False,
    ):
        chunk = chunk[chunk[SENSOR_COL].eq("Eye Tracker")]
        if not chunk.empty:
            pieces.append(chunk)
    if not pieces:
        return pd.DataFrame(columns=usecols)
    eye = pd.concat(pieces, ignore_index=True)
    eye[TIMESTAMP_COL] = pd.to_numeric(eye[TIMESTAMP_COL], errors="coerce")
    eye = eye.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL)
    return eye


def extract_unique_fixation_events(eye: pd.DataFrame, file_name: str, recording: str,
                                   participant: str) -> tuple[pd.DataFrame, float]:
    diffs = eye[TIMESTAMP_COL].diff()
    diffs = diffs[diffs.gt(0)]
    sample_interval_ms = float(diffs.median()) if not diffs.empty else np.nan

    fx = eye[
        eye[FIXATION_TYPE_COL].astype(str).str.casefold().eq("fixation")
        & eye[FIXATION_INDEX_COL].notna()
    ].copy()

    # Require actual mapped coordinates when the export supplies them.
    if FIXATION_X_COL in fx.columns and FIXATION_Y_COL in fx.columns:
        fx = fx[fx[FIXATION_X_COL].notna() & fx[FIXATION_Y_COL].notna()]

    if fx.empty:
        return pd.DataFrame(), sample_interval_ms

    events = (
        fx.groupby(FIXATION_INDEX_COL, sort=False)
        .agg(
            Fixation_start_ms=(TIMESTAMP_COL, "min"),
            Fixation_end_ms=(TIMESTAMP_COL, "max"),
            Eye_tracker_samples=(TIMESTAMP_COL, "size"),
        )
        .reset_index()
        .rename(columns={FIXATION_INDEX_COL: "Fixation_index"})
    )
    events["Fixation_duration_ms"] = (
        events["Fixation_end_ms"] - events["Fixation_start_ms"] + sample_interval_ms
    )
    events["Fixation_midpoint_ms"] = (
        events["Fixation_start_ms"] + events["Fixation_end_ms"]
    ) / 2
    events["Participant"] = participant
    events["File"] = file_name
    events["Recording"] = recording
    events["Sample_interval_ms"] = sample_interval_ms
    events["Duration_QC_flag"] = np.where(
        events["Fixation_duration_ms"].le(0)
        | events["Fixation_duration_ms"].gt(LONG_FIXATION_REVIEW_MS),
        "Review", "OK"
    )
    return events, sample_interval_ms


def calculate_attempt_metrics(
    eye: pd.DataFrame,
    events: pd.DataFrame,
    attempts: pd.DataFrame,
    file_name: str,
    recording: str,
    sample_interval_ms: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    valid_mask = at_least_one_eye_valid(
        eye[VALIDITY_LEFT_COL], eye[VALIDITY_RIGHT_COL]
    )
    eye = eye.assign(_valid=valid_mask)

    file_min = float(eye[TIMESTAMP_COL].min())
    file_max = float(eye[TIMESTAMP_COL].max())

    # Keep intervals overlapping this recording timeline.
    candidate_attempts = attempts[
        attempts["Attempt_end_ms"].ge(file_min)
        & attempts["Attempt_start_ms"].le(file_max)
    ].copy()

    for _, attempt in candidate_attempts.iterrows():
        start = float(attempt["Attempt_start_ms"])
        end = float(attempt["Attempt_end_ms"])

        attempt_eye = eye[eye[TIMESTAMP_COL].between(start, end, inclusive="both")]
        attempt_events = events[
            events["Fixation_midpoint_ms"].between(start, end, inclusive="both")
        ] if not events.empty else events

        if attempt_eye.empty:
            continue

        durations = attempt_events["Fixation_duration_ms"] if not attempt_events.empty else pd.Series(dtype=float)
        q1 = float(durations.quantile(0.25)) if not durations.empty else np.nan
        q3 = float(durations.quantile(0.75)) if not durations.empty else np.nan
        interval_duration = end - start
        valid_prop = float(attempt_eye["_valid"].mean())

        row = attempt.to_dict()
        row.update({
            "File": file_name,
            "Recording": recording,
            "Recording_min_timestamp_ms": file_min,
            "Recording_max_timestamp_ms": file_max,
            "Eye_sample_count": int(len(attempt_eye)),
            "Valid_eye_sample_count": int(attempt_eye["_valid"].sum()),
            "Valid_sample_proportion": valid_prop,
            "Fixation_count": int(len(attempt_events)),
            "Mean_fixation_duration_ms": float(durations.mean()) if not durations.empty else np.nan,
            "Median_fixation_duration_ms": float(durations.median()) if not durations.empty else np.nan,
            "SD_fixation_duration_ms": float(durations.std(ddof=1)) if len(durations) > 1 else np.nan,
            "Q1_fixation_duration_ms": q1,
            "Q3_fixation_duration_ms": q3,
            "IQR_fixation_duration_ms": q3 - q1 if np.isfinite(q1) and np.isfinite(q3) else np.nan,
            "Total_fixation_duration_ms": float(durations.sum()) if not durations.empty else 0.0,
            "Max_fixation_duration_ms": float(durations.max()) if not durations.empty else np.nan,
            "Fixation_rate_per_second": len(attempt_events) / (interval_duration / 1000),
            "Fixation_time_proportion": float(durations.sum()) / interval_duration if interval_duration > 0 else np.nan,
            "Long_fixations_review": int(
                attempt_events["Duration_QC_flag"].eq("Review").sum()
            ) if not attempt_events.empty else 0,
            "Sample_interval_ms": sample_interval_ms,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def recording_number(value: object) -> int:
    match = re.search(r"(\d+)\s*$", str(value))
    return int(match.group(1)) if match else 10**9


def assign_sessions_to_recordings(
    metrics: pd.DataFrame, attempts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Match timestamp-reset sessions to Tobii recordings in chronological order.

    A participant can have several TSV recordings whose timestamp ranges overlap
    numerically because each recording restarts at zero. Selecting the row with
    the most samples can therefore assign an attempt to the wrong recording.
    This function maps each insertion session to one recording, preserving file
    order and maximizing the number of covered attempt intervals. Extra partial
    or duplicate recordings may be skipped.
    """
    if metrics.empty:
        return metrics, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    duplicate_counts = metrics.groupby("Attempt_ID").size()
    duplicate_ids = duplicate_counts[duplicate_counts.gt(1)].index
    duplicates = metrics[metrics["Attempt_ID"].isin(duplicate_ids)].copy()

    chosen_rows: list[pd.DataFrame] = []
    map_rows: list[dict] = []
    missing_rows: list[dict] = []

    for participant, participant_attempts in attempts.groupby("Participant", sort=False):
        participant_attempts = participant_attempts.sort_values(
            ["Session_index", "Attempt_sequence"], kind="stable"
        )
        participant_metrics = metrics[metrics["Participant"].eq(participant)].copy()
        recordings = sorted(
            participant_metrics["Recording"].dropna().unique().tolist(),
            key=recording_number,
        )
        sessions = sorted(participant_attempts["Session_index"].unique().tolist())

        if not recordings:
            for _, attempt in participant_attempts.iterrows():
                missing_rows.append({
                    "Participant": participant,
                    "Session_index": int(attempt["Session_index"]),
                    "Attempt_ID": attempt["Attempt_ID"],
                    "Reason": "No Tobii recording found for participant",
                })
            continue

        attempt_ids_by_session = {
            session: participant_attempts.loc[
                participant_attempts["Session_index"].eq(session), "Attempt_ID"
            ].tolist()
            for session in sessions
        }

        # Dynamic programming: preserve chronological order, but allow extra
        # partial/duplicate recordings to be skipped.
        from functools import lru_cache

        @lru_cache(None)
        def best(i: int, j: int) -> tuple[int, tuple[str, ...]]:
            if i == len(sessions):
                return 0, tuple()
            if j == len(recordings):
                return -10**9, tuple()

            skip_score, skip_assignment = best(i, j + 1)

            session = sessions[i]
            recording = recordings[j]
            available_ids = set(
                participant_metrics.loc[
                    participant_metrics["Recording"].eq(recording), "Attempt_ID"
                ]
            )
            coverage = sum(
                attempt_id in available_ids
                for attempt_id in attempt_ids_by_session[session]
            )
            next_score, next_assignment = best(i + 1, j + 1)
            assign_score = coverage + next_score

            if assign_score > skip_score:
                return assign_score, (recording,) + next_assignment
            return skip_score, skip_assignment

        total_score, assignment = best(0, 0)
        if len(assignment) != len(sessions):
            raise RuntimeError(
                f"Could not map all sessions to recordings for {participant}. "
                f"Sessions={sessions}; recordings={recordings}"
            )

        assigned_recordings = set(assignment)
        for session, recording in zip(sessions, assignment):
            expected_ids = attempt_ids_by_session[session]
            candidates = participant_metrics[
                participant_metrics["Recording"].eq(recording)
                & participant_metrics["Attempt_ID"].isin(expected_ids)
            ].copy()
            recovered_ids = set(candidates["Attempt_ID"])

            # There should be at most one row per attempt in a recording.
            candidates = candidates.sort_values(
                ["Attempt_ID", "Eye_sample_count", "Valid_sample_proportion"],
                ascending=[True, False, False],
            ).drop_duplicates("Attempt_ID", keep="first")
            chosen_rows.append(candidates)

            map_rows.append({
                "Participant": participant,
                "Session_index": int(session),
                "Recording": recording,
                "Expected_attempts": len(expected_ids),
                "Recovered_attempts": len(recovered_ids),
                "Coverage_complete": len(recovered_ids) == len(expected_ids),
            })

            for attempt_id in expected_ids:
                if attempt_id not in recovered_ids:
                    missing_rows.append({
                        "Participant": participant,
                        "Session_index": int(session),
                        "Recording": recording,
                        "Attempt_ID": attempt_id,
                        "Reason": "Assigned recording did not contain Eye Tracker samples for interval",
                    })

        for recording in recordings:
            if recording not in assigned_recordings:
                map_rows.append({
                    "Participant": participant,
                    "Session_index": np.nan,
                    "Recording": recording,
                    "Expected_attempts": 0,
                    "Recovered_attempts": 0,
                    "Coverage_complete": np.nan,
                    "Note": "Recording skipped as partial or duplicate export",
                })

    selected = (
        pd.concat(chosen_rows, ignore_index=True)
        if chosen_rows else pd.DataFrame(columns=metrics.columns)
    )
    selected = selected.sort_values(
        ["Participant", "Attempt_sequence"], kind="stable"
    ).reset_index(drop=True)

    return (
        selected,
        duplicates,
        pd.DataFrame(map_rows),
        pd.DataFrame(missing_rows),
    )


def make_descriptives(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["Usable_mapped_fixation"] = d["Fixation_count"].gt(0)

    totals = (
        d.groupby(["Expertise", "Device_name"], dropna=False)
        .agg(
            Participants=("Participant", "nunique"),
            Total_attempts=("Attempt_ID", "size"),
            Usable_attempts=("Usable_mapped_fixation", "sum"),
        )
        .reset_index()
    )

    usable = d[d["Usable_mapped_fixation"]].copy()
    metrics = (
        usable.groupby(["Expertise", "Device_name"], dropna=False)
        .agg(
            Mean_of_attempt_means_ms=("Mean_fixation_duration_ms", "mean"),
            SD_of_attempt_means_ms=("Mean_fixation_duration_ms", "std"),
            Median_of_attempt_means_ms=("Mean_fixation_duration_ms", "median"),
            Q1_of_attempt_means_ms=("Mean_fixation_duration_ms", lambda x: x.quantile(0.25)),
            Q3_of_attempt_means_ms=("Mean_fixation_duration_ms", lambda x: x.quantile(0.75)),
            Median_fixation_count=("Fixation_count", "median"),
            Q1_fixation_count=("Fixation_count", lambda x: x.quantile(0.25)),
            Q3_fixation_count=("Fixation_count", lambda x: x.quantile(0.75)),
            Median_fixation_rate_per_second=("Fixation_rate_per_second", "median"),
            Median_valid_sample_proportion=("Valid_sample_proportion", "median"),
        )
        .reset_index()
    )
    return totals.merge(metrics, on=["Expertise", "Device_name"], how="left")


def fit_repeated_measures_models(d: pd.DataFrame) -> None:
    try:
        import statsmodels.formula.api as smf
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import NegativeBinomial
    except ImportError:
        LOGGER.warning("statsmodels not installed; extraction completed but models were skipped.")
        return

    model = d[
        (~d["Abort_participant"])
        & d["Device_name"].isin(["MST", "ATG", "CON"])
        & d["Expertise"].notna()
        & d["Mean_fixation_duration_ms"].gt(0)
        & d["Attempt_duration_ms"].gt(0)
    ].copy()

    model["log_mean_fixation_duration"] = np.log(model["Mean_fixation_duration_ms"])
    model["log_attempt_seconds"] = np.log(model["Attempt_duration_ms"] / 1000)
    model.to_csv(OUTPUT_FOLDER / "model_dataset_attempt_level.csv", index=False)

    if model["Participant"].nunique() < 3:
        LOGGER.warning("Too few participants to fit repeated-measures models.")
        return

    duration_formula = (
        "log_mean_fixation_duration ~ C(Device_name) + C(Expertise) "
        "+ Repetition + Attempt_sequence"
    )
    try:
        m = smf.mixedlm(
            duration_formula,
            data=model,
            groups=model["Participant"],
            re_formula="1",
        ).fit(method="lbfgs", reml=False)
        with open(OUTPUT_FOLDER / "mixed_model_log_mean_fixation_duration.txt", "w", encoding="utf-8") as f:
            f.write(m.summary().as_text())
            f.write("\n\nExponentiated fixed effects (geometric mean ratios):\n")
            ci = m.conf_int()
            names = [n for n in m.params.index if n != "Group Var"]
            out = pd.DataFrame({
                "Term": names,
                "Ratio": np.exp(m.params.loc[names].values),
                "CI_low": np.exp(ci.loc[names, 0].values),
                "CI_high": np.exp(ci.loc[names, 1].values),
            })
            f.write(out.to_string(index=False))
    except Exception as exc:
        LOGGER.exception("Duration mixed model failed: %s", exc)

    count_formula = (
        "Fixation_count ~ C(Device_name) + C(Expertise) "
        "+ Repetition + Attempt_sequence"
    )
    try:
        g = smf.gee(
            count_formula,
            groups="Participant",
            data=model,
            family=NegativeBinomial(),
            cov_struct=Exchangeable(),
            offset=model["log_attempt_seconds"],
        ).fit()
        with open(OUTPUT_FOLDER / "gee_negative_binomial_fixation_count.txt", "w", encoding="utf-8") as f:
            f.write(g.summary().as_text())
            f.write("\n\nExponentiated coefficients (incidence-rate ratios):\n")
            ci = g.conf_int()
            out = pd.DataFrame({
                "Term": g.params.index,
                "IRR": np.exp(g.params.values),
                "CI_low": np.exp(ci.iloc[:, 0].values),
                "CI_high": np.exp(ci.iloc[:, 1].values),
            })
            f.write(out.to_string(index=False))
    except Exception as exc:
        LOGGER.exception("Count GEE failed: %s", exc)

    r_code = r'''# Confirmatory models recommended for the manuscript
library(lme4)
library(glmmTMB)
library(emmeans)

D <- read.csv("model_dataset_attempt_level.csv")
D$Device_name <- relevel(factor(D$Device_name), ref = "MST")
D$Expertise <- relevel(factor(D$Expertise), ref = "Novice")

m_duration <- lmer(
  log(Mean_fixation_duration_ms) ~ Device_name + Expertise +
    Repetition + Attempt_sequence + (1 | Participant),
  data = D, REML = FALSE
)
summary(m_duration)
emmeans(m_duration, pairwise ~ Device_name, adjust = "tukey", type = "response")
emmeans(m_duration, pairwise ~ Expertise, adjust = "tukey", type = "response")

m_count <- glmmTMB(
  Fixation_count ~ Device_name + Expertise + Repetition + Attempt_sequence +
    offset(log(Attempt_duration_ms / 1000)) + (1 | Participant),
  family = nbinom2, data = D
)
summary(m_count)
emmeans(m_count, pairwise ~ Device_name, adjust = "tukey", type = "response")
'''
    (OUTPUT_FOLDER / "recommended_confirmatory_models.R").write_text(r_code, encoding="utf-8")


# -----------------------------------------------------------------------------
# 3. MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    attempts = load_attempt_intervals(INTERVAL_FILE)
    attempts.to_csv(OUTPUT_FOLDER / "attempt_intervals_cleaned.csv", index=False)

    files = sorted(INPUT_FOLDER.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No TSV files found in {INPUT_FOLDER}")

    LOGGER.info("Found %d TSV files", len(files))
    all_events: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    qc_rows: list[dict] = []

    for i, file_path in enumerate(files, start=1):
        LOGGER.info("[%d/%d] Processing %s", i, len(files), file_path.name)
        try:
            usecols = inspect_required_tsv_columns(file_path)
            eye = read_eye_tracker_rows(file_path, usecols)
            if eye.empty:
                qc_rows.append({"File": file_path.name, "Status": "No Eye Tracker rows"})
                continue

            participant_values = eye[PARTICIPANT_COL].dropna().map(normalize_participant)
            if participant_values.empty:
                raise ValueError("No participant identifier found in Eye Tracker rows")
            participant = participant_values.mode().iloc[0]
            recording = (
                str(eye[RECORDING_COL].dropna().iloc[0])
                if RECORDING_COL in eye.columns and eye[RECORDING_COL].notna().any()
                else file_path.stem
            )

            participant_attempts = attempts[attempts["Participant"].eq(participant)].copy()
            if participant_attempts.empty:
                raise ValueError(f"No insertion.xlsx rows found for {participant}")

            events, sample_interval = extract_unique_fixation_events(
                eye, file_path.name, recording, participant
            )
            metrics = calculate_attempt_metrics(
                eye, events, participant_attempts, file_path.name,
                recording, sample_interval
            )

            if not events.empty:
                all_events.append(events)
            if not metrics.empty:
                all_metrics.append(metrics)

            qc_rows.append({
                "File": file_path.name,
                "Participant": participant,
                "Recording": recording,
                "Eye_tracker_rows": len(eye),
                "Unique_US_fixations": len(events),
                "Attempt_rows_created": len(metrics),
                "Recording_start_ms": eye[TIMESTAMP_COL].min(),
                "Recording_end_ms": eye[TIMESTAMP_COL].max(),
                "Median_sample_interval_ms": sample_interval,
                "Status": "OK",
            })
        except Exception as exc:
            LOGGER.exception("Failed to process %s", file_path.name)
            qc_rows.append({"File": file_path.name, "Status": f"ERROR: {exc}"})

    pd.DataFrame(qc_rows).to_csv(OUTPUT_FOLDER / "file_level_QC.csv", index=False)

    if not all_metrics:
        raise RuntimeError(
            "No attempt-level metrics were created. Check insertion.xlsx paths, "
            "participant IDs, and timestamp alignment in file_level_QC.csv."
        )

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    metrics_all_files = pd.concat(all_metrics, ignore_index=True)
    metrics, duplicate_rows, session_map, session_missing = assign_sessions_to_recordings(
        metrics_all_files, attempts
    )

    if not events.empty:
        events.to_csv(OUTPUT_FOLDER / "fixation_events_unique_all_recordings.csv", index=False)
    metrics_all_files.to_csv(OUTPUT_FOLDER / "attempt_metrics_before_duplicate_resolution.csv", index=False)
    metrics.to_csv(OUTPUT_FOLDER / "attempt_level_fixation_metrics_all.csv", index=False)
    duplicate_rows.to_csv(OUTPUT_FOLDER / "duplicate_attempt_rows_review.csv", index=False)
    session_map.to_csv(OUTPUT_FOLDER / "session_recording_map.csv", index=False)
    session_missing.to_csv(OUTPUT_FOLDER / "session_assignment_missing_attempts.csv", index=False)

    expected = set(attempts["Attempt_ID"])
    observed = set(metrics["Attempt_ID"])
    missing = attempts[attempts["Attempt_ID"].isin(expected - observed)].copy()
    missing.to_csv(OUTPUT_FOLDER / "attempts_without_eye_tracking_metrics.csv", index=False)

    primary = metrics[~metrics["Abort_participant"]].copy()
    primary.to_csv(OUTPUT_FOLDER / "attempt_level_fixation_metrics_primary_nonabort.csv", index=False)
    zero_mapped = primary[primary["Fixation_count"].eq(0)].copy()
    zero_mapped.to_csv(OUTPUT_FOLDER / "attempts_with_zero_mapped_fixations.csv", index=False)
    make_descriptives(primary).to_csv(
        OUTPUT_FOLDER / "descriptives_by_expertise_device.csv", index=False
    )

    fit_repeated_measures_models(metrics)

    LOGGER.info("Completed. Results written to %s", OUTPUT_FOLDER)
    LOGGER.info("Attempt rows after duplicate resolution: %d", len(metrics))
    LOGGER.info("Primary non-abort attempt rows: %d", len(primary))
    LOGGER.info("Attempts without Eye Tracker rows after session mapping: %d", len(missing))
    LOGGER.info("Primary attempts with zero mapped US fixation events: %d", len(zero_mapped))
    LOGGER.info(
        "These are whole-attempt ultrasound-mapped fixation metrics. "
        "Do not label them as needle-entry metrics unless separate phase intervals are supplied."
    )


if __name__ == "__main__":
    main()

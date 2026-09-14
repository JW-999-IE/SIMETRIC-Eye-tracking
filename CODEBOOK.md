# SIMETRIC Data Codebook

## model_dataset_attempt_level.csv

The primary analysis dataset: 403 gaze-valid, non-aborted attempts from 27 clinicians.

| Variable | Type | Description |
|----------|------|-------------|
| Attempt_ID | string | Unique identifier: `{Participant}_T{trial}_{DeviceCode}_{row}` |
| Participant | string | De-identified clinician code (P1–P30, excluding P7, P8, P14) |
| Device_name | string | Device: MST, ATG, or CON |
| Expertise | string | Prior experience: Expert, Intermediate, or Novice |
| Repetition | int | Within-device repetition (1–5) |
| Attempt_sequence | int | Chronological attempt number within session (1–15) |
| Success | int | Eventual procedural success within the attempt: 1=success, 0=failure. Requires vessel puncture, catheter advancement, simulated blood return, aspiration and flush, and ultrasound-confirmed intraluminal position. Multiple skin punctures are permitted and no time limit was imposed. **Not first-attempt success:** 10 of the 414 attempts record two or more needle-entry events, 7 of which are coded successful. The source column in the observation log is labelled `FTIS`, which is a misnomer for this variable. |
| Fixation_count | int | Number of unique ultrasound-mapped fixation events |
| Mean_fixation_duration_ms | float | Mean fixation duration in milliseconds |
| Median_fixation_duration_ms | float | Median fixation duration in milliseconds |
| SD_fixation_duration_ms | float | Standard deviation of fixation durations |
| Q1_fixation_duration_ms | float | 25th percentile fixation duration |
| Q3_fixation_duration_ms | float | 75th percentile fixation duration |
| IQR_fixation_duration_ms | float | Interquartile range of fixation durations |
| Total_fixation_duration_ms | float | Sum of all fixation durations |
| Max_fixation_duration_ms | float | Longest individual fixation |
| Fixation_rate_per_second | float | Fixation count / attempt duration in seconds |
| Fixation_time_proportion | float | Total fixation duration / attempt duration |
| Attempt_duration_ms | float | Total attempt duration in milliseconds |
| Valid_sample_proportion | float | Proportion of eye samples with valid gaze (0–1) |
| Needle In (ms) | float | Timestamp of needle insertion (ms from recording start) |
| Wire In (ms) | float | Timestamp of guidewire insertion |
| Needle Out (ms) | float | Timestamp of needle removal |
| Catheter Over Needle (ms) | float | Timestamp of catheter advance |
| Wire Out (ms) | float | Timestamp of guidewire removal |

## phase_analysis_mst_phases.csv

MST-only phase-level data: 590 rows from 124 attempts × up to 5 phases.

| Variable | Type | Description |
|----------|------|-------------|
| Attempt_ID | string | Links to model_dataset_attempt_level.csv |
| Participant | string | De-identified clinician code |
| Expertise | string | Prior experience stratum |
| Success | int | Attempt-level success, as defined above: eventual placement within the attempt, not first-pass |
| Phase | string | One of: Pre-needle, Needle-to-Wire, Wire-to-NeedleOut, NeedleOut-to-CON, CON-to-WireOut |
| Phase_duration_ms | float | Duration of this phase in milliseconds |
| Fixation_count | int | Fixations during this phase |
| Fixation_rate | float | Fixation count / phase duration in seconds |
| Mean_fixation_duration_ms | float | Mean fixation duration during this phase |
| Phase_fraction | float | Phase duration / total attempt duration |

Phase names use the corrected terminology:
- **NeedleOut-to-CON** = needle-out-to-catheter-advance
- **CON-to-WireOut** = catheter-advance-to-wire-out

> **Warning — `CON` means two different things in this dataset.** In `Phase`
> labels it abbreviates *catheter advance*, a step within the modified
> Seldinger technique. In `Device_name` it denotes the *catheter-over-needle
> configuration*, which is a separate device arm. These phases exist only for
> MST attempts; CON-configuration attempts have no phase records at all. Do
> not join or filter on the string `CON` across the two columns.

## phase_analysis_temporal_thirds.csv

Temporal tercile analysis: attempts divided into thirds (T1/T2/T3) of each participant's session timeline.

## attempts_with_zero_mapped_fixations.csv

QC file listing all 11 attempts with zero AOI-mapped fixations. Key fields match model_dataset_attempt_level.csv plus the original observation-log columns (Round, Time, FTIS, etc.). All have Valid_sample_proportion > 0.91, confirming the eye tracker was functional — the zero is due to failed scene-to-AOI mapping, not tracking failure.

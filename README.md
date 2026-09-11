# SIMETRIC Eye-Tracking Analysis

Analysis code and processed data for the SIMETRIC study: **Simulation-Based Training for Midline Catheter Insertion — Eye-Tracking and Procedural Outcomes**.

Submitted to *Behavior Research Methods*.

## Study Overview

SIMETRIC is a randomised crossover study in which 30 clinicians (27 retained after 3 discontinued) each performed 5 rounds of midline catheter insertions on a simulation model using three ultrasound-guidance devices (CON, ATG, MST) under Tobii Pro Glasses 3 eye tracking (100 Hz, I-VT fixation filter: velocity threshold 30°/s, minimum fixation duration 60 ms). The primary outcomes are ultrasound-screen fixation behaviour (count, duration, rate) and procedural success (first-attempt catheter placement), analysed with participant-clustered generalised estimating equations (GEE) and Bayesian sensitivity models.

## Repository Structure

```
SIMETRIC-eye-tracking-repository/
├── README.md                     # This file
├── CODEBOOK.md                   # Variable definitions for all datasets
├── requirements.txt              # Python dependencies
├── .gitignore
│
├── scripts/
│   ├── 01_extract_fixations.py           # Raw TSV → attempt-level fixation metrics
│   ├── 02_procedural_success_gee.py      # Logistic GEE for first-attempt success
│   ├── 03_confirmatory_fixation_models.py # NB-GEE (count) + LMM (duration)
│   ├── 04_saccade_analysis.py            # Saccade rate and amplitude GEE
│   ├── 05_fixation_duration_sensitivity.py # Median/trimmed-mean sensitivity
│   ├── 06_temporal_tercile_analysis.py   # Device × Tercile interaction NB-GEE
│   ├── 07_mst_phase_analysis.py          # Poisson GEE phase analysis + NB-GEE sensitivity
│   ├── 08_bayesian_sensitivity.py        # PyMC cell-means models (count, duration, success)
│   ├── 09_thirty_clinician_sensitivity.py # NB-GEE robustness with N=30
│   └── 10_generate_figures.py            # Figures 3–6 generation
│
├── data/
│   ├── raw/                              # NOT INCLUDED — see Data Availability
│   │   └── .gitkeep
│   ├── processed/
│   │   ├── model_dataset_attempt_level.csv        # Main analysis dataset (N=403)
│   │   ├── phase_analysis_mst_phases.csv          # MST phase-level data (N=124 attempts)
│   │   ├── phase_analysis_temporal_thirds.csv     # Temporal tercile analysis
│   │   └── attempts_with_zero_mapped_fixations.csv # QC: zero-gaze verification
│   └── intermediate/
│       ├── attempt_intervals_cleaned.csv
│       ├── file_level_QC.csv
│       ├── session_recording_map.csv
│       ├── fixation_events_unique_all_recordings.csv
│       └── attempts_without_eye_tracking_metrics.csv
│
├── results/
│   ├── fixation_count/                   # NB-GEE fixation count outputs
│   ├── fixation_duration/                # LMM fixation duration outputs
│   ├── procedural_success/               # Logistic GEE success outputs
│   ├── saccade/                          # Saccade rate/amplitude outputs
│   ├── sensitivity/                      # Duration sensitivity checks
│   ├── temporal_tercile/                 # Temporal tercile interaction analysis
│   ├── mst_phase/                        # MST Seldinger phase analysis
│   ├── bayesian/                         # Bayesian sensitivity posteriors
│   └── thirty_clinician_sensitivity/     # N=30 robustness check
│
└── figures/
    ├── fig_flow_diagram.png              # Figure 3: Analysis flow
    ├── fig_bayesian_forest.png           # Figure 4: Bayesian forest plot
    ├── fig_temporal_tercile.png          # Figure 5: Temporal tercile distributions
    └── fig_mst_phases.png               # Figure 6: MST phase analysis
```

## Analysis Pipeline

The pipeline runs sequentially. Each script reads from the `data/` directory and writes results to `results/`.

### Step 1: Fixation Extraction (`01_extract_fixations.py`)

Reads 59 raw Tobii Pro Lab TSV exports and the observation log (`insertion.xlsx`) to:
- Parse attempt intervals from the observation log
- Extract unique ultrasound-mapped fixation events per attempt
- Compute attempt-level metrics: fixation count, mean/median duration, fixation rate, valid sample proportion
- Identify and flag: multi-puncture attempts, zero-mapped-fixation attempts, duplicate recordings, mapping failures
- Output: `model_dataset_attempt_level.csv` (403 gaze-valid attempts) and QC audit files

**Exclusion flow (whole-attempt analysis):**
- 438 total attempts across 30 clinicians × 3 devices × 5 rounds
- −24 from 3 discontinued participants → 414 primary attempts
- −11 mapping failures (no AOI-mapped gaze) → 403 gaze-valid attempts

**Exclusion flow (MST phase analysis):**
- 137 MST attempts total
- −10 multi-puncture (>1 needle insertion) → 127
- −3 zero-gaze mapping failures (all from P2, Recording 16) → 124 phase-analysed attempts
- Of these, 107 have all 5 phases; 17 have fewer

### Step 2: Procedural Success GEE (`02_procedural_success_gee.py`)

Logistic GEE with exchangeable working correlation, clustered by participant:
- Outcome: first-attempt success (binary)
- Predictors: device, experience stratum, repetition, attempt sequence
- Contrasts: pairwise device comparisons with Holm correction
- Both three-level and binary experience specifications

### Step 3: Confirmatory Fixation Models (`03_confirmatory_fixation_models.py`)

Two model families for the 403 gaze-valid attempts:

**Fixation count:** Negative-binomial GEE with log(attempt duration) offset, clustered by participant. Reports incidence rate ratios (IRR).

**Mean fixation duration:** Linear mixed-effects model on log-transformed duration with participant random intercept.

Both models: device + experience + repetition + attempt sequence. Pairwise contrasts with Holm correction.

### Step 4: Saccade Analysis (`04_saccade_analysis.py`)

- Saccade rate: NB-GEE with log(duration) offset
- Mean saccadic amplitude: Gaussian GEE on log-transformed values
- Same predictor structure as Step 3

### Step 5: Duration Sensitivity (`05_fixation_duration_sensitivity.py`)

Two robustness checks for fixation duration:
1. Attempt-level **median** fixation duration (LMM, log-transformed)
2. **Trimmed mean** excluding individual fixation events >10 s (LMM, log-transformed)

### Step 6: Temporal Tercile Analysis (`06_temporal_tercile_analysis.py`)

NB-GEE with Device × Tercile interaction, participant clustering:
- Each attempt divided into three equal-duration temporal terciles (T1, T2, T3)
- Tests whether fixation-rate decline across attempt phases differs by device
- Reports marginal tercile contrasts and device-specific tercile contrasts
- Main-effects (no interaction) model for comparison

### Step 7: MST Phase Analysis (`07_mst_phase_analysis.py`)

Phase-level fixation-rate analysis across the five Seldinger phases of MST insertion:
- **Primary:** Poisson GEE with clinician clustering, Pre-needle as reference
- **Sensitivity:** NB-GEE with identical specification
- 10 pairwise phase contrasts with Holm correction

### Step 8: Bayesian Sensitivity (`08_bayesian_sensitivity.py`)

Three PyMC cell-means models with participant random intercepts:
1. **Fixation count:** negative-binomial, log(attempt seconds) offset
2. **Fixation duration:** log-normal on seconds
3. **Procedural success:** Bernoulli (logit-link)

Priors: Normal(0, 2) cell means, HalfNormal(0.5) participant SD, HalfNormal(1) residual SD.
Sampling: 4 chains × 2000 post-warmup draws, 1000 warmup, target_accept = 0.95.

### Step 9: 30-Clinician Sensitivity (`09_thirty_clinician_sensitivity.py`)

Robustness check: refit NB-GEE fixation-count model on all 30 recruited clinicians (including 3 discontinued: P7, P8, P14). All 24 discontinued-clinician attempts coded as unsuccessful.

### Step 10: Figure Generation (`10_generate_figures.py`)

Generates Figures 3–6:
- **Figure 3:** Analysis flow diagram (participant/attempt exclusion)
- **Figure 4:** Bayesian posterior forest plot (fixation count IRR)
- **Figure 5:** Temporal tercile fixation distributions by device
- **Figure 6:** MST Seldinger phase fixation-rate profiles

## Key Technical Details

### Eye-Tracking Hardware and Software
- **Device:** Tobii Pro Glasses 3 (100 Hz binocular)
- **Software:** Tobii Pro Lab v1.183
- **Fixation filter:** I-VT (Attention), velocity threshold 30°/s, minimum fixation duration 60 ms
- **AOI mapping:** Assisted mapping to two reference surfaces (ultrasound screen, simulator screen) via Tobii Pro Lab's surface-mapping feature

### Clock Synchronisation
Procedure timestamps (needle-in, wire-in, etc.) were recorded within Tobii Pro Lab's own timeline using the software's event-marking interface. Both the eye-tracking data stream (`Recording timestamp`) and the procedure events share a common zero (recording start), eliminating the need for external clock synchronisation. The `Computer timestamp` column in the TSV exports provides the concurrent laptop clock for audit purposes, with millisecond-level alignment verified via the recording start time metadata (e.g., `11:32:52.679`).

### Device Coding
| Code | Device | Colour |
|------|--------|--------|
| DEVICE1 / MST | Micro-Short-over-the-Needle Technique | #4CAF50 (green) |
| DEVICE2 / ATG | Accelerated Technique with Guide | #2196F3 (blue) |
| DEVICE3 / CON | Conventional over-the-needle | #FF9800 (orange) |

### Zero-Gaze Verification
The 3 MST attempts excluded as "zero gaze" (P2_T6_DEVICE1-2, P2_T11_DEVICE1-4, P2_T13_DEVICE1-5) are from P2's Recording 16, in which **all 8 attempts across all 3 devices** had zero AOI-mapped fixations despite high valid-sample proportions (>91%). This pattern indicates a session-wide scene-to-AOI mapping failure, not a behavioural choice. The eye tracker was functioning normally; the gaze-to-surface mapping was unsuccessful for this entire recording session. See `data/processed/attempts_with_zero_mapped_fixations.csv` for the full list of 11 affected attempts (8 from P2 Recording 16, 3 from P27 Recording 71).

### Statistical Models
- **GEE:** Participant-clustered with exchangeable working correlation, bias-reduced covariance (statsmodels)
- **Bayesian:** Cell-means parameterisation (one parameter per device level), participant random intercepts, weakly informative priors. Posteriors described with HDI-based language, not frequentist "significant/non-significant".
- **MST phase analysis:** Poisson GEE with clinician clustering, Pre-needle as reference. Holm correction across 10 pairwise comparisons. NB-GEE sensitivity check.
- **Temporal tercile:** NB-GEE with Device × Tercile interaction to test differential fixation-rate decline across attempt phases.

## Data Availability

**Raw data** (59 Tobii TSV exports, ~1–50 MB each; `insertion.xlsx` observation log) are not included in this repository due to size and participant-privacy considerations. They are available from the corresponding author upon reasonable request, subject to ethics-board approval.

**Processed datasets** (`data/processed/`) contain de-identified attempt-level summaries and are included.

## Requirements

```
Python ≥ 3.9
numpy, pandas, scipy, statsmodels, openpyxl
matplotlib, seaborn                          # figures
pymc ≥ 5.0, arviz ≥ 0.15                    # Bayesian models (Step 8 only)
```

Install: `pip install -r requirements.txt`

**Note:** PyMC and ArviZ are only required for `08_bayesian_sensitivity.py`. All other scripts run with the core statistical stack (numpy, pandas, scipy, statsmodels).

## Reproduction

1. Place raw TSV files and `insertion.xlsx` in `data/raw/`
2. Run scripts in order: `python scripts/01_extract_fixations.py` through `10_generate_figures.py`
3. Each script auto-detects input files and writes results to `results/`

Scripts 06–10 depend only on processed data files in `data/processed/` and can be run independently of Steps 1–5.

## License

[To be determined — consult institutional policy]

## Citation

[Manuscript reference to be added upon acceptance]

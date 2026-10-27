# SIMETRIC — Eye tracking, procedural success and attempt duration

Analysis code and de-identified processed data for the eye-tracking, procedural-success and
attempt-duration components of the SIMETRIC study (Simulation and Imaging Methods using Eye
Tracking for the Recording of Peripheral Intravenous Catheter Insertion).

Companion repositories:
[hand motion and grip](https://github.com/JW-999-IE/SIMETRIC-Hand-motion) ·
[surveys](https://github.com/JW-999-IE/SIMETRIC-Surveys-Pre-and-Post-task)

Manuscript: *Ultrasound-guided long peripheral catheter configurations differ in procedural
performance, gaze-event rates and hand kinematics in the SIMETRIC simulation study*
(submitted to *Scientific Reports*).

## Study design

SIMETRIC was a repeated-measures crossover simulation study. Thirty clinicians enrolled and 27
completed the protocol. Each completing clinician was scheduled to perform five ultrasound-guided
insertions with each of three 8-cm long peripheral catheter (LPC) configurations on a vascular-access
phantom (Kyoto Kagaku) using a Philips EPIQ Elite ultrasound system. Insertion orders were prepared
before data collection as counterbalanced block sequences; they were not independently randomised
for each clinician.

| Code in manuscript | Configuration | Legacy codes in some data files |
|---|---|---|
| CON | catheter-over-needle | DEVICE3 |
| ATG | accelerated technique with guidewire | DEVICE2 |
| DST | direct Seldinger technique (needle, separate guidewire, catheter; no dilator or sheath) | DEVICE1, MST |

Files produced before the final terminology was adopted label DST as `MST`. The two codes refer
to the same configuration.

Analysis populations: 414 non-aborted attempts from 27 clinicians (CON 136, ATG 141, DST 137;
405 scheduled plus 9 participant-initiated attempts). Eleven attempts had no recoverable
gaze-to-ultrasound mapping (8 from P2 Recording 16, 3 from P27 Recording 71), leaving 403
gaze-valid attempts. The 24 unsuccessful attempts of the three clinicians who discontinued are
excluded from the 414-attempt analyses.

## Outcomes

- **Eventual procedural success**: catheter placed in the phantom vessel with blood return,
  aspiration and flush, and ultrasound confirmation. Multiple punctures were permitted, so this is
  not first-puncture success.
- **Attempt duration**: from ultrasound probe positioning to catheter removal after the success
  check (successful attempts) or withdrawal/abandonment (unsuccessful attempts).
- **Ultrasound-directed gaze events**: fixation and saccade events mapped to the ultrasound display,
  analysed as counts and as rates per attempt-second.

## Eye-tracking acquisition and event classification

- Tobii Pro Glasses 3, **50 Hz** gaze sampling.
- Tobii Pro Lab 25.23.1545, assisted real-world mapping to the ultrasound display.
- I-VT (Attention) filter: **100°/s** velocity threshold, 20-ms velocity window, 3-sample median
  noise filter, no gap filling, merging of adjacent fixations within 75 ms and 0.5°, fixations
  shorter than 60 ms discarded.

At this threshold the software-defined "fixation" class includes periods of sustained attention
with head and eye movement, so it should not be equated with conventional stationary fixations.
Event rates are not a measure of cognitive load.

## Statistical models (as reported in the manuscript)

- Eventual success: binomial-logit GEE, clinician clusters, exchangeable working correlation,
  bias-reduced covariance; covariates configuration, prior LPC experience (none, 1–50, >50),
  repetition and chronological attempt sequence. Pairwise contrasts Holm-adjusted.
- Attempt duration: Gaussian GEE on log seconds, same structure; geometric mean ratios.
- Gaze-event counts: negative-binomial GEE with log attempt duration as offset; incidence rate ratios.
- Sensitivity analyses: scheduled attempts only (first five attempts per clinician–configuration
  cell, 405 attempts), leave-one-clinician-out refits, and inclusion of all 30 recruited clinicians
  (438 attempts) for procedural success.

Key results reproduced by `scripts/07_procedural_success_and_duration.py`:

| Comparison | All attempts (414) | Scheduled only (405) |
|---|---|---|
| Success, CON vs ATG | OR 2.00 (1.10–3.66), Holm P = 0.048 | OR 1.97 (1.05–3.68), Holm P = 0.067 |
| Success, DST vs ATG | OR 2.55 (1.37–4.73), Holm P = 0.009 | OR 2.53 (1.34–4.78), Holm P = 0.012 |
| Duration, ATG vs CON | GMR 1.17 (1.05–1.29), P = 0.004 | — |
| Duration, DST vs CON | GMR 1.90 (1.65–2.19), P < 0.001 | — |

The CON vs ATG success contrast was Holm-significant in 11 of 27 leave-one-clinician-out refits;
DST vs ATG was significant in all 27.

## Repository structure

```
├── README.md
├── CODEBOOK.md
├── requirements.txt
├── scripts/
│   ├── 01_extract_fixations.py               # raw Tobii exports → attempt-level gaze metrics
│   ├── 02_procedural_success_gee.py          # earlier success model script
│   ├── 03_confirmatory_fixation_models.py    # negative-binomial GEE for fixation-event rate
│   ├── 04_saccade_analysis.py                # saccade-event rate
│   ├── 05_fixation_duration_sensitivity.py   # supplementary checks
│   └── 07_procedural_success_and_duration.py # success, duration, scheduled-only, leave-one-out
├── data/
│   ├── processed/
│   │   ├── model_dataset_attempt_level.csv   # 403 gaze-valid attempts
│   │   └── procedural_attempts_414.csv       # 414 attempts: configuration, experience, success, duration
│   └── intermediate/                         # QC and mapping files
├── results/                                  # model outputs
```

Some scripts and outputs (fixation duration, pupil, Bayesian and phase analyses) were produced for
an earlier analysis of the same recordings and are not reported in the *Scientific Reports*
manuscript. They are retained for transparency.

## Data availability

Raw Tobii recordings and scene videos contain participant data and are not deposited. They may
be requested from the corresponding author, subject to institutional and ethical approval.
Processed attempt-level data in `data/processed/` contain pseudonymous participant codes only.

## Ethics

Approved by the Clinical Research Ethics Committee at the University of Galway (C.A. 3392) before
recruitment. Written informed consent was obtained from all participants.

## Requirements

Python ≥ 3.9 with numpy, pandas, scipy, statsmodels and openpyxl (see `requirements.txt`).

## Citation

Manuscript reference to be added on acceptance.

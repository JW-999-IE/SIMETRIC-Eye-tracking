# SIMETRIC Eye-Tracking Analysis

Analysis code and processed data for the SIMETRIC study: **Simulation and Imaging Methods for
Eye Tracking and Recording Intravenous Catheter insertion**.

## Study Overview

SIMETRIC is a counterbalanced, repeated-measures simulation study in which 30 clinicians
(27 retained after 3 discontinued) each performed five insertions with each of three
ultrasound-guided **long peripheral catheter (LPC)** configurations — catheter-over-needle
(CON), accelerated technique with guidewire (ATG), and modified Seldinger technique (MST) —
on a vascular-access phantom, while wearing Tobii Pro Glasses 3.

Outcomes include procedural success and duration, ultrasound-display gaze behaviour, hand
kinematics, needle-entry grip posture and post-simulation survey measures. Primary inference
uses participant-clustered generalised estimating equations (GEE), with Bayesian mixed-effects
models as sensitivity analyses.

**Terminology.** The parent study protocol refers to these devices as midline catheters,
reflecting the nomenclature current when it was written. At 8 cm insertable length they fall
within the long peripheral catheter category under current definitions. **LPC is used
throughout this repository and in all associated manuscripts; MLC is not used.**

## Key Technical Details

### Devices

All three are LPCs of equivalent 8 cm insertable length, differing in insertion mechanism:

| Code | Configuration | Description |
|---|---|---|
| DEVICE1 / MST | Modified Seldinger technique | 2 Fr Seldipur Smartmidline: polyurethane catheter with integrated extension line and rounded tip, 21 G x 4.5 cm puncture needle, stainless-steel guidewire. Five phases: needle access, guidewire insertion, needle removal, catheter advancement, guidewire extraction. No dilator and no peel-away sheath were used. |
| DEVICE2 / ATG | Accelerated technique with guidewire | 22 G integrated catheter with a preloaded guidewire |
| DEVICE3 / CON | Catheter-over-needle | Catheter advanced directly over the introducer needle; no wire or introducer |

### Eye-tracking hardware and software

- **Device:** Tobii Pro Glasses 3, binocular
- **Analysed sampling:** 20 ms sample interval (50 Hz) throughout the exported data
- **Software:** Tobii Pro Lab 25.23.1545 (the application updates automatically; earlier
  processing steps may have used an earlier build)
- **Gaze filter:** Tobii **I-VT (Attention)**, confirmed in the `Recording Fixation filter name`
  column of all 59 exports. Full configuration:

| Setting | Value |
|---|---|
| Gap fill-in (interpolation) | Disabled (max gap length setting 75 ms) |
| Noise reduction | Moving median, window 3 samples |
| Velocity calculator window length | 20 ms |
| I-VT classifier threshold | **100 °/s** |
| Merge adjacent fixations | Enabled — max 75 ms, max 0.5° |
| Discard short fixations | Enabled — minimum fixation duration 60 ms |
| Reclassify discarded as saccade | Disabled |

At a 100 °/s threshold, slower gaze shifts, smooth pursuit and vestibulo-ocular compensation are
absorbed into the fixation class. Classified fixations are therefore best understood as periods
of sustained attention to the ultrasound display rather than classical foveal fixations.

- **AOI mapping:** assisted (semi-automatic) real-world mapping to the ultrasound display

### Event extraction rules

Ultrasound-mapped gaze events are identified as follows:

1. Retain rows where `Sensor == "Eye Tracker"`.
2. **Fixation event:** mapped type `fixation`, non-null mapped event index, **and** non-null
   `Mapped fixation X/Y [US]`.
3. **Saccade event:** mapped type `saccade` and non-null mapped event index. Saccades have no
   mapped coordinate, so the coordinate requirement does not apply.
4. Group samples sharing an event index into one event; event midpoint = (min + max timestamp) / 2.
5. Assign each event to the attempt whose interval contains its **midpoint**, inclusive at both
   ends, so every event counts once and events spanning consecutive attempts are not double-counted.

### Device coding

`DEVICE1 = MST`, `DEVICE2 = ATG`, `DEVICE3 = CON`, per the table above.

### Exclusion flow (whole-attempt analysis)

- 438 recorded attempt rows from 30 clinicians
- −24 attempts from 3 clinicians who discontinued → **414** non-aborted attempts
- −11 technical mapping failures → **403** gaze-valid attempts

Configuration counts across the 414: ATG 141, CON 136, MST 137. Of the 414, 405 are the
scheduled five-per-configuration attempts; nine are additional participant-initiated retries
(six ATG, one CON, two MST). A protocol-only sensitivity analysis restricted to the first five
observed repetitions per configuration yields exactly 135 per configuration.

### Zero-gaze verification

The 11 mapping failures comprise 8 attempts from P2 (Recording 16) and 3 from P27
(Recording 71). In the former, all 8 attempts across all 3 configurations returned zero mapped
fixations despite valid-sample proportions above 91%, indicating a session-wide scene-to-surface
mapping failure rather than absence of ultrasound viewing.

### Device-identity review

A review of 18 MST-labelled attempts proposed two reassignments (P6_T11 → ATG, P19_T03 → CON).
Both were **rejected** on adjudication against the session phase records: each carries a complete
Seldinger phase sequence, including guidewire-in and guidewire-out timestamps for P19_T03, which
is incompatible with the guidewire-free CON configuration. The original workbook identities are
retained for all 414 attempts.

### Statistical models

- **GEE:** participant-clustered, exchangeable working correlation, bias-reduced covariance
- **Counts:** negative binomial with log attempt-duration offset
- **Grip:** nominal (multinomial) GEE, global odds-ratio working structure, IPG reference
- **Software:** Python 3.14.5, statsmodels 0.14.6, scipy 1.17.1, numpy 2.4.6, pandas 3.0.3

## Data Availability

Raw data (59 Tobii TSV exports; `insertion.xlsx` observation log) are not included, for size and
participant-privacy reasons. Processed datasets in `data/processed/` contain de-identified
attempt-level summaries.

## Requirements

```
Python >= 3.9
numpy, pandas, scipy, statsmodels, openpyxl
matplotlib, seaborn        # figures
pymc >= 5.0, arviz >= 0.15 # Bayesian sensitivity models only
```

---

# CHANGELOG — corrections applied to this README

The previous version of this README contained the following errors. Each is corrected above.
Anyone who cloned the repository before this revision should re-read this section.

| # | Previous text | Corrected to | Why |
|---|---|---|---|
| 1 | "midline catheter insertion" throughout | long peripheral catheter (LPC) | The devices are 8 cm insertable length and are classified as LPCs. MLC is not used in any associated manuscript. |
| 2 | "Tobii Pro Glasses 3 (100 Hz)" | 20 ms sample interval, 50 Hz | Every row of the analysis dataset and the extracted event file carries a 20 ms interval; none carries 10 ms. |
| 3 | "I-VT fixation filter: velocity threshold 30°/s" (Study Overview) | I-VT (Attention), threshold 100 °/s | The filter name is recorded in all 59 exports; the threshold is read from the Pro Lab gaze-filter settings. 30 °/s is the I-VT (Fixation) default and was never applied. |
| 4 | "I-VT (Attention), velocity threshold 30°/s" (Technical Details) | as above | The README previously named two different filters in two places, and gave the wrong threshold in both. |
| 5 | "Tobii Pro Lab v1.183" | 25.23.1545 | Pro Lab updates automatically; the version above is that used for the final analysis. |
| 6 | "DEVICE1 / MST = Micro-Short-over-the-Needle Technique" | Modified Seldinger Technique | MST is the modified Seldinger technique. The previous expansion was incorrect. |
| 7 | "DEVICE2 / ATG = Accelerated Technique with Guide" | Accelerated Technique with Guidewire | Completed the term. |
| 8 | "DEVICE3 / CON = Conventional over-the-needle" | Catheter-over-needle | Matches the manuscripts. |
| 9 | "randomised crossover study" | counterbalanced, repeated-measures | Allocation sequences were computer-generated. No seed was retained and the sequences cannot be reproduced, so the design is described as counterbalanced rather than randomised. This is a deviation from the parent protocol, which specified a randomisation calculator. The realised allocation has been characterised directly from the deposited attempt-level data: every clinician received exactly five insertions of each configuration and sequences alternated on 13.6 of 15 slots on average, but only 16 distinct sequences were used across the 27 clinicians and configuration is not balanced across scheduled position (chi-square(28) = 162.0, P < 0.001). |
| 10 | "438 total attempts across 30 clinicians × 3 devices × 5 rounds" | 438 recorded attempt rows | 30 × 3 × 5 = 450, not 438. The arithmetic did not hold. |
| 11 | "procedural success (first-attempt catheter placement)" | eventual procedural success | Ten attempts record two or more needle-entry events and seven of those are coded successful, which is impossible under a first-puncture definition. See CODEBOOK correction below. |
| 12 | Event extraction rules absent | documented above | Fixation- and saccade-event rates are wholly determined by the filter settings and the assignment rule; neither was previously stated. |
| 13 | "2 Fr catheter placed over a guidewire with a peel-away introducer" | 2 Fr Seldipur Smartmidline; needle, wire, catheter only | No dilator and no peel-away sheath were used in this study. The five analysed phases are needle access, guidewire insertion, needle removal, catheter advancement and guidewire extraction. Confirmed by the study team. |
| 14 | "generated with a large language model" | computer-generated | The generating tool was not a large language model. What is verifiable is that no seed was retained, so the allocation is counterbalanced rather than randomised; the realised allocation is now characterised from the data rather than asserted. |

## CODEBOOK.md — correction applied

`CODEBOOK.md` previously defined:

```
| Success | int | First-attempt catheter placement: 1=success, 0=failure |
```

That definition was wrong. Ten of the 414 attempts record two or more needle-entry events and
seven of those are coded successful, which is impossible under a first-puncture definition.
`CODEBOOK.md` now defines `Success` as eventual procedural success within the attempt,
requiring vessel puncture, catheter advancement, simulated blood return, aspiration and flush,
and ultrasound-confirmed intraluminal position, with multiple skin punctures permitted and no
time limit imposed.

The source column in the observation log is labelled `FTIS`. That label is a misnomer for a
variable recording eventual success; it is annotated as such in `CODEBOOK.md` and should be
renamed in the log itself.

`CODEBOOK.md` also now warns that `CON` carries two unrelated meanings in this dataset: in
`Phase` it abbreviates *catheter advance*, a step within the modified Seldinger technique,
while in `Device_name` it denotes the *catheter-over-needle configuration*. Phase records
exist only for MST attempts.

## Verification

The corrected rules reproduce the stored `Fixation_count` for all 403 gaze-valid attempts with
zero discrepancies, and refitting the gaze models reproduces the stored outputs exactly
(fixation α = 0.629452, χ²(2) = 74.68; saccade α = 0.754335, χ²(2) = 70.86).

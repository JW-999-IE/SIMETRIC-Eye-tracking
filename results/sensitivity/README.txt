SIMETRIC fixation-duration sensitivity analysis

Attempt-level input:
C:\Users\zeein\OneDrive - National University of Ireland, Galway\Desktop\ET_Fxation\Data export - SIMETRIC\Data export-SIMETRIC-full eye movement\model_dataset_attempt_level.csv

Unique-event input:
C:\Users\zeein\OneDrive - National University of Ireland, Galway\Desktop\ET_Fxation\Data export - SIMETRIC\Data export-SIMETRIC-full eye movement\corrected_fixation_results_v3\fixation_events_unique_all_recordings.csv

Sensitivity analysis 1:
Log attempt-level median fixation duration.
Attempts: 403
Participants: 27

Sensitivity analysis 2:
Log attempt-level mean fixation duration after excluding individual mapped
fixation events longer than 10 seconds.
Initial assigned unique fixation events:
22392

Events excluded (>10 seconds):
218

Attempts containing at least one excluded event:
124

Attempts excluded because no event remained:
1

Final trimmed-mean attempts:
402

Model:
Linear mixed-effects regression on the natural-log outcome, with participant
random intercept and fixed effects for device, experience, repetition, and
chronological attempt sequence. Exponentiated estimates are geometric mean
ratios. Pairwise contrasts use Holm adjustment.

Interpretation:
These are sensitivity analyses. They test whether the principal null device
finding changes when the skewed outcome is represented by the attempt-level
median or when unusually long individual fixation events are removed.

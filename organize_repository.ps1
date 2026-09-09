# SIMETRIC Repository Organization Script
# Run from: C:\Users\zeein\Claude\analysis\SIMETRIC-eye-tracking-repository\
# This copies files from the Data-fixation folder into the clean repository structure.

$SRC = "C:\Users\zeein\Claude\analysis\Data-fixation"
$REPO = $PSScriptRoot

# Create directory structure
$dirs = @(
    "data\raw",
    "data\processed",
    "data\intermediate",
    "scripts",
    "results\fixation_count",
    "results\fixation_duration",
    "results\procedural_success",
    "results\saccade",
    "results\sensitivity",
    "figures"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $REPO $d) | Out-Null
}

# --- Scripts (renamed with numbered prefixes) ---
Copy-Item "$SRC\fixation_analysis_revised_v3.py"           "$REPO\scripts\01_extract_fixations.py"
Copy-Item "$SRC\procedural_success_gee.py"                 "$REPO\scripts\02_procedural_success_gee.py"
Copy-Item "$SRC\python_confirmatory_fixation_models.py"    "$REPO\scripts\03_confirmatory_fixation_models.py"
Copy-Item "$SRC\saccade_analysis_revised.py"               "$REPO\scripts\04_saccade_analysis.py"
Copy-Item "$SRC\fixation_duration_sensitivity.py"          "$REPO\scripts\05_fixation_duration_sensitivity.py"

# --- Processed data ---
Copy-Item "$SRC\model_dataset_attempt_level.csv"           "$REPO\data\processed\"
Copy-Item "$SRC\phase_analysis_mst_phases.csv"             "$REPO\data\processed\"
Copy-Item "$SRC\phase_analysis_temporal_thirds.csv"        "$REPO\data\processed\"
Copy-Item "$SRC\corrected_fixation_results_v3\attempts_with_zero_mapped_fixations.csv" "$REPO\data\processed\"

# --- Intermediate data ---
Copy-Item "$SRC\corrected_fixation_results_v3\attempt_intervals_cleaned.csv"               "$REPO\data\intermediate\"
Copy-Item "$SRC\corrected_fixation_results_v3\file_level_QC.csv"                           "$REPO\data\intermediate\"
Copy-Item "$SRC\corrected_fixation_results_v3\session_recording_map.csv"                   "$REPO\data\intermediate\"
Copy-Item "$SRC\corrected_fixation_results_v3\fixation_events_unique_all_recordings.csv"   "$REPO\data\intermediate\"
Copy-Item "$SRC\corrected_fixation_results_v3\attempts_without_eye_tracking_metrics.csv"   "$REPO\data\intermediate\"
Copy-Item "$SRC\corrected_fixation_results_v3\attempt_level_fixation_metrics_all.csv"      "$REPO\data\intermediate\"
Copy-Item "$SRC\corrected_fixation_results_v3\attempt_level_fixation_metrics_primary_nonabort.csv" "$REPO\data\intermediate\"
Copy-Item "$SRC\corrected_fixation_results_v3\descriptives_by_expertise_device.csv"        "$REPO\data\intermediate\"
Copy-Item "$SRC\corrected_fixation_results_v3\duplicate_attempt_rows_review.csv"           "$REPO\data\intermediate\"

# --- Results: fixation count ---
$fc = "$SRC\python_confirmatory_fixation_results"
Copy-Item "$fc\count_negative_binomial_gee_three_level.txt"   "$REPO\results\fixation_count\"
Copy-Item "$fc\count_alpha_estimation_three_level.txt"        "$REPO\results\fixation_count\"
Copy-Item "$fc\count_device_contrasts_three_level.csv"        "$REPO\results\fixation_count\"
Copy-Item "$fc\count_experience_contrasts_three_level.csv"    "$REPO\results\fixation_count\"
Copy-Item "$fc\count_overall_tests_three_level.csv"           "$REPO\results\fixation_count\"
Copy-Item "$fc\count_negative_binomial_gee_binary.txt"        "$REPO\results\fixation_count\"
Copy-Item "$fc\count_alpha_estimation_binary.txt"             "$REPO\results\fixation_count\"
Copy-Item "$fc\count_device_contrasts_binary.csv"             "$REPO\results\fixation_count\"
Copy-Item "$fc\count_experience_contrasts_binary.csv"         "$REPO\results\fixation_count\"
Copy-Item "$fc\count_overall_tests_binary.csv"                "$REPO\results\fixation_count\"

# --- Results: fixation duration ---
Copy-Item "$fc\duration_mixedlm_three_level.txt"              "$REPO\results\fixation_duration\"
Copy-Item "$fc\duration_device_contrasts_three_level.csv"     "$REPO\results\fixation_duration\"
Copy-Item "$fc\duration_experience_contrasts_three_level.csv" "$REPO\results\fixation_duration\"
Copy-Item "$fc\duration_overall_tests_three_level.csv"        "$REPO\results\fixation_duration\"
Copy-Item "$fc\duration_mixedlm_binary.txt"                   "$REPO\results\fixation_duration\"
Copy-Item "$fc\duration_device_contrasts_binary.csv"          "$REPO\results\fixation_duration\"
Copy-Item "$fc\duration_experience_contrasts_binary.csv"      "$REPO\results\fixation_duration\"
Copy-Item "$fc\duration_overall_tests_binary.csv"             "$REPO\results\fixation_duration\"

# --- Results: procedural success ---
$ps = "$SRC\procedural_success_gee_results\primary_nonabort_414"
Copy-Item "$ps\success_gee_three_level.txt"                   "$REPO\results\procedural_success\"
Copy-Item "$ps\success_gee_binary_experience.txt"             "$REPO\results\procedural_success\"
Copy-Item "$ps\success_overall_tests_three_level.csv"         "$REPO\results\procedural_success\"
Copy-Item "$ps\success_device_contrasts_three_level.csv"      "$REPO\results\procedural_success\"
Copy-Item "$ps\success_experience_contrasts_three_level.csv"  "$REPO\results\procedural_success\"
Copy-Item "$ps\success_overall_tests_binary_experience.csv"   "$REPO\results\procedural_success\"
Copy-Item "$ps\success_device_contrasts_binary_experience.csv" "$REPO\results\procedural_success\"
Copy-Item "$ps\success_experience_contrast_binary.csv"        "$REPO\results\procedural_success\"
Copy-Item "$ps\success_descriptives_by_device.csv"            "$REPO\results\procedural_success\"
Copy-Item "$ps\analysis_dataset_used.csv"                     "$REPO\results\procedural_success\"

# --- Results: saccade ---
$sc = "$SRC\corrected_saccade_results"
Copy-Item "$sc\saccade_rate_three_level_gee.txt"              "$REPO\results\saccade\"
Copy-Item "$sc\saccade_amplitude_three_level_gee.txt"         "$REPO\results\saccade\"
Copy-Item "$sc\saccade_rate_binary_experience_gee.txt"        "$REPO\results\saccade\"
Copy-Item "$sc\saccade_amplitude_binary_experience_gee.txt"   "$REPO\results\saccade\"
Get-ChildItem "$sc\*.csv" | Copy-Item -Destination "$REPO\results\saccade\"

# --- Results: sensitivity ---
$sens = "$SRC\fixation_duration_sensitivity_results"
Get-ChildItem "$sens\*.txt" | Copy-Item -Destination "$REPO\results\sensitivity\"
Get-ChildItem "$sens\*.csv" | Copy-Item -Destination "$REPO\results\sensitivity\"

# --- Figures ---
Copy-Item "$SRC\fig_flow_diagram.png"      "$REPO\figures\"
Copy-Item "$SRC\fig_bayesian_forest.png"   "$REPO\figures\"
Copy-Item "$SRC\fig_temporal_tercile.png"  "$REPO\figures\"
Copy-Item "$SRC\fig_mst_phases.png"        "$REPO\figures\"

# --- Raw data placeholder ---
"# Place raw Tobii TSV files and insertion.xlsx here`n# These are not tracked by git (see .gitignore)" | Out-File "$REPO\data\raw\.gitkeep" -Encoding utf8

Write-Host ""
Write-Host "Repository organized successfully." -ForegroundColor Green
Write-Host "Files copied to: $REPO" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. cd $REPO"
Write-Host "  2. git init"
Write-Host "  3. git add ."
Write-Host '  4. git commit -m "Initial commit: SIMETRIC eye-tracking analysis"'
Write-Host "  5. Optionally copy insertion.xlsx and TSV files to data\raw\ (not tracked by git)"

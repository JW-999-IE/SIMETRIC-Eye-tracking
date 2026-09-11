"""
SIMETRIC Bayesian sensitivity analyses.

Three cell-means models with participant random intercepts:
  1. Fixation count  — negative-binomial (log-link), log(attempt seconds) offset
  2. Fixation duration — log-normal on seconds
  3. Procedural success — Bernoulli (logit-link)

Priors (weakly informative, log scale for count/duration, logit scale for success):
  - Cell means: Normal(0, 2) per device level
  - Participant SD: HalfNormal(0.5)
  - Residual SD (duration only): HalfNormal(1)
  - NB alpha (count only): HalfNormal(1)

Sampling: 4 chains × 2000 post-warmup draws, 1000 warmup, target_accept=0.95.

Key manuscript values to reproduce:
  - P(IRR > 1) = 84.6% for CON-vs-ATG fixation count
  - Duration GMRs within ≤ 0.013 of GEE estimates

Required file:
  data/processed/model_dataset_attempt_level.csv

Run:
  pip install pandas numpy pymc arviz
  python scripts/08_bayesian_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    SCRIPT_DIR.parent / "data" / "processed" / "model_dataset_attempt_level.csv",
    SCRIPT_DIR / "model_dataset_attempt_level.csv",
]
OUT_DIR = SCRIPT_DIR.parent / "results" / "bayesian"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CHAINS = 4
N_DRAWS = 2000
N_TUNE = 1000
TARGET_ACCEPT = 0.95


def find_data() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find model_dataset_attempt_level.csv. "
        "Looked in: " + ", ".join(str(p) for p in DATA_CANDIDATES)
    )


def prepare_data(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[
        d["Mean_fixation_duration_ms"].notna()
        & (d["Mean_fixation_duration_ms"] > 0)
        & (d["Attempt_duration_ms"] > 0)
    ].copy()

    # Encode devices
    devices = sorted(d["Device_name"].unique())  # ATG, CON, MST
    dev_map = {dv: i for i, dv in enumerate(devices)}
    d["dev_idx"] = d["Device_name"].map(dev_map)

    # Encode participants
    participants = sorted(d["Participant"].unique())
    part_map = {p: i for i, p in enumerate(participants)}
    d["part_idx"] = d["Participant"].map(part_map)

    # Derived columns
    d["log_attempt_seconds"] = np.log(d["Attempt_duration_ms"] / 1000.0)
    d["Mean_fixation_duration_s"] = d["Mean_fixation_duration_ms"] / 1000.0
    d["log_duration_s"] = np.log(d["Mean_fixation_duration_s"])

    return d, devices, dev_map, participants


def compute_contrasts(mu_draws, devices, dev_map, label_prefix=""):
    """Compute pairwise contrasts from posterior draws."""
    contrasts = [
        ("CON vs ATG", dev_map["CON"], dev_map["ATG"]),
        ("MST vs ATG", dev_map["MST"], dev_map["ATG"]),
        ("CON vs MST", dev_map["CON"], dev_map["MST"]),
    ]
    rows = []
    for name, idx_a, idx_b in contrasts:
        diff = mu_draws[:, idx_a] - mu_draws[:, idx_b]
        ratio = np.exp(diff)
        rows.append({
            "Contrast": f"{label_prefix}{name}",
            "Ratio_median": float(np.median(ratio)),
            "Ratio_mean": float(np.mean(ratio)),
            "HDI_2.5": float(np.percentile(ratio, 2.5)),
            "HDI_97.5": float(np.percentile(ratio, 97.5)),
            "P_ratio_gt_1": float(np.mean(ratio > 1) * 100),
            "P_ratio_lt_1": float(np.mean(ratio < 1) * 100),
        })
    return pd.DataFrame(rows)


def convergence_report(trace, var_names, label):
    """Write convergence diagnostics."""
    import arviz as az

    lines = [f"Convergence diagnostics: {label}\n"]

    try:
        rhat = az.rhat(trace, var_names=var_names)
        lines.append("R-hat:")
        for var in rhat.data_vars:
            vals = rhat[var].values
            if np.ndim(vals) == 0:
                lines.append(f"  {var}: {float(vals):.4f}")
            else:
                for i, v in enumerate(vals):
                    lines.append(f"  {var}[{i}]: {v:.4f}")
    except Exception as e:
        lines.append(f"R-hat computation failed: {e}")

    try:
        ess = az.ess(trace, var_names=var_names)
        lines.append("\nEffective sample size:")
        for var in ess.data_vars:
            vals = ess[var].values
            if np.ndim(vals) == 0:
                lines.append(f"  {var}: {float(vals):.0f}")
            else:
                for i, v in enumerate(vals):
                    lines.append(f"  {var}[{i}]: {v:.0f}")
    except Exception as e:
        lines.append(f"ESS computation failed: {e}")

    return "\n".join(lines)


def fit_count_model(d, devices, dev_map, participants):
    """Negative-binomial model for fixation count."""
    import pymc as pm
    import arviz as az

    print("\n── Bayesian fixation count (NB) ──")
    n_devices = len(devices)
    n_participants = len(participants)

    with pm.Model() as count_model:
        mu_device = pm.Normal("mu_device", mu=0, sigma=2, shape=n_devices)
        sigma_part = pm.HalfNormal("sigma_part", sigma=0.5)
        z_part = pm.Normal("z_part", mu=0, sigma=1, shape=n_participants)
        part_effect = z_part * sigma_part
        nb_alpha = pm.HalfNormal("nb_alpha", sigma=1)

        log_mu = (
            mu_device[d["dev_idx"].values]
            + part_effect[d["part_idx"].values]
            + d["log_attempt_seconds"].values
        )

        y_obs = pm.NegativeBinomial(
            "y_obs",
            mu=pm.math.exp(log_mu),
            alpha=nb_alpha,
            observed=d["Fixation_count"].astype(int).values,
        )

        trace = pm.sample(
            N_DRAWS, tune=N_TUNE, chains=N_CHAINS, cores=1,
            random_seed=SEED, target_accept=TARGET_ACCEPT,
            return_inferencedata=True, progressbar=True,
        )

    # Summary
    try:
        summary = az.summary(trace, var_names=["mu_device", "sigma_part", "nb_alpha"],
                             hdi_prob=0.95)
    except TypeError:
        summary = az.summary(trace, var_names=["mu_device", "sigma_part", "nb_alpha"],
                             ci_prob=0.95)
    (OUT_DIR / "count_posterior_summary.txt").write_text(
        summary.to_string(), encoding="utf-8"
    )

    # Contrasts
    mu_draws = trace.posterior["mu_device"].values.reshape(-1, n_devices)
    contrasts = compute_contrasts(mu_draws, devices, dev_map, "Count: ")
    contrasts.to_csv(OUT_DIR / "count_device_contrasts.csv", index=False)

    print("  Count contrasts (IRR):")
    for _, row in contrasts.iterrows():
        print(f"    {row['Contrast']}: {row['Ratio_median']:.2f} "
              f"[{row['HDI_2.5']:.2f}, {row['HDI_97.5']:.2f}], "
              f"P(IRR>1) = {row['P_ratio_gt_1']:.1f}%")

    # Convergence
    conv = convergence_report(trace, ["mu_device", "sigma_part", "nb_alpha"], "Count model")
    (OUT_DIR / "count_convergence.txt").write_text(conv, encoding="utf-8")


def fit_duration_model(d, devices, dev_map, participants):
    """Log-normal model for mean fixation duration (seconds)."""
    import pymc as pm
    import arviz as az

    print("\n── Bayesian fixation duration (log-normal, seconds) ──")
    n_devices = len(devices)
    n_participants = len(participants)

    with pm.Model() as duration_model:
        mu_device = pm.Normal("mu_device", mu=0, sigma=2, shape=n_devices)
        sigma_part = pm.HalfNormal("sigma_part", sigma=0.5)
        z_part = pm.Normal("z_part", mu=0, sigma=1, shape=n_participants)
        part_effect = z_part * sigma_part
        sigma_resid = pm.HalfNormal("sigma_resid", sigma=1)

        mu = mu_device[d["dev_idx"].values] + part_effect[d["part_idx"].values]

        y_obs = pm.Normal(
            "y_obs", mu=mu, sigma=sigma_resid,
            observed=d["log_duration_s"].values,
        )

        trace = pm.sample(
            N_DRAWS, tune=N_TUNE, chains=N_CHAINS, cores=1,
            random_seed=SEED, target_accept=TARGET_ACCEPT,
            return_inferencedata=True, progressbar=True,
        )

    try:
        summary = az.summary(trace, var_names=["mu_device", "sigma_part", "sigma_resid"],
                             hdi_prob=0.95)
    except TypeError:
        summary = az.summary(trace, var_names=["mu_device", "sigma_part", "sigma_resid"],
                             ci_prob=0.95)
    (OUT_DIR / "duration_posterior_summary.txt").write_text(
        summary.to_string(), encoding="utf-8"
    )

    mu_draws = trace.posterior["mu_device"].values.reshape(-1, n_devices)
    contrasts = compute_contrasts(mu_draws, devices, dev_map, "Duration GMR: ")
    contrasts.to_csv(OUT_DIR / "duration_device_contrasts.csv", index=False)

    print("  Duration contrasts (GMR):")
    for _, row in contrasts.iterrows():
        print(f"    {row['Contrast']}: {row['Ratio_median']:.2f} "
              f"[{row['HDI_2.5']:.2f}, {row['HDI_97.5']:.2f}], "
              f"P(GMR>1) = {row['P_ratio_gt_1']:.1f}%")

    conv = convergence_report(trace, ["mu_device", "sigma_part", "sigma_resid"], "Duration model")
    (OUT_DIR / "duration_convergence.txt").write_text(conv, encoding="utf-8")


def fit_success_model(d, devices, dev_map, participants):
    """Bernoulli model for procedural success."""
    import pymc as pm
    import arviz as az

    print("\n── Bayesian procedural success (Bernoulli) ──")
    n_devices = len(devices)
    n_participants = len(participants)

    # Use all attempts (including those without valid gaze data, if available).
    # For consistency with GEE success model, include all 414 non-discontinued attempts.
    # But model_dataset_attempt_level.csv has 403 gaze-valid; success data are present
    # for all rows regardless.
    d_success = d.copy()

    with pm.Model() as success_model:
        mu_device = pm.Normal("mu_device", mu=0, sigma=2, shape=n_devices)
        sigma_part = pm.HalfNormal("sigma_part", sigma=0.5)
        z_part = pm.Normal("z_part", mu=0, sigma=1, shape=n_participants)
        part_effect = z_part * sigma_part

        logit_p = mu_device[d_success["dev_idx"].values] + part_effect[d_success["part_idx"].values]

        y_obs = pm.Bernoulli(
            "y_obs",
            logit_p=logit_p,
            observed=d_success["Success"].astype(int).values,
        )

        trace = pm.sample(
            N_DRAWS, tune=N_TUNE, chains=N_CHAINS, cores=1,
            random_seed=SEED, target_accept=TARGET_ACCEPT,
            return_inferencedata=True, progressbar=True,
        )

    try:
        summary = az.summary(trace, var_names=["mu_device", "sigma_part"],
                             hdi_prob=0.95)
    except TypeError:
        summary = az.summary(trace, var_names=["mu_device", "sigma_part"],
                             ci_prob=0.95)
    (OUT_DIR / "success_posterior_summary.txt").write_text(
        summary.to_string(), encoding="utf-8"
    )

    mu_draws = trace.posterior["mu_device"].values.reshape(-1, n_devices)

    # Contrasts as odds ratios
    contrasts_info = [
        ("CON vs ATG", dev_map["CON"], dev_map["ATG"]),
        ("MST vs ATG", dev_map["MST"], dev_map["ATG"]),
        ("CON vs MST", dev_map["CON"], dev_map["MST"]),
    ]
    rows = []
    for name, idx_a, idx_b in contrasts_info:
        diff = mu_draws[:, idx_a] - mu_draws[:, idx_b]
        or_val = np.exp(diff)
        rows.append({
            "Contrast": f"Success OR: {name}",
            "OR_median": float(np.median(or_val)),
            "OR_mean": float(np.mean(or_val)),
            "HDI_2.5": float(np.percentile(or_val, 2.5)),
            "HDI_97.5": float(np.percentile(or_val, 97.5)),
            "P_OR_gt_1": float(np.mean(or_val > 1) * 100),
            "P_OR_lt_1": float(np.mean(or_val < 1) * 100),
        })
    success_df = pd.DataFrame(rows)
    success_df.to_csv(OUT_DIR / "success_device_contrasts.csv", index=False)

    # Also compute predicted success probabilities per device
    prob_rows = []
    for dev in devices:
        i = dev_map[dev]
        logit_draws = mu_draws[:, i]
        p_draws = 1 / (1 + np.exp(-logit_draws))
        prob_rows.append({
            "Device": dev,
            "P_success_median": float(np.median(p_draws)),
            "P_success_mean": float(np.mean(p_draws)),
            "HDI_2.5": float(np.percentile(p_draws, 2.5)),
            "HDI_97.5": float(np.percentile(p_draws, 97.5)),
        })
    pd.DataFrame(prob_rows).to_csv(OUT_DIR / "success_predicted_probabilities.csv", index=False)

    print("  Success contrasts (OR):")
    for _, row in success_df.iterrows():
        print(f"    {row['Contrast']}: {row['OR_median']:.2f} "
              f"[{row['HDI_2.5']:.2f}, {row['HDI_97.5']:.2f}], "
              f"P(OR>1) = {row['P_OR_gt_1']:.1f}%")

    conv = convergence_report(trace, ["mu_device", "sigma_part"], "Success model")
    (OUT_DIR / "success_convergence.txt").write_text(conv, encoding="utf-8")


def main() -> None:
    path = find_data()
    print(f"Data: {path}")
    d, devices, dev_map, participants = prepare_data(path)
    print(f"Gaze-valid attempts: {len(d)}")
    print(f"Participants: {len(participants)}")
    print(f"Devices: {devices}")
    print(f"Device encoding: {dev_map}")

    try:
        import pymc  # noqa: F401
        import arviz  # noqa: F401
    except ImportError:
        print("\nERROR: PyMC and ArviZ are required for Bayesian models.")
        print("Install: pip install pymc arviz")
        return

    fit_count_model(d, devices, dev_map, participants)
    fit_duration_model(d, devices, dev_map, participants)
    fit_success_model(d, devices, dev_map, participants)

    readme = (
        "Bayesian sensitivity analysis results\n\n"
        f"Gaze-valid attempts: {len(d)}\n"
        f"Participants: {len(participants)}\n\n"
        "Models:\n"
        "- count: NB cell-means, participant random intercepts, log(attempt-s) offset\n"
        "- duration: log-normal cell-means on seconds, participant random intercepts\n"
        "- success: Bernoulli cell-means, participant random intercepts\n\n"
        "Priors:\n"
        "- Cell means: Normal(0, 2)\n"
        "- Participant SD: HalfNormal(0.5)\n"
        "- Residual SD (duration): HalfNormal(1)\n"
        "- NB alpha (count): HalfNormal(1)\n\n"
        "Sampling: 4 chains × 2000 draws, 1000 warmup, target_accept=0.95\n"
    )
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")
    print(f"\nResults written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

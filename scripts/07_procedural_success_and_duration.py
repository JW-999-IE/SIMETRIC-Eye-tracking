"""Procedural success and attempt-duration models reported in the SIMETRIC manuscript.

Input : data/processed/procedural_attempts_414.csv (414 non-aborted attempts, 27 clinicians)
Output: results/procedural_and_duration/*.csv

Models
- Eventual success: binomial-logit GEE, clinician clusters, exchangeable working
  correlation, bias-reduced covariance; covariates configuration, prior LPC experience
  (three strata), centred within-configuration repetition (1–5) and centred chronological attempt sequence.
- Attempt duration: Gaussian GEE on log seconds with the same structure; results are
  geometric mean ratios.
- Scheduled-only sensitivity: first five attempts in each clinician-configuration cell.
- Leave-one-clinician-out refits of the success model.
Pairwise configuration contrasts are Holm-adjusted within each model.
"""
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "procedural_and_duration"; OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(ROOT / "data" / "processed" / "procedural_attempts_414.csv")
d["cfg"] = pd.Categorical(d.configuration, ["ATG", "CON", "DST"])
d["exp3"] = pd.Categorical(d.prior_lpc_experience,
                           ["No prior experience", "Some prior experience", "Higher prior experience"])
d["rep_c"] = d.repetition - d.repetition.mean()
d["seq_c"] = d.attempt_sequence - d.attempt_sequence.mean()
d["log_dur"] = np.log(d.attempt_duration_s)
RHS = "C(cfg) + C(exp3) + rep_c + seq_c"
PAIRS = [("CON", "ATG"), ("DST", "ATG"), ("CON", "DST")]

def fit(data, outcome, family):
    return smf.gee(f"{outcome} ~ {RHS}", groups="participant", data=data,
                   cov_struct=sm.cov_struct.Exchangeable(), family=family).fit(cov_type="bias_reduced")

def contrasts(m, label):
    p, V, names = m.params, m.cov_params(), list(m.params.index)
    rows = []
    for a, b in PAIRS:
        L = np.zeros(len(p))
        if a != "ATG": L[names.index(f"C(cfg)[T.{a}]")] += 1
        if b != "ATG": L[names.index(f"C(cfg)[T.{b}]")] -= 1
        est, se = L @ p, np.sqrt(L @ V @ L)
        rows.append(dict(model=label, contrast=f"{a} vs {b}", ratio=np.exp(est),
                         ci_low=np.exp(est - 1.96 * se), ci_high=np.exp(est + 1.96 * se),
                         p=2 * norm.sf(abs(est / se))))
    out = pd.DataFrame(rows); out["p_holm"] = multipletests(out.p, method="holm")[1]
    R = np.zeros((2, len(p))); R[0, names.index("C(cfg)[T.CON]")] = 1; R[1, names.index("C(cfg)[T.DST]")] = 1
    w = m.wald_test(R, scalar=True)
    out["omnibus_chi2_2df"] = float(np.squeeze(w.statistic)); out["omnibus_p"] = float(w.pvalue)
    return out

sched = d.sort_values(["participant", "attempt_sequence"]).copy()
sched["k"] = sched.groupby(["participant", "configuration"]).cumcount() + 1
sched = sched[sched.k <= 5]

res = pd.concat([
    contrasts(fit(d, "eventual_success", sm.families.Binomial()), "success_all_414 (OR)"),
    contrasts(fit(sched, "eventual_success", sm.families.Binomial()), "success_scheduled_405 (OR)"),
    contrasts(fit(d, "log_dur", sm.families.Gaussian()), "duration_all_414 (GMR)"),
])
res.round(4).to_csv(OUT / "configuration_contrasts.csv", index=False)

loco = []
for pid in sorted(d.participant.unique(), key=lambda s: int(s[1:])):
    c = contrasts(fit(d[d.participant != pid], "eventual_success", sm.families.Binomial()), f"omit {pid}")
    loco.append(c)
loco = pd.concat(loco)
loco.round(4).to_csv(OUT / "success_leave_one_clinician_out.csv", index=False)
summ = loco.groupby("contrast").agg(n_refits=("p_holm", "size"),
                                    n_holm_significant=("p_holm", lambda s: int((s < 0.05).sum())),
                                    ratio_min=("ratio", "min"), ratio_max=("ratio", "max"))
summ.round(3).to_csv(OUT / "success_leave_one_clinician_out_summary.csv")

desc = d.groupby("configuration").agg(attempts=("eventual_success", "size"),
                                      successes=("eventual_success", "sum"),
                                      median_duration_s=("attempt_duration_s", "median"),
                                      q1=("attempt_duration_s", lambda s: s.quantile(.25)),
                                      q3=("attempt_duration_s", lambda s: s.quantile(.75)))
desc.round(1).to_csv(OUT / "descriptives_by_configuration.csv")
print(res.round(3).to_string(index=False)); print(summ); print(desc.round(1))

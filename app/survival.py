"""Survival statistics used by the app: Kaplan-Meier, log-rank, stratified Cox, concordance.

Conventions follow R's survival package so results can be checked against the R notebook:
log-transformed 95% confidence intervals for Kaplan-Meier, Efron ties for Cox regression.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
from statsmodels.duration.hazard_regression import PHReg

Z95 = stats.norm.ppf(0.975)
STAGES = ["I", "II", "III", "IV"]


def kaplan_meier(time, event):
    """Step-function table: time, survival and log-transformed 95% CI (as R's survfit default)."""
    sf = SurvfuncRight(np.asarray(time, float), np.asarray(event, int))
    s, se = sf.surv_prob, sf.surv_prob_se
    with np.errstate(divide="ignore", invalid="ignore"):
        half = Z95 * np.where(s > 0, se / s, np.nan)
    lower = np.clip(s * np.exp(-half), 0, 1)
    upper = np.clip(s * np.exp(half), 0, 1)
    curve = pd.DataFrame({"time": sf.surv_times, "surv": s, "lower": lower, "upper": upper})
    start = pd.DataFrame({"time": [0.0], "surv": [1.0], "lower": [1.0], "upper": [1.0]})
    return pd.concat([start, curve], ignore_index=True)


def median_survival(time, event):
    """Median survival with its 95% CI; NaN where the curve never crosses 0.5."""
    sf = SurvfuncRight(np.asarray(time, float), np.asarray(event, int))
    med = sf.quantile(0.5)
    try:
        with np.errstate(divide="ignore", invalid="ignore"):  # statsmodels logs 0 at the curve's tail
            lo, hi = sf.quantile_ci(0.5)
    except Exception:
        lo, hi = np.nan, np.nan
    return med, lo, hi


def number_at_risk(time, at):
    time = np.asarray(time, float)
    return [int((time >= t).sum()) for t in at]


def logrank(time, event, group):
    """Log-rank test across groups: (chi-square, degrees of freedom, p-value)."""
    group = pd.Series(group).astype(str).to_numpy()
    k = len(np.unique(group))
    if k < 2:
        return np.nan, 0, np.nan
    chisq, p = survdiff(np.asarray(time, float), np.asarray(event, int), group)
    return float(chisq), k - 1, float(p)


def design_matrix(df, covariates):
    """Age per year, male vs female, stage II-IV vs stage I. Complete cases only."""
    cols = {"age": ["age"], "sex": ["sex"], "stage": ["stage"]}
    need = [c for cov in covariates for c in cols[cov]]
    d = df.dropna(subset=need).copy()
    X = pd.DataFrame(index=d.index)
    if "age" in covariates:
        X["Age (per year)"] = d["age"]
    if "sex" in covariates:
        X["Male vs female"] = (d["sex"] == "male").astype(float)
    if "stage" in covariates:
        for s in STAGES[1:]:
            X[f"Stage {s} vs I"] = (d["stage"] == s).astype(float)
    return d, X


def cox(df, covariates, stratify=True):
    """Cox proportional-hazards fit (Efron ties), optionally stratified by cohort."""
    d, X = design_matrix(df, covariates)
    X = X.loc[:, X.std() > 0]  # a filter can leave a covariate constant
    strata = d["cohort"].to_numpy() if stratify else None
    fit = PHReg(d["time_months"].to_numpy(), X.to_numpy(), status=d["event"].to_numpy(),
                strata=strata, ties="efron").fit()
    ci = np.exp(fit.conf_int())
    table = pd.DataFrame({
        "Hazard ratio": np.exp(fit.params),
        "95% CI lower": ci[:, 0],
        "95% CI upper": ci[:, 1],
        "p-value": fit.pvalues,
    }, index=X.columns)
    risk = X.to_numpy() @ fit.params
    c = concordance(d["time_months"].to_numpy(), d["event"].to_numpy(), risk, strata)
    return table, {"n": len(d), "events": int(d["event"].sum()), "concordance": c}


def concordance(time, event, risk, strata=None):
    """Harrell's C, counting pairs within strata only (as R does for a stratified model)."""
    strata = np.zeros(len(time)) if strata is None else np.asarray(strata)
    concordant = total = 0.0
    for s in np.unique(strata):
        m = strata == s
        t, e, r = time[m], event[m], risk[m]
        for i in np.where(e == 1)[0]:
            later = t > t[i]  # pairs where patient i failed first
            total += later.sum()
            concordant += (r[i] > r[later]).sum() + 0.5 * (r[i] == r[later]).sum()
    return concordant / total if total else np.nan

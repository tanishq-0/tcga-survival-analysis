"""Check the app's statistics reproduce the R notebook (R/survival_analysis.Rmd / .html).

Expected values are copied from the rendered R output. Run: python app/check_against_r.py
"""
from pathlib import Path

import pandas as pd
from survival import median_survival, logrank, cox

R = {  # from R/survival_analysis.html
    "patients": 6864, "deaths": 2047, "median_os_months": 88.1,
    "logrank_chisq": 424, "cox_n": 4332, "cox_events": 1053,
    "hr_age": 1.036, "hr_male": 0.9405, "hr_II": 1.4668, "hr_III": 2.1712, "hr_IV": 5.5563,
    "concordance": 0.684,
}

dat = pd.read_csv(Path(__file__).parent / "data" / "tcga_survival.csv")
med, _, _ = median_survival(dat["time_months"], dat["event"])
staged = dat.dropna(subset=["stage"])
chisq, df, p = logrank(staged["time_months"], staged["event"], staged["stage"])
table, info = cox(dat, ["age", "sex", "stage"], stratify=True)
hr = table["Hazard ratio"]

got = {
    "patients": len(dat), "deaths": int(dat["event"].sum()), "median_os_months": round(med, 1),
    "logrank_chisq": round(chisq), "cox_n": info["n"], "cox_events": info["events"],
    "hr_age": round(hr["Age (per year)"], 4), "hr_male": round(hr["Male vs female"], 4),
    "hr_II": round(hr["Stage II vs I"], 4), "hr_III": round(hr["Stage III vs I"], 4),
    "hr_IV": round(hr["Stage IV vs I"], 4), "concordance": round(info["concordance"], 3),
}
ok = True
for k, expected in R.items():
    close = abs(got[k] - expected) <= max(0.01 * abs(expected), 0.002)
    ok &= close
    print(f"{k:18s} R={expected:<10} app={got[k]:<10} {'OK' if close else 'MISMATCH'}")
print("ALL MATCH" if ok else "SOME MISMATCHES")

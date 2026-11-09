"""Build the app's patient-level survival table from the GDC metadata extract.

Mirrors the cleaning in R/survival_analysis.Rmd so the app and the
R notebook analyse exactly the same patients:
  - GDC's several missing-value tokens are all treated as missing;
  - one row per case_id (the extract is keyed by slide);
  - event = 1 if Dead, 0 if Alive; time = days_to_death for deaths, else days_to_last_follow_up;
  - records with no event, no time, or time <= 0 are dropped;
  - AJCC stage collapsed to I-IV, testing IV before III before II before I.

Usage: python prepare_data.py [path to metadata.csv]
The default input is R/data/metadata.csv, as written by R/fetch_metadata.py.
Writes data/tcga_survival.csv with only the columns the app needs (no identifiers).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

NA_TOKENS = ["", "NA", "'--", "--", "not reported", "Not Reported",
             "unknown", "Unknown", "Not Allowed To Collect"]
DAYS_PER_MONTH = 30.44


def collapse_stage(s):
    if pd.isna(s):
        return np.nan
    for level in ("IV", "III", "II", "I"):  # order matters, as in the R case_when
        if level in s:
            return level
    return np.nan


def prepare(metadata_path):
    raw = pd.read_csv(metadata_path, na_values=NA_TOKENS, keep_default_na=False, low_memory=False)
    cases = raw.sort_values(["case_id", "slide_id"]).drop_duplicates("case_id")

    event = cases["vital_status"].map({"Dead": 1, "Alive": 0})
    time_days = np.where(event == 1, cases["days_to_death"], cases["days_to_last_follow_up"]).astype(float)
    dat = pd.DataFrame({
        "cohort": cases["project_id"].str.replace("TCGA-", "", regex=False),
        "primary_site": cases["primary_site"],
        "event": event,
        "time_days": time_days,
        "stage": cases["ajcc_pathologic_stage"].map(collapse_stage),
        "sex": cases["gender"],
        "age": pd.to_numeric(cases["age_at_index"], errors="coerce"),
    })
    counts = {"records": len(raw), "patients": len(cases)}
    dat = dat[dat["event"].notna() & dat["time_days"].notna() & (dat["time_days"] > 0)].copy()
    counts["analysable"] = len(dat)
    dat["event"] = dat["event"].astype(int)
    dat["time_months"] = (dat["time_days"] / DAYS_PER_MONTH).round(3)
    return dat.drop(columns="time_days").reset_index(drop=True), counts


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "R" / "data" / "metadata.csv"
    dat, counts = prepare(source)
    out = Path(__file__).parent / "data" / "tcga_survival.csv"
    out.parent.mkdir(exist_ok=True)
    dat.to_csv(out, index=False)
    print(counts, "deaths:", int(dat["event"].sum()), "->", out)

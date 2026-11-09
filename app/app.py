"""TCGA Survival Explorer: interactive survival analysis of 6,864 TCGA cancer patients.

Run locally (from the repository root):  streamlit run app/app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from survival import kaplan_meier, median_survival, number_at_risk, logrank, cox, STAGES

st.set_page_config(page_title="TCGA Survival Explorer", page_icon="📈", layout="wide")
PALETTE = ["#2E5EAA", "#E07B39", "#3A9D5D", "#C0392B", "#8E5EA2", "#7F7F7F", "#B8860B", "#17A2B8"]


@st.cache_data
def load():
    return pd.read_csv(Path(__file__).parent / "data" / "tcga_survival.csv")


data = load()

# ---------------- sidebar filters ----------------
st.sidebar.header("Filter patients")
sites = sorted(data["primary_site"].dropna().unique())
chosen_sites = st.sidebar.multiselect("Primary site", sites, placeholder="All sites")
cohorts = sorted(data.loc[data["primary_site"].isin(chosen_sites or sites), "cohort"].unique())
chosen_cohorts = st.sidebar.multiselect("TCGA cohort", cohorts, placeholder="All cohorts in selected sites")
sexes = st.sidebar.multiselect("Sex", ["female", "male"], default=["female", "male"])
age_min, age_max = int(data["age"].min()), int(data["age"].max())
ages = st.sidebar.slider("Age at diagnosis", age_min, age_max, (age_min, age_max))

# Patients with no recorded age stay in unless the age range has been narrowed.
age_ok = data["age"].between(*ages) | (data["age"].isna() & (ages == (age_min, age_max)))
df = data[data["cohort"].isin(chosen_cohorts or cohorts) & data["sex"].isin(sexes) & age_ok]
st.sidebar.caption(f"{len(df):,} of {len(data):,} patients selected")
st.sidebar.download_button("Download selected data (CSV)", df.to_csv(index=False), "tcga_survival_selection.csv")

st.title("TCGA Survival Explorer")
st.caption("Overall survival of TCGA cancer patients from NCI Genomic Data Commons clinical data. "
           "Observational data: results are associations, not causal effects.")
if len(df) < 20 or df["event"].sum() < 5:
    st.warning("Too few patients or deaths in this selection for a meaningful analysis. Widen the filters.")
    st.stop()

tab_overview, tab_km, tab_cox, tab_quality, tab_about = st.tabs(
    ["Overview", "Kaplan–Meier", "Cox model", "Data quality", "About"])

# ---------------- overview ----------------
with tab_overview:
    med, lo, hi = median_survival(df["time_months"], df["event"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{len(df):,}")
    c2.metric("Deaths", f"{int(df['event'].sum()):,}", f"{df['event'].mean():.0%} of patients", delta_color="off")
    c3.metric("Median overall survival", "not reached" if np.isnan(med) else f"{med:.1f} months")
    c4.metric("Median follow-up (observed)", f"{df['time_months'].median():.1f} months")

    summary = (df.groupby("cohort")
                 .agg(site=("primary_site", "first"), patients=("event", "size"), deaths=("event", "sum"),
                      staged=("stage", lambda s: 100 * s.notna().mean()), median_follow_up=("time_months", "median"))
                 .sort_values("patients", ascending=False))
    st.subheader("By cohort")
    st.dataframe(summary, use_container_width=True, column_config={
        "staged": st.column_config.ProgressColumn("AJCC staged", format="%.0f%%", min_value=0, max_value=100),
        "median_follow_up": st.column_config.NumberColumn("Median follow-up (months)", format="%.1f")})

# ---------------- Kaplan-Meier ----------------
GROUPINGS = {"None (all selected patients)": None, "AJCC stage": "stage", "Sex": "sex",
             "Age band": "age_band", "Cohort (largest 8)": "cohort"}

with tab_km:
    left, right = st.columns([1, 3])
    with left:
        grouping = GROUPINGS[st.radio("Compare by", list(GROUPINGS))]
        horizon = st.slider("Show up to (months)", 12, 240, 120, step=12)
        show_ci = st.checkbox("95% confidence bands", value=True)
    km = df.copy()
    km["age_band"] = pd.cut(km["age"], [0, 50, 60, 70, 120], labels=["<50", "50–59", "60–69", "70+"], right=False)
    if grouping == "cohort":
        km = km[km["cohort"].isin(km["cohort"].value_counts().index[:8])]
    if grouping is not None:
        km = km.dropna(subset=[grouping])
    if grouping == "stage":
        right.caption(f"{len(df) - len(km):,} unstaged patients excluded (see Data quality).")
    groups = ([("All", km)] if grouping is None else
              [(str(g), sub) for g, sub in km.groupby(grouping, observed=True) if len(sub) >= 10])
    if grouping == "stage":
        groups.sort(key=lambda g: STAGES.index(g[0]))

    fig = go.Figure()
    medians, ticks = [], list(range(0, horizon + 1, 24 if horizon > 60 else 12))
    risk_rows = {}
    for i, (name, sub) in enumerate(groups):
        colour = PALETTE[i % len(PALETTE)]
        curve = kaplan_meier(sub["time_months"], sub["event"])
        curve = curve[curve["time"] <= horizon]
        label = name if grouping is None else f"{name} (n={len(sub):,})"
        if show_ci:
            fig.add_trace(go.Scatter(x=curve["time"], y=curve["upper"], line_shape="hv", mode="lines",
                                     line=dict(width=0), showlegend=False, hoverinfo="skip"))
            r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
            fig.add_trace(go.Scatter(x=curve["time"], y=curve["lower"], line_shape="hv", mode="lines",
                                     line=dict(width=0), fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.15)",
                                     showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=curve["time"], y=curve["surv"], line_shape="hv", mode="lines",
                                 name=label, line=dict(color=colour, width=2.5),
                                 hovertemplate="%{x:.0f} months: %{y:.1%}<extra>" + name + "</extra>"))
        m, mlo, mhi = median_survival(sub["time_months"], sub["event"])
        medians.append({"Group": name, "Patients": len(sub), "Deaths": int(sub["event"].sum()),
                        "Median survival (months)": m, "95% CI lower": mlo, "95% CI upper": mhi})
        risk_rows[name] = number_at_risk(sub["time_months"], ticks)
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      xaxis_title="Months since diagnosis", yaxis_title="Overall survival probability",
                      yaxis=dict(range=[0, 1.02], tickformat=".0%"), legend=dict(orientation="h", y=1.08, traceorder="normal"))
    right.plotly_chart(fig, use_container_width=True)

    with right:
        st.caption("Number at risk")
        st.dataframe(pd.DataFrame(risk_rows, index=[f"{t} m" for t in ticks]).T, use_container_width=True)
        if grouping is not None and len(groups) > 1:
            chisq, dof, p = logrank(km["time_months"], km["event"], km[grouping])
            p_text = "p < 1e-16" if p < 1e-16 else f"p = {p:.2g}"
            st.info(f"**Log-rank test:** χ² = {chisq:.1f} on {dof} df, {p_text}")
        st.dataframe(pd.DataFrame(medians).round(1), hide_index=True, use_container_width=True)

# ---------------- Cox model ----------------
with tab_cox:
    left, right = st.columns([1, 3])
    with left:
        covs = [c for c, on in [("age", st.checkbox("Age", True)), ("sex", st.checkbox("Sex", True)),
                                ("stage", st.checkbox("AJCC stage", True))] if on]
        stratify = st.checkbox("Stratify by cohort", True,
                               help="Lets each cohort have its own baseline hazard, so differences between "
                                    "cancer types don't masquerade as covariate effects.")
    if not covs:
        right.info("Choose at least one covariate.")
    else:
        try:
            table, info = cox(df, covs, stratify)
        except Exception as err:
            right.error(f"The model could not be fitted on this selection ({err}). Widen the filters.")
        else:
            with right:
                c1, c2, c3 = st.columns(3)
                c1.metric("Patients in model", f"{info['n']:,}")
                c2.metric("Deaths", f"{info['events']:,}")
                c3.metric("Concordance (C-index)", f"{info['concordance']:.3f}")
                if "stage" in covs:
                    st.caption("Complete cases only: unstaged patients are excluded when stage is in the model.")
                forest = go.Figure(go.Scatter(
                    x=table["Hazard ratio"], y=table.index, mode="markers",
                    marker=dict(size=10, color=PALETTE[0]),
                    error_x=dict(type="data", symmetric=False,
                                 array=table["95% CI upper"] - table["Hazard ratio"],
                                 arrayminus=table["Hazard ratio"] - table["95% CI lower"])))
                forest.add_vline(x=1, line_dash="dash", line_color="grey")
                forest.update_layout(height=80 + 55 * len(table), margin=dict(l=10, r=10, t=10, b=10),
                                     xaxis=dict(type="log", title="Hazard ratio (log scale, 95% CI)",
                                                tickvals=[0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]),
                                     yaxis=dict(autorange="reversed"))
                st.plotly_chart(forest, use_container_width=True)
                st.dataframe(table.style.format({"Hazard ratio": "{:.3f}", "95% CI lower": "{:.3f}",
                                                 "95% CI upper": "{:.3f}", "p-value": "{:.2g}"}),
                             use_container_width=True)
                st.caption("Hazard ratio > 1 means a higher risk of death. Efron ties; "
                           "matches R's coxph() on the full dataset (see check_against_r.py).")

# ---------------- data quality ----------------
with tab_quality:
    st.markdown("**Staging is missing by design, not at random.** AJCC staging doesn't apply to several tumour "
                "types (for example brain tumours are graded, not staged), so whole cohorts have no stage. "
                "Analyses that use stage silently drop those cohorts.")
    miss = (df.assign(staged=df["stage"].notna())
              .groupby("cohort")["staged"].mean().sort_values())
    bar = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation="h", marker_color=PALETTE[0]))
    bar.update_layout(height=max(300, 18 * len(miss)), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="Share of patients with an AJCC stage", tickformat=".0%", range=[0, 1]))
    st.plotly_chart(bar, use_container_width=True)
    hist = go.Figure(go.Histogram(x=df["time_months"], nbinsx=60, marker_color=PALETTE[1]))
    hist.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Follow-up time",
                       xaxis_title="Months", yaxis_title="Patients")
    st.plotly_chart(hist, use_container_width=True)

# ---------------- about ----------------
with tab_about:
    st.markdown("""
**Data.** Clinical metadata for TCGA diagnostic slides from the NIH/NCI Genomic Data Commons (open access),
reduced to one row per patient. 7,690 patients → 6,864 with a usable survival time (826 dropped for missing
vital status or follow-up, or follow-up ≤ 0 days). Survival time is days to death for patients who died and
days to last follow-up for those alive (right-censored), converted to months (÷ 30.44).

**Methods.** Kaplan–Meier estimates with log-transformed 95% confidence intervals, log-rank tests, and Cox
proportional-hazards regression (Efron ties, optionally stratified by cohort) with Harrell's concordance.
AJCC stage is collapsed to I–IV.

**Validation.** On the full dataset this app reproduces my R analysis (survival / coxph) to four decimal
places: median overall survival 88.1 months, log-rank χ² = 424 across stages, stage IV vs I hazard ratio
5.556, concordance 0.684. Run `python app/check_against_r.py` to repeat the check.

**Caveats.** Observational repository data, not trial data: no randomisation, sparse treatment information
and follow-up that varies between cohorts. Estimates are associational only.
""")

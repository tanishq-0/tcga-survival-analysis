# TCGA Survival and Regression Analysis (R)

Survival and regression analysis of patient metadata retrieved from the NIH/NCI
Genomic Data Commons REST API, covering 32 TCGA project cohorts (6,864 patients
with evaluable follow-up, 2,047 deaths). The rendered notebook is
[`survival_analysis.html`](survival_analysis.html).

## Methods covered

- Kaplan-Meier estimation and log-rank testing
- Cox proportional hazards regression, stratified by cohort
- Proportional hazards assumption testing (`cox.zph`)
- Logistic regression with calibration assessment by decile of predicted risk

## Setup

1. Install the R packages:

   ```r
   install.packages(c("readr", "dplyr", "stringr", "ggplot2", "survival", "survminer", "broom"))
   ```

2. Download `annotation712_update.json` from the
   [HistGen dataset](https://huggingface.co/datasets/david4real/HistGen) on
   Hugging Face. It is free, but you need to sign in and accept the dataset
   terms first. The file lists the TCGA diagnostic slides that define the cohort.
3. Rebuild the clinical metadata from the GDC API (Python 3, standard library
   only; takes a few minutes):

   ```bash
   python fetch_metadata.py path/to/annotation712_update.json
   ```

   This writes `data/metadata.csv`, which the notebook's `paths` chunk reads.
4. Knit `survival_analysis.Rmd` to HTML in RStudio, or run
   `rmarkdown::render("survival_analysis.Rmd")`.

## Data

`fetch_metadata.py` looks up each slide on the GDC Files endpoint and keeps the
case-level clinical fields below. It is adapted from the extraction code written
for my MSc research project. The output holds GDC case identifiers, so it is
git-ignored rather than committed.

| Field | Use |
|---|---|
| `vital_status` | Event indicator (Dead / Alive) |
| `days_to_death` | Time to event for patients who died |
| `days_to_last_follow_up` | Censoring time for patients alive at last contact |
| `ajcc_pathologic_stage` | Prognostic covariate, collapsed to I-IV |
| `age_at_index`, `gender` | Model covariates |
| `project_id` | Cohort, used as a stratification variable |
| `case_id` | Patient identifier, used to deduplicate from slide level |

The dataset is indexed by slide and a patient may contribute several slides, so
it is reduced to one row per `case_id` before analysis. Survival is a
patient-level question and analysing it at slide level would weight patients by
how many slides they happen to have.

### Re-running against the current GDC release

The rendered notebook was produced from the extract made for my MSc project.
GDC has since changed its clinical data model: sex is now recorded as
`sex_at_birth`, and last-contact times have largely moved from the diagnosis
record to separate follow-up records. `fetch_metadata.py` maps the current
fields onto the column names above. It also takes stage from the diagnosis
flagged as the primary disease. The original extraction took the first diagnosis
record returned, which is not always the primary one, so some staged patients
were probably treated as unstaged. A fresh download therefore keeps more
patients, more of them staged, and the estimates will differ from those in the
rendered HTML.

## Caveats

Observational repository data, not trial data. No randomisation, sparse
treatment information, and variable follow-up between cohorts. The estimates
are associational only. Full limitations are recorded at the end of the
notebook.

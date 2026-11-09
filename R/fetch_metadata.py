#!/usr/bin/env python3
"""Rebuild the clinical metadata used by survival_analysis.Rmd from the NCI Genomic Data Commons API.

The cohort is the set of TCGA diagnostic slides in the HistGen dataset
(annotation712_update.json from https://huggingface.co/datasets/david4real/HistGen;
free, but you need to sign in and accept the dataset terms to download it).
Each slide id has the form <slide-submitter-id>.<image-uuid>, and the matching
GDC slide image is the file "<id>.svs". The script looks each one up on the
public GDC /files endpoint and keeps the case-level clinical fields the analysis
uses. If an image UUID has changed since the annotation was made, it falls back
to matching on the slide submitter id (the part before the dot).

Usage:
    python fetch_metadata.py path/to/annotation712_update.json
    python fetch_metadata.py slide_ids.txt            # or one slide id per line

Writes data/metadata.csv next to this script: one row per slide. The file holds
GDC case identifiers, so it is git-ignored rather than committed.

GDC's data model changes between releases, so the script maps the current
fields onto the column names the notebook expects:
  - gender: demographic.sex_at_birth (older releases: demographic.gender);
  - ajcc_pathologic_stage: from the diagnosis flagged as the primary disease
    (a case can have several diagnosis records, in no guaranteed order);
  - days_to_last_follow_up: the latest of the primary diagnosis's
    days_to_last_follow_up and any follow-up record's days_to_follow_up
    (newer releases keep last contact in the follow-up records).
A fresh download therefore differs from the extract behind the rendered
notebook; see the README.
"""
import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GDC_FILES = "https://api.gdc.cancer.gov/files"
BATCH = 100
MAX_RETRIES = 5
DEFAULT_OUT = Path(__file__).parent / "data" / "metadata.csv"

FIELDS = [
    "file_name",
    "cases.case_id",
    "cases.primary_site",
    "cases.project.project_id",
    "cases.project.primary_site",
    "cases.demographic.gender",
    "cases.demographic.sex_at_birth",
    "cases.demographic.vital_status",
    "cases.demographic.age_at_index",
    "cases.demographic.days_to_death",
    "cases.diagnoses.diagnosis_is_primary_disease",
    "cases.diagnoses.ajcc_pathologic_stage",
    "cases.diagnoses.days_to_last_follow_up",
    "cases.follow_ups.days_to_follow_up",
]
COLUMNS = ["slide_id", "case_id", "project_id", "primary_site", "gender", "vital_status",
           "age_at_index", "days_to_death", "days_to_last_follow_up", "ajcc_pathologic_stage"]


def read_slide_ids(path):
    """Slide ids from a HistGen annotation file (all splits) or a plain list, duplicates dropped."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = [item["id"] for items in data.values() for item in items]
    else:
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return list(dict.fromkeys(ids))


def query(filters, size):
    """POST a filter to the GDC /files endpoint and return the hits, retrying with back-off."""
    payload = {"filters": filters, "fields": ",".join(FIELDS), "format": "JSON", "size": str(size)}
    request = urllib.request.Request(GDC_FILES, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)["data"]["hits"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            print(f"  request failed (attempt {attempt}): {err}", file=sys.stderr)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("GDC request failed after retries")


def first(items):
    return items[0] if items else {}


def joined(value):
    """GDC returns some fields (e.g. project primary_site) as lists."""
    return "|".join(str(v) for v in value) if isinstance(value, list) else value


def primary_diagnosis(case):
    """The diagnosis flagged as the primary disease, else the first one listed."""
    diagnoses = case.get("diagnoses") or []
    return next((d for d in diagnoses if d.get("diagnosis_is_primary_disease") is True), first(diagnoses))


def last_follow_up(case, diagnosis):
    """Latest recorded contact in days, across the primary diagnosis and the follow-up records."""
    days = [diagnosis.get("days_to_last_follow_up")]
    days += [f.get("days_to_follow_up") for f in case.get("follow_ups") or []]
    days = [d for d in days if d is not None]
    return max(days) if days else None


def to_row(slide_id, hit):
    case = first(hit.get("cases") or [])
    project = case.get("project") or {}
    demographic = case.get("demographic") or {}
    diagnosis = primary_diagnosis(case)
    return {
        "slide_id": slide_id,
        "case_id": case.get("case_id"),
        "project_id": project.get("project_id"),
        "primary_site": joined(project.get("primary_site") or case.get("primary_site")),
        "gender": demographic.get("sex_at_birth") or demographic.get("gender"),
        "vital_status": demographic.get("vital_status"),
        "age_at_index": demographic.get("age_at_index"),
        "days_to_death": demographic.get("days_to_death"),
        "days_to_last_follow_up": last_follow_up(case, diagnosis),
        "ajcc_pathologic_stage": diagnosis.get("ajcc_pathologic_stage"),
    }


def fetch(slide_ids):
    rows = {}
    for start in range(0, len(slide_ids), BATCH):
        batch = slide_ids[start:start + BATCH]
        by_file_name = {sid + ".svs": sid for sid in batch}
        hits = query({"op": "in", "content": {"field": "file_name", "value": list(by_file_name)}},
                     size=len(batch) + 10)
        for hit in hits:
            sid = by_file_name.get(hit.get("file_name"))
            if sid is not None:
                rows[sid] = to_row(sid, hit)
        print(f"batch {start // BATCH + 1}: {sum(s in rows for s in batch)}/{len(batch)} matched", flush=True)

    # Fallback for stale image UUIDs: match on the slide submitter id instead.
    missing = [sid for sid in slide_ids if sid not in rows]
    for sid in missing:
        slide = sid.split(".")[0]
        hits = query({"op": "and", "content": [
            {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
            {"op": "in", "content": {"field": "cases.samples.portions.slides.submitter_id", "value": [slide]}},
        ]}, size=10)
        hit = next((h for h in hits if h.get("file_name", "").startswith(slide + ".")), None)
        if hit is not None:
            rows[sid] = to_row(sid, hit)
    if missing:
        print(f"fallback: recovered {sum(s in rows for s in missing)} of {len(missing)} unmatched slides")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("slides", help="HistGen annotation .json, or a text file with one slide id per line")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output CSV (default: {DEFAULT_OUT.name} in data/)")
    args = parser.parse_args()

    slide_ids = read_slide_ids(args.slides)
    print(f"{len(slide_ids)} slide ids")
    rows = fetch(slide_ids)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows[sid] for sid in slide_ids if sid in rows)
    unresolved = len(slide_ids) - len(rows)
    print(f"wrote {len(rows)} rows ({len({r['case_id'] for r in rows.values()})} cases) to {args.out}"
          + (f"; {unresolved} slides not found on GDC" if unresolved else ""))


if __name__ == "__main__":
    main()

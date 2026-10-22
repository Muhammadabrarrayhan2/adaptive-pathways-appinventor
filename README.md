# Adaptive Pathways for App Inventor — data pipeline

**Author:** Muhammad Abrar Rayhan ([@Muhammadabrarrayhan2](https://github.com/Muhammadabrarrayhan2))

Reproducible data pipeline for the pilot *"Adaptive Pathways for App Inventor: An Explainable Skill-Gap and
Next-Step Recommendation Pilot"* (M. A. Rayhan, 2026). It builds deliverables D1 (skill-gap taxonomy, draft),
the data layer of D2 (explainable next-step prototype) and D4 (reproducibility kit) from public sources only.

```
pip install -r requirements.txt
gh auth login                       # or set GITHUB_TOKEN — used for the GitHub source
python run_pipeline.py all          # ingest -> parse -> features -> cluster -> stages -> catalog -> report
python run_pipeline.py recommend path/to/project.aia
python -m pytest -q tests
```

## Stages

| # | Stage | Output | What it does |
|---|-------|--------|--------------|
| 1 | `ingest` | `data/raw/aia/<sha256>.aia`, `data/raw/manifest.jsonl` | Collects `.aia` projects from four sources (below). Content-addressed storage removes byte-identical copies; every file is validated (real App Inventor project, size and zip-bomb limits) before it is kept. |
| 2 | `parse` | `data/interim/parsed.jsonl` | Parses each project with **mit-cml/aiatools** plus a direct `.scm`/`.bky` reader that adds block-tree depth, procedure parameters and extension components, and cross-checks aiatools. Pre-2013 block formats are normalised. |
| 3 | `features` | `data/processed/profiles.csv`, `feature_spec.json` | One skill profile per project (≈70 documented features: size, events, control flow, procedures, data structures, component families, AI use). Flags near-duplicates (e.g. unmodified tutorials) and empty projects. |
| 4 | `cluster` | `clusters.csv`, `cluster_profiles.csv`, `cluster_selection.json` | Ward hierarchical clustering + k-prototypes; k chosen by silhouette among bootstrap-stable solutions; seed stability and exemplars per cluster for teacher review. |
| 5 | `stages` | `taxonomy/stage_labels.yaml`, `stage_alignment.json` | Draft taxonomy v0 (UI-only → event-driven → data-aware → sensor-aware → AI-component), cluster/stage alignment, and an editable file for the two-teacher review. |
| 6 | `catalog` | `data/catalog/resources.json` | Next-step resources from the official tutorial and AI-curriculum indexes; concepts are measured from each tutorial's own solution `.aia` where one exists. Every URL is checked. |
| 7 | `report` | `reports/data_report.md`, `reports/figures/` | Data quality, parser agreement, comparison with the published Gallery baseline, clustering and stage results, limitations. |

`recommend` uses the outputs above: it places an `.aia` in a stage and returns three diverse next steps, each with
a templated rationale built only from parsed facts and corpus statistics (no generated text).

## Sources

| Source | How | Notes |
|--------|-----|-------|
| MIT App Inventor Gallery | public listings + `storage.googleapis.com/mitai2gallery/gallery/apps/<id>/aia` | Anonymous access reaches only the featured listings; search requires login. |
| Official tutorials & AI curriculum | `appinventor.mit.edu/explore/ai2/tutorials`, `/explore/ai-with-mit-app-inventor` | Supplies the catalog and tutorial solution projects. |
| Public GitHub repositories | repository search (App Inventor topics/keywords) → git tree → `*.aia` | Only repos with an explicitly open licence (`allowed_licenses`); forks skipped; at most 8 projects per repo (seeded sample). |
| Consented classroom artifacts | `data/raw/classroom_inbox/` + `consent.csv` | Ingested only with consent (parental consent for minors); media removed and identity replaced by the consent id. |

## Ethics and privacy (proposal §4)

* No author identity is stored in analysis tables: project.properties `main` (which embeds the App Inventor account
  name), app names and screen names are dropped at parse time; provenance URLs live only in the raw manifest.
* Classroom archives are rebuilt without assets (photos/voices of children) before storage.
* Raw third-party archives are excluded from git (`.gitignore`); `manifest.jsonl` records how to re-fetch them.

## Configuration

Everything tunable — sources, queries, per-repo caps, corpus target, clustering range, seed — is in `config.yaml`.
Stages can be re-run individually (`python run_pipeline.py cluster`) or resumed (`python run_pipeline.py all --from parse`).
Ingestion is incremental: already-fetched URLs are skipped.

## Data sources and terms (checked 2026-10-22)

| Source | Terms / robots.txt | Status |
|--------|--------------------|--------|
| App Inventor Gallery | App Inventor Terms: all Gallery projects are CC BY-SA 4.0 ("anyone can view, modify, and redistribute"); no robots.txt rules; no clause against automated access. | Used. Attribution kept via `public_url`. |
| appinventor.mit.edu tutorials | robots.txt disallows only `/ai2` and `/setup`; documentation licensed CC BY 4.0. | Used. |
| GitHub | Accessed through the official API with a token, within rate limits. GitHub's Acceptable Use Policies allow research use of public data. | Only openly licensed repos; unlicensed (`unspecified`/`NOASSERTION`) files are excluded and purged by `enforce_policy`. Raw files are not committed (`.gitignore`). |
| Community forum | Forum ToS: "You may not automate access to the forum ... with a web crawler ... or other computer program that is not a web browser"; robots.txt disallows `/search`. | **Not used.** An earlier forum source was removed together with all data it produced. Use only with written permission from the forum operators. |

## License

Code: MIT (see `LICENSE`). Derived data under `data/` describes third-party projects that keep their own licences
(recorded per project in `data/raw/manifest.jsonl`); raw `.aia` archives are not included in this repository.

## How to cite and attribution

This project was created by **Muhammad Abrar Rayhan** ([@Muhammadabrarrayhan2](https://github.com/Muhammadabrarrayhan2)).
If you use the code, the derived data or the report, please cite it (GitHub's "Cite this repository" button uses
`CITATION.cff`):

> Rayhan, M. A. (2026). *Adaptive Pathways for App Inventor: data pipeline* (Version 0.1.1) [Computer software].
> https://github.com/Muhammadabrarrayhan2/adaptive-pathways-appinventor

Under the MIT License, copies and substantial portions of this code must keep the copyright notice in `LICENSE`
and in each source file.

# Retinal AV Segmentation & Vascular Biomarker Pipeline

Three tasks on color fundus photos (CFP): (1) artery/vein segmentation from CFP alone,
(2) cross-modal AV segmentation with paired FFA, (3) vascular biomarker quantification.

Full strategy, timeline, and open questions: see the plan doc referenced in this
project's Claude Code session, or `reports/technical_report_draft.md` once started.

## Layout

- `external/` — git submodules: `rrwnet` (baseline architecture, pretrained weights),
  `cmrrwnet` (reference baseline architecture), `mnet_deepcdr` (optic disc
  localization), `minima` (CFP/FFA registration).
- `configs/` — training configs (YAML), one per task/fold variant.
- `src/datasets/` — shared Dataset classes for Task 1 (CFP-only) and Task 2 (CFP+FFA).
- `src/models/` — RRWNet, baseline-equivalent 5ch fusion, and improved fusion models.
- `src/losses/` — Dice/BCE + soft-clDice (topology-aware).
- `src/metrics/` — local reimplementation of the official scoring formulas (pixel +
  topology metrics for Task 1/2, MAE/SMAPE for Task 3) — build and calibrate this
  BEFORE trusting any local model comparison.
- `src/biomarkers/` — SIVA zone geometry, Knudtson CRAE/CRVE/AVR, density, fractal
  dimension (Task 3).
- `src/od_localization/` — optic disc detection (required for Task 3 zones; not
  provided in the dataset).
- `src/registration/` — MINIMA-based CFP/FFA registration (required for Task 2; pairs
  are not pre-registered).
- `data/` — not tracked in git. Download per `scripts/00_download_and_inspect.py`
  instructions; do not re-host (dataset usage terms).
- `notebooks/validate_metrics_against_baseline.ipynb` — reproduce the reference
  baseline's published scores locally before trusting any of this repo's own
  experiments.
- `notebooks/validate_biomarkers_against_gt.ipynb` — reproduce provided GT biomarker
  labels from GT masks before ever running Task 3 on predicted masks.
- `submissions_log.csv` — every real leaderboard submission: date, task, config hash,
  local score, leaderboard score. Submission window closes 2026-07-31 23:59, capped
  at 5 submissions/day — don't burn slots on unvalidated experiments.

## Setup

```bash
conda env create -f environment.yml
conda activate gave2
git submodule update --init --recursive
```

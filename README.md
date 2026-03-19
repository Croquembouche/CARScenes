# CARScenes Release Workspace

This repository contains the finalized CARScenes annotation workspace, release artifacts, benchmark tooling, and NeurIPS paper sources.

## What Is Here
- `dataset/`: unified final label JSON files and source image layout.
- `release/carscenes-v1/`: canonical annotation-only release, schema, splits, reports, and Croissant metadata.
- `scripts/build_carscenes_release.py`: rebuilds the release from the unified labels.
- `scripts/validate_carscenes_release.py`: validates the release records and split manifests.
- `scripts/evaluate_carscenes_predictions.py`: scores model predictions against a CARScenes split.
- `scripts/export_prediction_template.py`: exports a prediction template aligned to a benchmark split.
- `scripts/generate_benchmark_report.py`: aggregates multiple runs into CSV, Markdown, and LaTeX benchmark tables.
- `scripts/carscenes_review_dashboard.py`: local web app for benchmark upload/evaluation and second-pass human review.
- `scripts/carscenes_spotcheck_app.py`: local browser UI for independent spot-check annotation.
- `scripts/score_spotcheck_agreement.py`: computes agreement between released labels and second-pass spot-check labels.
- `configs/carscenes_annotation_taxonomy.json`: finalized standardized annotation taxonomy.
- `configs/carscenes_benchmark_slices.json`: fixed benchmark slice definitions.
- `paper/`: NeurIPS manuscript source and final output PDF.

## Core Commands
Rebuild and validate the canonical release:

```bash
python scripts/build_carscenes_release.py
python scripts/validate_carscenes_release.py
```

Evaluate one prediction JSONL against a split:

```bash
python scripts/evaluate_carscenes_predictions.py \
  --reference release/carscenes-v1/splits/gold-test-100.jsonl \
  --predictions path/to/predictions.jsonl
```

Export a blank prediction template for the held-out benchmark:

```bash
python scripts/export_prediction_template.py \
  --reference release/carscenes-v1/splits/gold-test-100.jsonl \
  --output runs/benchmark/gold-test-100/template_predictions.jsonl \
  --copy-metadata
```

Generate aggregate benchmark tables from multiple model runs:

```bash
python scripts/generate_benchmark_report.py \
  --manifest templates/benchmark_runs_template.csv \
  --output-dir runs/benchmark/reports/gold-test-100
```

Launch the unified local review dashboard:

```bash
python scripts/carscenes_review_dashboard.py
```

Then open `http://127.0.0.1:8770/`.

## Licensing
- CARScenes annotations, schema, split manifests, and metadata: `CC BY 4.0`
- Repository code: `MIT`
- Source images remain under the original upstream dataset licenses and are not relicensed here.

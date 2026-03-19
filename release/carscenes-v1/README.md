# CARScenes v1 Release

This directory contains the annotation-only CARScenes v1 release artifacts generated from the raw labels in `dataset/train/labels`.

## Contents
- `data/carscenes_v1_records.jsonl`: canonicalized annotation records with metadata.
- `schema/carscenes_v1_schema.json`: machine-readable schema for the benchmark fields.
- `splits/train.json`, `splits/silver-dev-500.json`, `splits/gold-test-100.json`: fixed split manifests.
- `audit/gold_test_100.jsonl`: held-out gold split records.
- `reports/release_summary.json`: dataset and split counts.
- `croissant.json`: Croissant metadata.

## Current State
- Total canonical records: 5192
- Missing raw labels skipped: 3
- Extra raw labels ignored: 2
- Normalization actions applied: 0

## Benchmark Policy
- `silver-dev-500` is for model selection only.
- Released records are fully human-corrected after VLM-assisted first-pass labeling.
- `gold-test-100` is the held-out gold split for headline evaluation.
- Images are not redistributed here; records point back to upstream image paths.

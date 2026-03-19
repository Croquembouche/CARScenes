#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from carscenes_release_utils import (
    adverse_conditions,
    build_release,
    has_traffic_light,
    has_vru,
    summarize_records,
    write_json,
    write_jsonl,
)


def build_croissant(base_path: Path) -> dict:
    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "ml": "http://mlcommons.org/croissant/",
        },
        "@type": "cr:Dataset",
        "name": "CARScenes v1",
        "description": "Annotation-only release for the CARScenes scene-understanding benchmark with fully human-corrected labels.",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "carscenes-records",
                "name": "carscenes_v1_records.jsonl",
                "contentUrl": str((base_path / "data" / "carscenes_v1_records.jsonl").as_posix()),
                "encodingFormat": "application/jsonlines",
            },
            {
                "@type": "cr:FileObject",
                "@id": "carscenes-schema",
                "name": "carscenes_v1_schema.json",
                "contentUrl": str((base_path / "schema" / "carscenes_v1_schema.json").as_posix()),
                "encodingFormat": "application/json",
            },
        ],
    }


def build_release_notes(summary: dict) -> str:
    return f"""# CARScenes v1 Release

This directory contains the annotation-only CARScenes v1 release artifacts generated from the raw labels in `dataset/train/labels`.

## Contents
- `data/carscenes_v1_records.jsonl`: canonicalized annotation records with metadata.
- `schema/carscenes_v1_schema.json`: machine-readable schema for the benchmark fields.
- `splits/train.json`, `splits/silver-dev-500.json`, `splits/gold-test-100.json`: fixed split manifests.
- `audit/gold_test_100.jsonl`: held-out gold split records.
- `reports/release_summary.json`: dataset and split counts.
- `croissant.json`: Croissant metadata.

## Current State
- Total canonical records: {summary['records']}
- Missing raw labels skipped: {summary['issues']['missing_labels']}
- Extra raw labels ignored: {summary['issues']['extra_labels']}
- Normalization actions applied: {summary['issues']['normalization_actions']}

## Benchmark Policy
- `silver-dev-500` is for model selection only.
- Released records are fully human-corrected after VLM-assisted first-pass labeling.
- `gold-test-100` is the held-out gold split for headline evaluation.
- Images are not redistributed here; records point back to upstream image paths.
"""


def build_license_matrix() -> str:
    return """# License Matrix

## CARScenes Contributions
- Canonicalized annotations, split files, schema, validation code, and benchmark tooling: intended for release under CC BY 4.0 for data artifacts and MIT for code.

## Upstream Image Sources
- Cityscapes: upstream terms apply.
- KITTI: upstream terms apply.
- Argoverse1: upstream terms apply.
- nuScenes: upstream terms apply.

This repository does not re-license or redistribute the upstream image assets.
"""


def enrich_records(records: list[dict]) -> list[dict]:
    enriched = []
    for record in records:
        enriched_record = copy.deepcopy(record)
        enriched_record["_slice_tags"] = {
            "severity_bucket": "1-3" if record["Severity"] <= 3 else "4-6" if record["Severity"] <= 6 else "7-10",
            "adverse_conditions": adverse_conditions(record),
            "vru_present": has_vru(record),
            "traffic_light_present": has_traffic_light(record),
            "lighting_bucket": "day" if record["TimeOfDay"] == "Daytime" else "dusk_or_night",
        }
        enriched.append(enriched_record)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical CARScenes v1 release.")
    parser.add_argument("--dataset-root", default="dataset", help="Path to the raw dataset root.")
    parser.add_argument(
        "--output-root",
        default="release/carscenes-v1",
        help="Directory to write the release artifacts to.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic split generation.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    release = build_release(dataset_root=dataset_root, seed=args.seed)
    records = enrich_records(release.records)
    summary = summarize_records(records, release.issues, release.splits)

    write_jsonl(output_root / "data" / "carscenes_v1_records.jsonl", records)
    write_json(output_root / "schema" / "carscenes_v1_schema.json", release.schema)
    write_json(output_root / "reports" / "release_summary.json", summary)
    write_json(output_root / "reports" / "release_issues.json", release.issues)
    write_json(output_root / "croissant.json", build_croissant(output_root))

    for split_name, ids in release.splits.items():
        split_records = [record for record in records if record["_id"] in set(ids)]
        write_json(output_root / "splits" / f"{split_name}.json", ids)
        write_jsonl(output_root / "splits" / f"{split_name}.jsonl", split_records)

    write_jsonl(
        output_root / "audit" / "gold_test_100.jsonl",
        [record for record in records if record["_id"] in set(release.splits["gold-test-100"])],
    )
    write_json(
        output_root / "audit" / "gold_agreement_25.json",
        release.splits["gold-agreement-25"],
    )
    (output_root / "README.md").write_text(build_release_notes(summary), encoding="utf-8")
    (output_root / "LICENSES.md").write_text(build_license_matrix(), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from carscenes_annotation_taxonomy import normalize_record_for_agreement
from evaluate_carscenes_predictions import (
    evaluate_list_field,
    evaluate_scalar_field,
    load_jsonl,
    severity_metrics,
)


SCALAR_FIELDS = [
    "Scene",
    "TimeOfDay",
    "Weather",
    "RoadConditions",
    "Directionality",
    "CameraCondition",
    "Visibility.General",
    "LaneInformation.NumberOfLanes",
    "LaneInformation.LaneMarkings",
    "TrafficSigns.TrafficSignsVisibility",
    "TrafficSigns.TrafficLightState",
    "Vehicles.TotalNumber",
    "Ego-Vehicle.Direction",
    "Ego-Vehicle.Maneuver",
]
LIST_FIELDS = [
    "LaneInformation.SpecialLanes",
    "TrafficSigns.TrafficSignsTypes",
    "Vehicles.VehicleTypes",
    "Vehicles.InMotion",
    "Vehicles.States",
    "Pedestrians",
    "Visibility.SpecificImpairments",
]


def path_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def write_field_summary(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["field", "metric", "value", "support"])
        writer.writeheader()
        for field, metrics in summary["scalar"].items():
            writer.writerow(
                {
                    "field": field,
                    "metric": "accuracy",
                    "value": metrics["accuracy"],
                    "support": metrics["support"],
                }
            )
        for field, metrics in summary["list"].items():
            for metric_name in ("precision", "recall", "f1"):
                writer.writerow(
                    {
                        "field": field,
                        "metric": metric_name,
                        "value": metrics[metric_name],
                        "support": metrics["support"],
                    }
                )
        for metric_name in ("accuracy", "mae", "rmse", "quadratic_weighted_kappa"):
            writer.writerow(
                {
                    "field": "Severity",
                    "metric": metric_name,
                    "value": summary["severity"][metric_name],
                    "support": summary["severity"]["support"],
                }
            )


def write_disagreements(path: Path, reference: list[dict[str, Any]], reviewed: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["_id", "field", "gold_value", "reviewer_value"])
        writer.writeheader()
        for record in reference:
            review = reviewed.get(record["_id"])
            if review is None:
                continue
            for field in SCALAR_FIELDS:
                gold = path_get(record, field)
                reviewer_value = path_get(review, field)
                if reviewer_value is None:
                    continue
                if reviewer_value != gold:
                    writer.writerow(
                        {
                            "_id": record["_id"],
                            "field": field,
                            "gold_value": json.dumps(gold, ensure_ascii=True),
                            "reviewer_value": json.dumps(reviewer_value, ensure_ascii=True),
                        }
                    )
            for field in LIST_FIELDS:
                gold = sorted(path_get(record, field, []) or [])
                reviewer_value = sorted(path_get(review, field, []) or [])
                if not reviewer_value:
                    continue
                if reviewer_value != gold:
                    writer.writerow(
                        {
                            "_id": record["_id"],
                            "field": field,
                            "gold_value": json.dumps(gold, ensure_ascii=True),
                            "reviewer_value": json.dumps(reviewer_value, ensure_ascii=True),
                        }
                    )
            gold_severity = record.get("Severity")
            reviewer_severity = review.get("Severity")
            if reviewer_severity is not None and reviewer_severity != gold_severity:
                writer.writerow(
                    {
                        "_id": record["_id"],
                        "field": "Severity",
                        "gold_value": json.dumps(gold_severity, ensure_ascii=True),
                        "reviewer_value": json.dumps(reviewer_severity, ensure_ascii=True),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an independent CARScenes spot-check subset against the released labels.")
    parser.add_argument(
        "--reference",
        default="release/carscenes-v1/splits/gold-agreement-25.jsonl",
        help="Reference JSONL file for the agreement subset.",
    )
    parser.add_argument(
        "--reviewed",
        required=True,
        help="JSONL file containing the independent reviewer annotations.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for summary JSON and CSV outputs.",
    )
    args = parser.parse_args()

    reference = [normalize_record_for_agreement(record) for record in load_jsonl(Path(args.reference))]
    reviewed_records = [normalize_record_for_agreement(record) for record in load_jsonl(Path(args.reviewed))]
    reviewed = {record["_id"]: record for record in reviewed_records}

    summary = {
        "coverage": sum(1 for record in reference if record["_id"] in reviewed) / len(reference) if reference else 0.0,
        "scalar": {field: evaluate_scalar_field(reference, reviewed, field) for field in SCALAR_FIELDS},
        "list": {field: evaluate_list_field(reference, reviewed, field) for field in LIST_FIELDS},
        "severity": severity_metrics(reference, reviewed),
        "reviewer": next((record.get("_review", {}).get("reviewer", "") for record in reviewed_records), ""),
        "support": len(reference),
        "completed_records": sum(1 for record in reviewed_records if record.get("_review", {}).get("complete")),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spotcheck_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_field_summary(output_dir / "spotcheck_field_summary.csv", summary)
    write_disagreements(output_dir / "spotcheck_disagreements.csv", reference, reviewed)

    print(f"wrote spot-check agreement report to {output_dir}")


if __name__ == "__main__":
    main()

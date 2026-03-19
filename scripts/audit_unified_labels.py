#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from carscenes_annotation_taxonomy import FIELD_PATH_TO_TAXONOMY_PATH, load_annotation_taxonomy


RAW_PATH_ALIASES = {
    "TrafficSigns.TrafficSignsTypes": ["TrafficSigns.TrafficSignsTypes", "TrafficSigns.Types"],
    "TrafficSigns.TrafficSignsVisibility": ["TrafficSigns.TrafficSignsVisibility", "TrafficSigns.Visibility"],
    "Vehicles.VehicleTypes": ["Vehicles.VehicleTypes", "Vehicles.Types", "TrafficSigns.VehicleTypes"],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def walk_json_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def path_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def candidate_paths(field_path: str) -> list[str]:
    return RAW_PATH_ALIASES.get(field_path, [field_path])


def extract_field_value(record: dict[str, Any], field_path: str) -> Any:
    for candidate in candidate_paths(field_path):
        value = path_get(record, candidate, None)
        if value is not None:
            return value
    return None


def taxonomy_allowed_values(taxonomy: dict[str, Any], field_path: str) -> set[str]:
    taxonomy_path = FIELD_PATH_TO_TAXONOMY_PATH[field_path]
    value = path_get(taxonomy, taxonomy_path, [])
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def field_kind(taxonomy: dict[str, Any], field_path: str) -> str:
    taxonomy_path = FIELD_PATH_TO_TAXONOMY_PATH[field_path]
    value = path_get(taxonomy, taxonomy_path, [])
    return "list" if isinstance(value, list) and value and not isinstance(value[0], int) and field_path in {
        "LaneInformation.SpecialLanes",
        "TrafficSigns.TrafficSignsTypes",
        "Vehicles.InMotion",
        "Vehicles.VehicleTypes",
        "Vehicles.States",
        "Pedestrians",
        "Visibility.SpecificImpairments",
    } else ("severity" if field_path == "Severity" else "scalar")


def iter_record_values(record: dict[str, Any], field_path: str, kind: str) -> list[str]:
    value = extract_field_value(record, field_path)
    if value is None:
        return []
    if kind == "list":
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]
    return [str(value)]


def audit_records(
    records: list[tuple[str, dict[str, Any]]],
    taxonomy: dict[str, Any],
    field_paths: list[str],
) -> dict[str, Any]:
    issues_by_field: dict[str, Counter[str]] = {field: Counter() for field in field_paths}
    examples_by_field: dict[str, dict[str, str]] = {field: {} for field in field_paths}
    support_by_field: Counter[str] = Counter()

    for record_id, record in records:
        for field_path in field_paths:
            kind = field_kind(taxonomy, field_path)
            values = iter_record_values(record, field_path, kind)
            if not values:
                continue
            support_by_field[field_path] += 1
            allowed = taxonomy_allowed_values(taxonomy, field_path)
            for value in values:
                if value not in allowed:
                    issues_by_field[field_path][value] += 1
                    examples_by_field[field_path].setdefault(value, record_id)

    summary: dict[str, Any] = {}
    for field_path in field_paths:
        issues = issues_by_field[field_path]
        summary[field_path] = {
            "records_with_field": support_by_field[field_path],
            "num_out_of_taxonomy_values": sum(issues.values()),
            "distinct_out_of_taxonomy_values": len(issues),
            "top_out_of_taxonomy_values": [
                {"value": value, "count": count, "example_record": examples_by_field[field_path][value]}
                for value, count in issues.most_common(20)
            ],
        }
    return summary


def build_raw_record_stream(labels_root: Path) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    for path in walk_json_files(labels_root):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.append((str(path.relative_to(labels_root)), record))
    return records


def build_release_record_stream(jsonl_path: Path) -> list[tuple[str, dict[str, Any]]]:
    return [(record.get("_id", f"record:{idx}"), record) for idx, record in enumerate(load_jsonl(jsonl_path))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UDriveVLMDataset labels against the standardized CARScenes taxonomy.")
    parser.add_argument(
        "--taxonomy",
        default="configs/carscenes_annotation_taxonomy.json",
        help="Standardized taxonomy JSON file.",
    )
    parser.add_argument(
        "--labels-root",
        default="dataset/train/labels",
        help="Root directory for raw label JSON files.",
    )
    parser.add_argument(
        "--release-jsonl",
        default="release/carscenes-v1/data/carscenes_v1_records.jsonl",
        help="Canonical release JSONL file to audit.",
    )
    parser.add_argument(
        "--output",
        default="tmp/unified_label_audit.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    taxonomy = load_annotation_taxonomy(args.taxonomy)
    field_paths = list(FIELD_PATH_TO_TAXONOMY_PATH.keys())

    raw_records = build_raw_record_stream(Path(args.labels_root))
    release_records = build_release_record_stream(Path(args.release_jsonl))

    report = {
        "taxonomy": str(Path(args.taxonomy).resolve()),
        "raw_labels_root": str(Path(args.labels_root).resolve()),
        "release_jsonl": str(Path(args.release_jsonl).resolve()),
        "raw_label_files": len(raw_records),
        "release_records": len(release_records),
        "raw_labels": audit_records(raw_records, taxonomy, field_paths),
        "release_records_audit": audit_records(release_records, taxonomy, field_paths),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote unified label audit to {output_path}")


if __name__ == "__main__":
    main()

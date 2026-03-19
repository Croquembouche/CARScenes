#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def empty_record_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for name, spec in schema.get("properties", {}).items():
        if name.startswith("_"):
            continue
        spec_type = spec.get("type")
        if spec_type == "object":
            record[name] = empty_record_from_schema(spec)
        elif spec_type == "array":
            record[name] = []
        else:
            record[name] = None
    return record


def merge_reference_fields(template: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(template))
    for key in list(merged.keys()):
        if key in reference:
            merged[key] = reference[key]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CARScenes prediction JSONL template.")
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference split JSONL used to derive record ids and field structure.",
    )
    parser.add_argument(
        "--schema",
        default="release/carscenes-v1/schema/carscenes_v1_schema.json",
        help="Schema JSON file for CARScenes.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for the prediction template.",
    )
    parser.add_argument(
        "--prefill-from-reference",
        action="store_true",
        help="Prefill benchmark fields with the reference labels instead of blanks.",
    )
    parser.add_argument(
        "--copy-metadata",
        action="store_true",
        help="Copy _source and image metadata into the template records.",
    )
    args = parser.parse_args()

    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    reference = load_jsonl(Path(args.reference))
    base_template = empty_record_from_schema(schema)

    output_records: list[dict[str, Any]] = []
    for record in reference:
        template = merge_reference_fields(base_template, record) if args.prefill_from_reference else json.loads(
            json.dumps(base_template)
        )
        template["_id"] = record["_id"]
        if args.copy_metadata:
            for meta_key in ("_source", "_image_relpath", "_raw_label_relpath", "_record_hash"):
                if meta_key in record:
                    template[meta_key] = record[meta_key]
        output_records.append(template)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"wrote {len(output_records)} template records to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carscenes_release_utils import validate_record


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a CARScenes v1 release.")
    parser.add_argument(
        "--release-root",
        default="release/carscenes-v1",
        help="Path to the generated release directory.",
    )
    args = parser.parse_args()

    release_root = Path(args.release_root)
    records = load_jsonl(release_root / "data" / "carscenes_v1_records.jsonl")
    schema = json.loads((release_root / "schema" / "carscenes_v1_schema.json").read_text(encoding="utf-8"))

    id_set = set()
    errors: list[str] = []
    for record in records:
        record_id = record.get("_id")
        if record_id in id_set:
            errors.append(f"duplicate_id:{record_id}")
        id_set.add(record_id)
        errors.extend(validate_record(record, schema))

    split_dir = release_root / "splits"
    for split_file in sorted(split_dir.glob("*.json")):
        split_ids = json.loads(split_file.read_text(encoding="utf-8"))
        for record_id in split_ids:
            if record_id not in id_set:
                errors.append(f"split_missing_record:{split_file.name}:{record_id}")

    if errors:
        for error in errors[:200]:
            print(error)
        raise SystemExit(1)

    print(f"validated {len(records)} records")


if __name__ == "__main__":
    main()

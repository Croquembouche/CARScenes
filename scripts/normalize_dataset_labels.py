#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carscenes_release_utils import SOURCE_NAMES, canonicalize_raw_label


WRAPPER_FILES = {"split.py", "unifiedAnalysis.json"}


def dataset_label_from_canonical(record: dict) -> dict:
    payload = {key: value for key, value in record.items() if not key.startswith("_")}
    auxiliary = payload.pop("_auxiliary", {})
    if isinstance(auxiliary, dict):
        for key, value in auxiliary.items():
            payload[key] = value
    return payload


def iter_label_files(dataset_root: Path) -> list[Path]:
    paths: list[Path] = []
    for source in SOURCE_NAMES:
        source_root = dataset_root / "train" / "labels" / source
        for path in source_root.rglob("*.json"):
            if path.name in WRAPPER_FILES:
                continue
            paths.append(path)
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite raw CARScenes label JSON files to the final unified taxonomy.")
    parser.add_argument("--dataset-root", default="dataset", help="Path to dataset root.")
    parser.add_argument("--dry-run", action="store_true", help="Compute changes without writing files.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    changed = 0
    examined = 0

    for path in iter_label_files(dataset_root):
        examined += 1
        raw = json.loads(path.read_text(encoding="utf-8"))
        canonical, _warnings = canonicalize_raw_label(raw)
        unified = dataset_label_from_canonical(canonical)
        before = json.dumps(raw, sort_keys=True, ensure_ascii=True)
        after = json.dumps(unified, sort_keys=True, ensure_ascii=True)
        if before == after:
            continue
        changed += 1
        if not args.dry_run:
            path.write_text(json.dumps(unified, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    mode = "would change" if args.dry_run else "changed"
    print(f"{mode} {changed} of {examined} label files")


if __name__ == "__main__":
    main()

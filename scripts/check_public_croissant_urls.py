#!/usr/bin/env python3
"""Check that Croissant distribution URLs are publicly fetchable and hash-stable."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--croissant",
        type=Path,
        default=Path("release/carscenes-v1/croissant.json"),
        help="Path to the Croissant JSON file to check.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="URL fetch timeout in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.croissant.read_text())
    failures = 0
    for item in metadata.get("distribution", []):
        item_id = item.get("@id", "<missing-id>")
        url = item.get("contentUrl")
        expected = item.get("sha256")
        if not url or not expected:
            print(f"{item_id}: missing contentUrl or sha256")
            failures += 1
            continue
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CARScenes-public-url-check/1.0"})
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"{item_id}: FETCH_ERROR {exc}")
            failures += 1
            continue
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            print(f"{item_id}: SHA_MISMATCH expected={expected} actual={actual}")
            failures += 1
        else:
            print(f"{item_id}: OK bytes={len(payload)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

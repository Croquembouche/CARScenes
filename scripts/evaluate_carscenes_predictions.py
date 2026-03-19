#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def path_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def evaluate_scalar_field(reference: list[dict], predictions: dict[str, dict], field: str) -> dict[str, Any]:
    correct = 0
    total = 0
    for record in reference:
        target = path_get(record, field)
        if target is None:
            continue
        pred = path_get(predictions.get(record["_id"], {}), field)
        if pred is None:
            continue
        total += 1
        correct += int(pred == target)
    return {"accuracy": correct / total if total else 0.0, "support": total}


def evaluate_list_field(reference: list[dict], predictions: dict[str, dict], field: str) -> dict[str, Any]:
    tp = fp = fn = support = 0
    for record in reference:
        gold = path_get(record, field)
        if gold is None:
            continue
        pred = path_get(predictions.get(record["_id"], {}), field, [])
        if pred is None:
            pred = []
        gold_set = set(gold)
        pred_set = set(pred)
        support += 1
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def severity_metrics(reference: list[dict], predictions: dict[str, dict]) -> dict[str, Any]:
    gold_values = []
    pred_values = []
    exact = 0
    for record in reference:
        gold = record.get("Severity")
        pred = predictions.get(record["_id"], {}).get("Severity")
        if gold is None or pred is None:
            continue
        gold_values.append(gold)
        pred_values.append(pred)
        exact += int(gold == pred)
    support = len(gold_values)
    if not support:
        return {"accuracy": 0.0, "mae": 0.0, "rmse": 0.0, "quadratic_weighted_kappa": 0.0, "support": 0}
    mae = sum(abs(g - p) for g, p in zip(gold_values, pred_values)) / support
    rmse = math.sqrt(sum((g - p) ** 2 for g, p in zip(gold_values, pred_values)) / support)
    return {
        "accuracy": exact / support,
        "mae": mae,
        "rmse": rmse,
        "quadratic_weighted_kappa": quadratic_weighted_kappa(gold_values, pred_values, min_rating=0, max_rating=10),
        "support": support,
    }


def quadratic_weighted_kappa(gold: list[int], pred: list[int], min_rating: int, max_rating: int) -> float:
    n = max_rating - min_rating + 1
    observed = [[0.0 for _ in range(n)] for _ in range(n)]
    gold_hist = [0.0 for _ in range(n)]
    pred_hist = [0.0 for _ in range(n)]
    for g, p in zip(gold, pred):
        observed[g - min_rating][p - min_rating] += 1.0
        gold_hist[g - min_rating] += 1.0
        pred_hist[p - min_rating] += 1.0
    total = float(len(gold))
    if total == 0:
        return 0.0
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(n):
            weight = ((i - j) ** 2) / ((n - 1) ** 2) if n > 1 else 0.0
            expected = (gold_hist[i] * pred_hist[j]) / total
            num += weight * observed[i][j]
            den += weight * expected
    return 1.0 - (num / den if den else 0.0)


def load_slices(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def record_matches(record: dict[str, Any], spec: dict[str, Any]) -> bool:
    def check_clause(clause: dict[str, Any]) -> bool:
        value = path_get(record, clause["field"])
        if "equals" in clause:
            return value == clause["equals"]
        if "in" in clause:
            return value in clause["in"]
        if "contains" in clause:
            if not isinstance(value, list):
                return False
            return clause["contains"] in value
        return False

    all_ok = all(check_clause(item) for item in spec.get("all", []))
    any_ok = True
    if "any" in spec:
        any_ok = any(check_clause(item) for item in spec["any"])
    return all_ok and any_ok


def slice_metrics(reference: list[dict], predictions: dict[str, dict], slices: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for spec in slices:
        subset = [record for record in reference if record_matches(record, spec)]
        results[spec["name"]] = {
            "count": len(subset),
            "severity": severity_metrics(subset, predictions),
            "Scene": evaluate_scalar_field(subset, predictions, "Scene"),
            "Weather": evaluate_scalar_field(subset, predictions, "Weather"),
            "Vehicles.TotalNumber": evaluate_scalar_field(subset, predictions, "Vehicles.TotalNumber"),
            "Pedestrians": evaluate_list_field(subset, predictions, "Pedestrians"),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CARScenes benchmark predictions.")
    parser.add_argument("--reference", required=True, help="Reference JSONL file.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL file with matching _id values.")
    parser.add_argument(
        "--slices",
        default="configs/carscenes_benchmark_slices.json",
        help="JSON slice configuration file.",
    )
    parser.add_argument("--output", default="", help="Optional path to write the JSON summary to.")
    args = parser.parse_args()

    reference = load_jsonl(Path(args.reference))
    predictions = {record["_id"]: record for record in load_jsonl(Path(args.predictions))}
    slices = load_slices(Path(args.slices))

    scalar_fields = [
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
    list_fields = [
        "LaneInformation.SpecialLanes",
        "TrafficSigns.TrafficSignsTypes",
        "Vehicles.VehicleTypes",
        "Vehicles.InMotion",
        "Vehicles.States",
        "Pedestrians",
        "Visibility.SpecificImpairments",
    ]

    summary = {
        "scalar": {field: evaluate_scalar_field(reference, predictions, field) for field in scalar_fields},
        "list": {field: evaluate_list_field(reference, predictions, field) for field in list_fields},
        "severity": severity_metrics(reference, predictions),
        "slices": slice_metrics(reference, predictions, slices),
        "support": len(reference),
    }

    rendered = json.dumps(summary, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

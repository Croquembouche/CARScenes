#!/usr/bin/env python3
"""
Evaluate the fine-tuned Qwen2-VL checkpoint against the validation set.

For every sample in val_dataset.jsonl, this script:
1. Runs inference with the LoRA checkpoint.
2. Parses the structured JSON prediction.
3. Compares each field against the ground-truth annotation.
4. Reports per-field accuracy/precision/recall metrics and logs mismatches.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from swift.llm import (
    BaseArguments,
    InferRequest,
    PtEngine,
    RequestConfig,
    get_model_tokenizer,
    get_template,
)
from swift.tuners import Swift

RUN_DIR = Path("output/qwen2vl_lora/v3-20251009-143039")
ADAPTER_DIR = RUN_DIR / "checkpoint-1911"
VAL_DATASET = RUN_DIR / "val_dataset.jsonl"
OUTPUT_DIR = RUN_DIR / "structured_eval"

# Ensure consistent GPU selection and preprocessing budget.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("MAX_PIXELS", "1003520")


ScalarField = Tuple[str, ...]
ListField = Tuple[str, ...]

SCALAR_FIELDS: Tuple[ScalarField, ...] = (
    ("Scene",),
    ("TimeOfDay",),
    ("Weather",),
    ("RoadConditions",),
    ("Directionality",),
    ("CameraCondition",),
    ("Severity",),
    ("Visibility", "General"),
    ("Ego-Vehicle", "Direction"),
    ("Ego-Vehicle", "Maneuver"),
    ("LaneInformation", "NumberOfLanes"),
    ("LaneInformation", "LaneMarkings"),
    ("TrafficSigns", "TrafficSignsVisibility"),
    ("TrafficSigns", "TrafficLightState"),
    ("Vehicles", "TotalNumber"),
)

LIST_FIELDS: Tuple[ListField, ...] = (
    ("LaneInformation", "SpecialLanes"),
    ("TrafficSigns", "TrafficSignsTypes"),
    ("TrafficSigns", "VehicleTypes"),
    ("Vehicles", "InMotion"),
    ("Vehicles", "States"),
    ("Pedestrians",),
    ("Visibility", "SpecificImpairments"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2-VL validation set.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RUN_DIR,
        help="Training run directory containing checkpoints and val_dataset.jsonl.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=ADAPTER_DIR,
        help="LoRA checkpoint directory to evaluate.",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=VAL_DATASET,
        help="Validation JSONL file.",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default="5",
        help="Number of samples to evaluate (set to 'None' for full set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write evaluation summary.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples to batch per inference call.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per sample.",
    )
    args = parser.parse_args()
    if args.limit.lower() == "none":
        args.limit = None
    else:
        args.limit = int(args.limit)
    return args


def load_dataset(path: Path, limit: Optional[int] = None) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def strip_loss(messages: Iterable[Dict]) -> List[Dict]:
    cleaned = []
    for msg in messages:
        cleaned.append({"role": msg["role"], "content": msg["content"]})
    return cleaned

def unwrap_single(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value

def parse_json_from_text(text: str) -> Optional[Dict]:
    candidate = text.strip()
    if not candidate:
        return None
    if "\n{" in candidate:
        candidate = "{" + candidate.split("\n{", 1)[1].strip()
    elif not candidate.startswith("{"):
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def get_nested(obj: Dict, path: Tuple[str, ...]):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def normalize_scalar(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Keep numeric severity as string representation for equality,
        # but return numeric separately for regression metrics.
        return str(value)
    return str(value)


def to_set(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, (tuple, set)):
        items = list(value)
    else:
        items = [value]
    cleaned = []
    for item in items:
        if item is None:
            continue
        cleaned.append(str(item))
    return cleaned



def main() -> None:
    parsed = parse_args()
    parsed.output_dir.mkdir(parents=True, exist_ok=True)

    args = BaseArguments.from_pretrained(str(parsed.run_dir))
    model, tokenizer = get_model_tokenizer(args.model)
    model = Swift.from_pretrained(model, str(parsed.adapter_dir))
    template = get_template(args.template, tokenizer, args.system)
    engine = PtEngine.from_model_template(model, template)

    data = load_dataset(parsed.val_file, limit=parsed.limit)
    config = RequestConfig(max_tokens=parsed.max_new_tokens, temperature=0.0)

    scalar_stats = {field: {"correct": 0, "total": 0} for field in SCALAR_FIELDS}
    list_stats = {field: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for field in LIST_FIELDS}
    severity_true: List[float] = []
    severity_pred: List[float] = []
    exact_matches = 0

    sample_reports: List[Dict] = []
    failures: List[Dict] = []

    batch_requests: List[InferRequest] = []
    batch_meta: List[Dict] = []

    def flush_batch():
        nonlocal batch_requests, batch_meta, exact_matches
        if not batch_requests:
            return
        start_time = time.time()
        responses = engine.infer(batch_requests, config)
        elapsed = time.time() - start_time
        per_latency = elapsed / len(batch_requests)

        for meta, response in zip(batch_meta, responses):
            pred_json = parse_json_from_text(response.choices[0].message.content)
            if pred_json is None:
                failures.append({"index": meta["index"], "reason": "Prediction JSON parse failure"})
                continue

            gt_json = meta["gt_json"]
            sample_report = {
                "index": meta["index"],
                "image": meta["image"],
                "latency_sec": per_latency,
                "correct_fields": [],
                "incorrect_fields": {},
                "scalar_values": {},
                "list_values": {},
            }

            is_exact = True

            for field in SCALAR_FIELDS:
                gt_value = unwrap_single(get_nested(gt_json, field))
                pred_value = unwrap_single(get_nested(pred_json, field))
                if gt_value is None:
                    continue
                scalar_stats[field]["total"] += 1

                sample_report["scalar_values"][".".join(field)] = {
                    "pred": pred_value,
                    "gt": gt_value,
                }

                gt_norm = normalize_scalar(gt_value)
                pred_norm = normalize_scalar(pred_value)
                if gt_norm == pred_norm:
                    scalar_stats[field]["correct"] += 1
                    sample_report["correct_fields"].append(".".join(field))
                else:
                    is_exact = False
                    sample_report["incorrect_fields"][".".join(field)] = {
                        "pred": pred_value,
                        "gt": gt_value,
                    }

                if field == ("Severity",) and gt_value is not None:
                    try:
                        severity_true.append(float(gt_value))
                        severity_pred.append(float(pred_value))
                    except (TypeError, ValueError):
                        pass

            for field in LIST_FIELDS:
                gt_value = get_nested(gt_json, field)
                pred_value = get_nested(pred_json, field)
                if gt_value is None:
                    continue

                gt_set = set(to_set(gt_value))
                pred_set = set(to_set(pred_value))

                sample_report["list_values"][".".join(field)] = {
                    "pred": sorted(pred_set),
                    "gt": sorted(gt_set),
                }

                tp = len(gt_set & pred_set)
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)

                list_stats[field]["tp"] += tp
                list_stats[field]["fp"] += fp
                list_stats[field]["fn"] += fn
                list_stats[field]["support"] += len(gt_set)

                if fp == 0 and fn == 0:
                    sample_report["correct_fields"].append(".".join(field))
                else:
                    is_exact = False
                    sample_report["incorrect_fields"][".".join(field)] = {
                        "pred": sorted(pred_set),
                        "gt": sorted(gt_set),
                    }

            if is_exact:
                exact_matches += 1
            sample_reports.append(sample_report)

        batch_requests = []
        batch_meta = []

    for idx, item in enumerate(data):
        messages = strip_loss(item.get("messages", []))
        if not messages:
            failures.append({"index": idx, "reason": "Missing messages"})
            continue
        image_entries = item.get("images", [])
        image_paths: List[str] = []
        for entry in image_entries:
            if isinstance(entry, str):
                image_paths.append(entry)
            elif isinstance(entry, dict):
                image_paths.append(entry.get("path", ""))
        gt_json = parse_json_from_text(messages[-1]["content"])
        if gt_json is None:
            failures.append({"index": idx, "reason": "Ground-truth JSON parse failure"})
            continue

        request = InferRequest(messages=messages, images=image_paths)
        batch_requests.append(request)
        batch_meta.append({"index": idx, "image": image_paths[0] if image_paths else "", "gt_json": gt_json})

        if len(batch_requests) >= parsed.batch_size:
            flush_batch()

    flush_batch()

    total_samples = len(sample_reports)

    # Aggregate metrics
    scalar_metrics = []
    for field, stats in scalar_stats.items():
        total = stats["total"]
        accuracy = stats["correct"] / total if total else None
        scalar_metrics.append({
            "field": ".".join(field),
            "total": total,
            "accuracy": accuracy,
        })

    list_metrics = []
    for field, stats in list_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        if precision is not None and recall is not None and (precision + recall):
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        list_metrics.append({
            "field": ".".join(field),
            "support": stats["support"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": fp,
            "false_negatives": fn,
        })

    severity_metrics = {}
    if severity_true and severity_pred:
        diffs = [p - t for p, t in zip(severity_pred, severity_true)]
        mae = sum(abs(d) for d in diffs) / len(diffs)
        rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        severity_metrics = {
            "mae": mae,
            "rmse": rmse,
        }

    summary = {
        "total_samples": total_samples,
        "failures": failures,
        "exact_match_rate": exact_matches / total_samples if total_samples else None,
        "scalar_metrics": scalar_metrics,
        "list_metrics": list_metrics,
        "severity_metrics": severity_metrics,
        "sample_reports": sample_reports,
    }

    (parsed.output_dir / "structured_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Processed {total_samples} samples")
    print(f"Exact match rate: {summary['exact_match_rate']:.4f}" if total_samples else "No samples.")
    print(f"Failures: {len(failures)} (details saved to structured_eval_summary.json)")

    print("\nScalar field accuracy:")
    for metric in scalar_metrics:
        if metric["total"]:
            print(f"  {metric['field']}: {metric['accuracy']:.4f} (n={metric['total']})")

    print("\nList field precision/recall/F1:")
    for metric in list_metrics:
        if metric["support"]:
            precision = metric['precision']
            recall = metric['recall']
            f1 = metric['f1']
            print(
                f"  {metric['field']}: P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
                f"(FP={metric['false_positives']}, FN={metric['false_negatives']})"
            )

    if severity_metrics:
        print("\nSeverity regression metrics:")
        print(f"  MAE: {severity_metrics['mae']:.4f}")
        print(f"  RMSE: {severity_metrics['rmse']:.4f}")

    print("\nExamples of mismatches:")
    mismatches = [r for r in sample_reports if r["incorrect_fields"]]
    for report in mismatches[:5]:
        print(f"- Image: {report['image']}")
        for field, values in report["incorrect_fields"].items():
            print(f"    {field}: pred={values['pred']} | gt={values['gt']}")
        print()


if __name__ == "__main__":
    main()

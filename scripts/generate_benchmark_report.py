#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluate_carscenes_predictions import (
    evaluate_list_field,
    evaluate_scalar_field,
    load_jsonl,
    load_slices,
    severity_metrics,
    slice_metrics,
)


HEADLINE_SCALAR_FIELDS = [
    "Scene",
    "TimeOfDay",
    "Weather",
    "Vehicles.TotalNumber",
]
HEADLINE_LIST_FIELDS = [
    "Pedestrians",
]
ALL_SCALAR_FIELDS = [
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
ALL_LIST_FIELDS = [
    "LaneInformation.SpecialLanes",
    "TrafficSigns.TrafficSignsTypes",
    "Vehicles.VehicleTypes",
    "Vehicles.InMotion",
    "Vehicles.States",
    "Pedestrians",
    "Visibility.SpecificImpairments",
]


def resolve_path(base: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base / raw_path).resolve()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def format_float(value: float) -> str:
    return f"{value:.3f}"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def evaluate_run(row: dict[str, str], manifest_dir: Path, slices: list[dict[str, Any]]) -> dict[str, Any]:
    reference_path = resolve_path(manifest_dir, row["reference_path"])
    predictions_path = resolve_path(manifest_dir, row["predictions_path"])
    reference = load_jsonl(reference_path)
    predictions_records = load_jsonl(predictions_path)
    predictions = {record["_id"]: record for record in predictions_records}

    scalar_summary = {field: evaluate_scalar_field(reference, predictions, field) for field in ALL_SCALAR_FIELDS}
    list_summary = {field: evaluate_list_field(reference, predictions, field) for field in ALL_LIST_FIELDS}
    severity = severity_metrics(reference, predictions)
    slices_summary = slice_metrics(reference, predictions, slices)

    matched_ids = sum(1 for record in reference if record["_id"] in predictions)
    coverage = matched_ids / len(reference) if reference else 0.0

    result = {
        "run_id": row["run_id"],
        "display_name": row["display_name"],
        "model_family": row.get("model_family", ""),
        "adaptation": row.get("adaptation", ""),
        "notes": row.get("notes", ""),
        "reference_path": str(reference_path),
        "predictions_path": str(predictions_path),
        "support": len(reference),
        "matched_predictions": matched_ids,
        "coverage": coverage,
        "scalar": scalar_summary,
        "list": list_summary,
        "severity": severity,
        "slices": slices_summary,
        "summary": {
            "scalar_macro_accuracy": average([scalar_summary[field]["accuracy"] for field in ALL_SCALAR_FIELDS]),
            "list_macro_f1": average([list_summary[field]["f1"] for field in ALL_LIST_FIELDS]),
            "scene_accuracy": scalar_summary["Scene"]["accuracy"],
            "timeofday_accuracy": scalar_summary["TimeOfDay"]["accuracy"],
            "weather_accuracy": scalar_summary["Weather"]["accuracy"],
            "vehicle_count_accuracy": scalar_summary["Vehicles.TotalNumber"]["accuracy"],
            "pedestrians_f1": list_summary["Pedestrians"]["f1"],
            "severity_accuracy": severity["accuracy"],
            "severity_qwk": severity["quadratic_weighted_kappa"],
            "severity_mae": severity["mae"],
            "severity_rmse": severity["rmse"],
        },
    }
    return result


def render_markdown_table(results: list[dict[str, Any]]) -> str:
    header = (
        "| Model | Family | Adaptation | Coverage | Scalar Acc | List F1 | Severity Acc | Severity QWK | "
        "Scene | Weather | Veh Count | Ped F1 |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows: list[str] = []
    for result in results:
        summary = result["summary"]
        rows.append(
            "| {display_name} | {model_family} | {adaptation} | {coverage} | {scalar_macro_accuracy} | "
            "{list_macro_f1} | {severity_accuracy} | {severity_qwk} | {scene_accuracy} | {weather_accuracy} | "
            "{vehicle_count_accuracy} | {pedestrians_f1} |".format(
                display_name=result["display_name"],
                model_family=result["model_family"] or "-",
                adaptation=result["adaptation"] or "-",
                coverage=format_pct(result["coverage"]),
                scalar_macro_accuracy=format_pct(summary["scalar_macro_accuracy"]),
                list_macro_f1=format_pct(summary["list_macro_f1"]),
                severity_accuracy=format_pct(summary["severity_accuracy"]),
                severity_qwk=format_float(summary["severity_qwk"]),
                scene_accuracy=format_pct(summary["scene_accuracy"]),
                weather_accuracy=format_pct(summary["weather_accuracy"]),
                vehicle_count_accuracy=format_pct(summary["vehicle_count_accuracy"]),
                pedestrians_f1=format_pct(summary["pedestrians_f1"]),
            )
        )
    return header + "\n" + "\n".join(rows) + "\n"


def render_latex_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{CARScenes benchmark summary on the specified evaluation split.}",
        "    \\label{tab:carscenes_benchmark_results}",
        "    \\begin{tabular}{lcccccccc}",
        "        \\toprule",
        "        Model & Coverage & Scalar Acc & List F1 & Sev. Acc & Sev. QWK & Scene & Weather & Ped. F1 \\\\",
        "        \\midrule",
    ]
    for result in results:
        summary = result["summary"]
        lines.append(
            "        {display_name} & {coverage} & {scalar_macro_accuracy} & {list_macro_f1} & "
            "{severity_accuracy} & {severity_qwk} & {scene_accuracy} & {weather_accuracy} & {pedestrians_f1} \\\\".format(
                display_name=latex_escape(result["display_name"]),
                coverage=format_pct(result["coverage"]),
                scalar_macro_accuracy=format_pct(summary["scalar_macro_accuracy"]),
                list_macro_f1=format_pct(summary["list_macro_f1"]),
                severity_accuracy=format_pct(summary["severity_accuracy"]),
                severity_qwk=format_float(summary["severity_qwk"]),
                scene_accuracy=format_pct(summary["scene_accuracy"]),
                weather_accuracy=format_pct(summary["weather_accuracy"]),
                pedestrians_f1=format_pct(summary["pedestrians_f1"]),
            )
        )
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "display_name",
        "model_family",
        "adaptation",
        "support",
        "matched_predictions",
        "coverage",
        "scalar_macro_accuracy",
        "list_macro_f1",
        "scene_accuracy",
        "timeofday_accuracy",
        "weather_accuracy",
        "vehicle_count_accuracy",
        "pedestrians_f1",
        "severity_accuracy",
        "severity_qwk",
        "severity_mae",
        "severity_rmse",
        "notes",
        "predictions_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "run_id": result["run_id"],
                "display_name": result["display_name"],
                "model_family": result["model_family"],
                "adaptation": result["adaptation"],
                "support": result["support"],
                "matched_predictions": result["matched_predictions"],
                "coverage": result["coverage"],
                "notes": result["notes"],
                "predictions_path": result["predictions_path"],
            }
            row.update(result["summary"])
            writer.writerow(row)


def write_slice_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "display_name",
        "slice_name",
        "count",
        "severity_accuracy",
        "severity_qwk",
        "scene_accuracy",
        "weather_accuracy",
        "vehicle_count_accuracy",
        "pedestrians_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for slice_name, metrics in result["slices"].items():
                writer.writerow(
                    {
                        "run_id": result["run_id"],
                        "display_name": result["display_name"],
                        "slice_name": slice_name,
                        "count": metrics["count"],
                        "severity_accuracy": metrics["severity"]["accuracy"],
                        "severity_qwk": metrics["severity"]["quadratic_weighted_kappa"],
                        "scene_accuracy": metrics["Scene"]["accuracy"],
                        "weather_accuracy": metrics["Weather"]["accuracy"],
                        "vehicle_count_accuracy": metrics["Vehicles.TotalNumber"]["accuracy"],
                        "pedestrians_f1": metrics["Pedestrians"]["f1"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready CARScenes benchmark result tables.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest listing benchmark runs and prediction files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for JSON, CSV, Markdown, and LaTeX report outputs.",
    )
    parser.add_argument(
        "--slices",
        default="configs/carscenes_benchmark_slices.json",
        help="JSON slice configuration file.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    slices = load_slices(resolve_path(Path.cwd(), args.slices))

    manifest_rows = load_manifest(manifest_path)
    if not manifest_rows:
        raise SystemExit("manifest is empty")

    results = [evaluate_run(row, manifest_dir, slices) for row in manifest_rows]

    (output_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(output_dir / "benchmark_summary.csv", results)
    write_slice_csv(output_dir / "benchmark_slices.csv", results)
    (output_dir / "benchmark_table.md").write_text(render_markdown_table(results), encoding="utf-8")
    (output_dir / "benchmark_table.tex").write_text(render_latex_table(results), encoding="utf-8")

    print(f"wrote benchmark report for {len(results)} runs to {output_dir}")


if __name__ == "__main__":
    main()

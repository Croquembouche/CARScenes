#!/usr/bin/env python3
"""Generate confusion-matrix heatmaps for scalar fields."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RUN_DIR = Path("output/qwen2vl_lora/v3-20251009-143039")
SUMMARY_PATH = RUN_DIR / "structured_eval" / "structured_eval_summary.json"
PLOTS_DIR = RUN_DIR / "structured_eval" / "figures"

def gather_scalar_fields(summary) -> list:
    fields = set()
    for sample in summary["sample_reports"]:
        for key in sample.get("scalar_values", {}):
            fields.add(key)
    return sorted(fields)


def gather_pairs(summary, field_name: str):
    pairs = []
    for sample in summary["sample_reports"]:
        scalar_values = sample.get("scalar_values", {})
        if field_name in scalar_values:
            values = scalar_values[field_name]
            pairs.append((str(values["gt"]), str(values["pred"])))
    return pairs


def plot_confusion(field: str, pairs):
    if not pairs:
        return
    df = pd.DataFrame(pairs, columns=["GroundTruth", "Predicted"])
    matrix = pd.crosstab(df["GroundTruth"], df["Predicted"], normalize="index")
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"{field} Confusion Matrix")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{field.replace('.', '_')}_confusion.png", dpi=200)
    plt.close()


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    target_fields = gather_scalar_fields(summary)
    for field in target_fields:
        pairs = gather_pairs(summary, field)
        plot_confusion(field, pairs)
    print(f"Confusion matrices saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

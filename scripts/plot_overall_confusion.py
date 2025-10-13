#!/usr/bin/env python3
"""Create a single confusion matrix aggregating all scalar fields."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RUN_DIR = Path("output/qwen2vl_lora/v3-20251009-143039")
SUMMARY_PATH = RUN_DIR / "structured_eval" / "structured_eval_summary.json"
PLOTS_DIR = RUN_DIR / "structured_eval" / "figures"


def build_pairs(summary):
    gt_labels = []
    pred_labels = []
    for sample in summary["sample_reports"]:
        scalar_values = sample.get("scalar_values", {})
        for field, values in scalar_values.items():
            gt = values.get("gt")
            pred = values.get("pred")
            gt_labels.append(f"{field}={gt}")
            pred_labels.append(f"{field}={pred}")
    return gt_labels, pred_labels


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    gt_labels, pred_labels = build_pairs(summary)
    df = pd.DataFrame({"GT": gt_labels, "Pred": pred_labels})

    if df.empty:
        print("No data available to plot overall confusion matrix.")
        return

    matrix = pd.crosstab(df["GT"], df["Pred"])
    plt.figure(figsize=(max(8, matrix.shape[0] * 0.5), max(6, matrix.shape[1] * 0.5)))
    sns.heatmap(matrix, annot=False, cmap="Blues")
    plt.title("Aggregated Confusion Matrix (All Scalar Fields)")
    plt.ylabel("Ground Truth (Field=Value)")
    plt.xlabel("Predicted (Field=Value)")
    plt.tight_layout()
    output_path = PLOTS_DIR / "overall_scalar_confusion.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved aggregated confusion matrix to {output_path}")


if __name__ == "__main__":
    main()

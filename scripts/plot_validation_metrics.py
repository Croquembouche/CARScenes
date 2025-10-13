#!/usr/bin/env python3
"""Visualize structured evaluation metrics from evaluate_validation_set.py."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUN_DIR = Path("output/qwen2vl_lora/v3-20251009-143039")
SUMMARY_PATH = RUN_DIR / "structured_eval" / "structured_eval_summary.json"
PLOTS_DIR = RUN_DIR / "structured_eval" / "figures"


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    scalar_df = pd.DataFrame(summary["scalar_metrics"])
    list_df = pd.DataFrame(summary["list_metrics"])

    # Bar plot: scalar field accuracy
    plt.figure(figsize=(10, 4))
    scalar_df = scalar_df.sort_values("accuracy", ascending=False)
    plt.bar(scalar_df["field"], scalar_df["accuracy"], color="#3f8efc")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Scalar Field Accuracy (Validation)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scalar_accuracy.png", dpi=200)
    plt.close()

    # Bar plot: list field F1
    plt.figure(figsize=(10, 4))
    list_df = list_df.sort_values("f1", ascending=False)
    plt.bar(list_df["field"], list_df["f1"], color="#fcb03f")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("F1 Score")
    plt.title("List Field F1 Scores (Validation)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "list_f1.png", dpi=200)
    plt.close()

    # Confusion-style summary for scalar fields (correct vs incorrect)
    plt.figure(figsize=(10, 4))
    scalar_df["incorrect"] = scalar_df["total"] - scalar_df["accuracy"] * scalar_df["total"]
    scalar_df_long = pd.melt(
        scalar_df,
        id_vars=["field"],
        value_vars=["accuracy", "incorrect"],
        var_name="type",
        value_name="count",
    )
    plt.bar(
        scalar_df_long["field"],
        scalar_df_long["count"],
        color=["#4caf50" if t == "accuracy" else "#e53935" for t in scalar_df_long["type"]],
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Scalar Field Correct vs Incorrect Counts")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scalar_counts.png", dpi=200)
    plt.close()

    print(f"Figures saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

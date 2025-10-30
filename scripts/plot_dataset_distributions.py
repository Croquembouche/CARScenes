#!/usr/bin/env python3
"""Generate dataset-wide attribute and severity distribution figures."""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_FILES = [Path('dataset/train_qwen.jsonl')]
OUTPUT_DIR = Path('paper/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

scene_counter = Counter()
time_counter = Counter()
severity_values = []

def extract_scalar(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value

for file in DATA_FILES:
    with file.open('r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            messages = row.get('messages', [])
            if not messages:
                continue
            text = messages[-1]['content']
            if '\n{' in text:
                json_str = '{' + text.split('\n{', 1)[1]
            else:
                json_str = text
            try:
                label = json.loads(json_str)
            except json.JSONDecodeError:
                continue
            scene = extract_scalar(label.get('Scene'))
            if scene:
                scene_counter[scene] += 1
            tod = extract_scalar(label.get('TimeOfDay'))
            if tod:
                time_counter[tod] += 1
            severity = extract_scalar(label.get('Severity'))
            if isinstance(severity, (int, float)):
                severity_values.append(severity)

# Scene distribution (top categories + "Other")
total_scenes = sum(scene_counter.values())
if total_scenes == 0:
    raise SystemExit('No scene labels found.')
top_scenes = scene_counter.most_common(10)
other_count = total_scenes - sum(count for _, count in top_scenes)
labels = [name for name, _ in top_scenes]
counts = [count for _, count in top_scenes]
if other_count > 0:
    labels.append('Other')
    counts.append(other_count)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
ax_scene, ax_time = axes
ax_scene.bar(labels, counts, color='#3f8efc')
ax_scene.set_ylabel('Frames')
ax_scene.set_title('Scene distribution (top categories)')
ax_scene.set_xticklabels(labels, rotation=30, ha='right')
time_labels = ['Daytime', 'Dusk/Dawn', 'Nighttime']
time_counts = [time_counter.get(lbl, 0) for lbl in time_labels]
ax_time.bar(time_labels, time_counts, color='#fcb03f')
ax_time.set_title('Time-of-day distribution')
ax_time.set_xticklabels(time_labels, rotation=20, ha='right')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'dataset_attribute_distribution.png', dpi=200)
plt.close(fig)

# Severity histogram
if severity_values:
    severity_fig, ax_sev = plt.subplots(figsize=(6, 4))
    bins = np.arange(0.5, 10.6, 1.0)
    ax_sev.hist(severity_values, bins=bins, edgecolor='black', color='#7cc36a')
    ax_sev.set_xlabel('Severity score')
    ax_sev.set_ylabel('Frames')
    ax_sev.set_title('Severity distribution (all frames)')
    ax_sev.set_xticks(range(1, 11))
    severity_fig.tight_layout()
    severity_fig.savefig(OUTPUT_DIR / 'dataset_severity_histogram.png', dpi=200)
    plt.close(severity_fig)
else:
    print('Warning: no severity values found.')

print('Saved figures to', OUTPUT_DIR)

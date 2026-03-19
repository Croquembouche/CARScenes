# CARScenes Independent Spot-Check Protocol

Use this for the second-pass human review on `gold-agreement-25`.

## Goal
- Have a second annotator label the `gold-agreement-25` subset independently from the released labels.
- Compare that second pass against the released benchmark records.
- Adjudicate disagreements that matter for the paper.

## Reviewer Instructions
- Do not open the released CARScenes labels while annotating.
- Use only the image, the CARScenes schema, and the label definitions.
- Mark each record complete only when every field has been reviewed.
- Use the notes box for ambiguous cases or fields that are not inferable from a single frame.

## Recommended Commands
```bash
python scripts/carscenes_spotcheck_app.py \
  --split-jsonl release/carscenes-v1/splits/gold-agreement-25.jsonl \
  --image-root dataset \
  --output audit_outputs/independent_spotcheck/reviewer2_gold_agreement_25.jsonl \
  --reviewer reviewer-2 \
  --prefill blank
```

Then score the agreement:

```bash
python scripts/score_spotcheck_agreement.py \
  --reference release/carscenes-v1/splits/gold-agreement-25.jsonl \
  --reviewed audit_outputs/independent_spotcheck/reviewer2_gold_agreement_25.jsonl \
  --output-dir audit_outputs/independent_spotcheck/report_reviewer2
```

## What To Report In The Paper
- Reviewer coverage on the 25-image subset.
- Per-field exact match for scalar fields.
- Per-field precision, recall, and F1 for list fields.
- Severity accuracy, MAE, RMSE, and quadratic weighted kappa.
- A short adjudication summary for meaningful disagreements.

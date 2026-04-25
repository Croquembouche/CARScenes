# NeurIPS 2026 Submission Bundle

This note lists the primary CARScenes artifacts needed for a NeurIPS 2026 Evaluations and Datasets submission.

## Paper
- Submission PDF: `paper/output/CARScenes_NIPS.pdf`
- Paper source: `paper/CARScenes_NIPS.tex`
- Official style file vendored locally: `paper/neurips_2026.sty`
- Official NeurIPS template source: `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`

## Public Artifact URLs
- GitHub repository: `https://github.com/Croquembouche/CARScenes`
- Hugging Face dataset landing page: `https://huggingface.co/datasets/williamhe0712/CARScenes`

## Croissant Metadata
- Local Croissant file: `release/carscenes-v1/croissant.json`
- Public raw Croissant URL: `https://raw.githubusercontent.com/Croquembouche/CARScenes/main/release/carscenes-v1/croissant.json`

## Canonical Release Files
- Records: `release/carscenes-v1/data/carscenes_v1_records.jsonl`
- Schema: `release/carscenes-v1/schema/carscenes_v1_schema.json`
- Train split: `release/carscenes-v1/splits/train.jsonl`
- Dev split: `release/carscenes-v1/splits/dev-500.jsonl`
- Held-out benchmark split: `release/carscenes-v1/splits/gold-test-100.jsonl`
- Agreement subset: `release/carscenes-v1/splits/gold-agreement-25.jsonl`
- Release readme: `release/carscenes-v1/README.md`
- License matrix: `release/carscenes-v1/LICENSES.md`

## Submission Notes
- CARScenes is released as an annotation-only benchmark; upstream images are not redistributed.
- The paper includes the official NeurIPS checklist format in `paper/sections/6_checklist.tex`.
- Before OpenReview submission, push or upload the regenerated `release/carscenes-v1/` directory so every Croissant `contentUrl` is publicly fetchable and hash-stable.
- Verify the public URLs with:

```bash
python3 scripts/check_public_croissant_urls.py \
  --croissant release/carscenes-v1/croissant.json
```

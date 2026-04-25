# CARScenes

CARScenes is an annotation-only dataset and benchmark for structured scene understanding and failure analysis in autonomous driving.

## Dataset Summary
- Version: `carscenes-v1`
- Records: `5,192`
- Sources: Argoverse1, Cityscapes, KITTI, nuScenes
- Release type: annotations, schema, split manifests, benchmark tooling, Croissant metadata

## Public Links
- GitHub: `https://github.com/Croquembouche/CARScenes`
- Paper PDF: `https://github.com/Croquembouche/CARScenes/tree/main/paper/output`

## What Is Included
- Canonical JSONL annotations
- Machine-readable schema
- Fixed split manifests
- Gold benchmark split and agreement subset manifests
- Croissant metadata
- Evaluation and validation scripts

## What Is Not Included
- Upstream source images are not redistributed
- Users must obtain the original images under the original dataset licenses and terms

## Source Pool Scope
- The released universe is the pre-annotation candidate pools stored in the CARScenes release artifacts, not the full upstream corpora.
- Those candidate pools were created from downsampled / decimated front-camera subsets before annotation.
- They are not exhaustive copies of the source datasets and are not intended to be statistically representative samples.
- The exact historical decimation recipe is not claimed as part of the benchmark specification.

## Splits
- `train`: 4,592 records
- `dev-500`: 500 records
- `gold-test-100`: 100 records
- `gold-agreement-25`: 25 records

## License
CARScenes annotations, schema, split manifests, and metadata are released under `CC BY 4.0`.

This license does not cover source images from Argoverse1, Cityscapes, KITTI, or nuScenes, which remain under their original licenses.

## Citation
```bibtex
@misc{he2026carscenes,
  title={CARScenes: A Benchmark for Severity-Aware Scene Understanding in Autonomous Driving},
  author={Yuankai He and Weisong Shi},
  year={2026},
  url={https://huggingface.co/datasets/williamhe0712/CARScenes}
}
```

## Benchmark Use
- Use `dev-500` for prompt selection or pilot calibration.
- Use `gold-test-100` for headline model reporting.
- Use `gold-agreement-25` only for independent second-pass agreement analysis.

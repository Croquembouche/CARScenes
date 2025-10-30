# UDriveVLMDataset

This repository packages multimodal driving-scene data into training-ready JSONL
files for fine-tuning visual-language models (VLMs) such as
`qwen2-vl-2b-instruct` using the
[ms-swift](https://github.com/modelscope/ms-swift) toolkit. It ships with a
converter script that pairs driving images with structured scene analyses and
creates conversational supervision samples that include both natural-language
summaries and the original JSON annotations.

## Project Layout

- `dataset/` – Source data with split folders (e.g. `train/images` and
  `train/labels`) plus the generated JSONL outputs.
- `configs/qwen_dataset.yaml` – Default configuration for running the converter.
- `utils/qwen_custom_dataset.py` – Core script that builds the ms-swift friendly
  dataset.
- `scripts/train_qwen2vl.sh` – Helper for LoRA fine-tuning with ms-swift.
- `scripts/infer_qwen2vl_image.py` – Batched inference utility that uses the
  fine-tuned checkpoint.

## Environment Setup

1. Create a Python environment (3.8+; examples use 3.10):
   ```bash
   conda create -n ms-swift python=3.10 -y
   conda activate ms-swift
   ```
2. Install ms-swift from the bundled checkout:
   ```bash
   cd ms-swift
   pip install -e .
   cd ..
   ```
3. Install dataset utilities:
   ```bash
   pip install pyyaml
   ```

## Prerequisites

- CUDA-capable GPUs with sufficient VRAM (tested on 4×24 GiB cards).
- Driving dataset arranged as `dataset/train/images` and `dataset/train/labels`
  with JSON labels matching `dataset/Overall.json`.
- Adequate disk space for generated JSONL files and ms-swift checkpoints.

## Running the Converter

The script reads its settings (input paths, output location, prompts, etc.)
from `configs/qwen_dataset.yaml`. You can run it with no arguments to produce
the full training JSONL:

```bash
python3 utils/qwen_custom_dataset.py
```

Key configuration fields:

- `dataset_dir` – Folder containing `images/` and `labels/` subdirectories.
- `output` – Destination JSONL file.
- `relative_paths` – Set `true` to store image paths relative to
  `dataset_dir`; `false` keeps absolute paths.
- `limit` – Optional sample cap (useful for smoke tests or validation splits).
- `system_prompt` / `user_prompt` – Prompts injected into each conversation.
- `debug` – When `true`, prints dataset diagnostics and previews the first N
  image/label pairs.
- `append_label_json` – When `true`, appends the raw label JSON after the
  narrative summary in each assistant response.

### Overriding Config Values Temporarily

You can adjust specific settings at runtime without editing the YAML:

```bash
python3 utils/qwen_custom_dataset.py --limit 100 --debug
```

This example processes only 100 samples and enables verbose logging.

## Output Format

Each JSONL line includes:

- `messages`: system, user, and assistant turns formatted for ms-swift SFT.
  The assistant message combines a natural-language summary of the scene and
  the original structured label JSON.
- `images`: list containing the associated image path (absolute or relative).

Sample entry:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "Natural-language summary...\n{ \"Scene\": \"Urban\", ... }"}
  ],
  "images": ["dataset/train/images/Argoverse1/example.jpg"]
}
```

The resulting JSONL can be supplied directly to `swift train` or other
ms-swift workflows for VLM fine-tuning.

## Next Steps

- Adjust prompts or formatting in `configs/qwen_dataset.yaml` to match your
  training objectives.
- Split the output JSONL into train/validation sets if needed.
- Integrate the generated dataset into an ms-swift training command, e.g.:

```bash
swift train \
  --model_type qwen2-vl-2b-instruct \
  --train_dataset dataset/train_qwen.jsonl \
  [other-options...]
```

Feel free to extend the utilities for other annotation schemas or VLM models.

## Fine-tuning with ms-swift

Once `train_qwen.jsonl` is ready you can launch LoRA fine-tuning through the
helper script in `scripts/train_qwen2vl.sh`. The script wraps the `swift sft`
CLI and exposes key hyperparameters via environment variables.

1. From the repository root:
   ```bash
   chmod +x scripts/train_qwen2vl.sh    # one-time
   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   MAX_PIXELS=1003520 \
   MAX_LENGTH=4096 \
   scripts/train_qwen2vl.sh
   ```
   The script downloads the base `Qwen/Qwen2-VL-2B-Instruct` model, applies LoRA
   tuning, and logs progress to `output/qwen2vl_lora/...`.
2. Override key settings as needed before the command:
   - `DATASET_PATH` – Path to the generated JSONL (defaults to
     `dataset/train_qwen.jsonl`).
   - `OUTPUT_DIR` – Checkpoint destination (defaults to `output/qwen2vl_lora`).
   - `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, `GRADIENT_ACCUMULATION_STEPS`, etc.
   - `FREEZE_VIT=false` if you also want to fine-tune the visual encoder.
   - `SAVE_STEPS` / `EVAL_STEPS` to adjust checkpoint cadence.

By default the script enables LoRA on the language model while freezing the
visual encoder, uses bfloat16, and reserves 2% of the data for validation via
`--split_dataset_ratio`.

## Inference

After training, run inference across a folder of test images with
`scripts/infer_qwen2vl_image.py`. The script loads the LoRA checkpoint (defaults
to the latest under `output/qwen2vl_lora/.../checkpoint-1911`), applies the
original prompts from `configs/qwen_dataset.yaml`, and writes outputs per image.

```bash
python scripts/infer_qwen2vl_image.py
```

- Input folder: `dataset/test/` (customize by editing
  `IMAGE_DIR` inside the script).
- Output: `output/qwen2vl_lora/<run>/inference_output/<image-name>.txt` contains
  the natural-language summary; `<image-name>.json` holds the structured
  analysis in the training schema. Console logs also show inference latency per
  frame.

To point at a different LoRA checkpoint or dataset, adjust `RUN_DIR`,
`ADAPTER_DIR`, or `IMAGE_DIR` at the top of the script before running.


Annotations & schema: CC BY 4.0 © 2025 [the CAR Lab @ UD].

Code: MIT © 2025 [the CAR Lab @ UD].

Images:

Cityscapes © Cityscapes authors, no redistribution; see license. 
cityscapes-dataset.com

KITTI © KITTI authors, CC BY-NC-SA 3.0. 
cvlibs.net

Argoverse © Argo AI, CC BY-NC-SA 4.0 (+ privacy). 
argoverse.github.io

nuScenes © Motional, non-commercial; commercial license available. 
nuscenes.org

Note: This is practical guidance, not legal advice. When in doubt (esp. for nuScenes redistribution), email the dataset owners and keep their response on file.
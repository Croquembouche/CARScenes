#!/usr/bin/env python3
"""Run batched image inference using the fine-tuned Qwen2-VL LoRA checkpoint."""

import json
import os
import time
from pathlib import Path

from swift.llm import (
    BaseArguments,
    InferRequest,
    PtEngine,
    RequestConfig,
    get_model_tokenizer,
    get_template,
)
from swift.tuners import Swift

import yaml

# Ensure the same GPU selection and vision budget every time.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("MAX_PIXELS", "1003520")

# Paths
RUN_DIR = Path("output/qwen2vl_lora/v3-20251009-143039")
ADAPTER_DIR = RUN_DIR / "checkpoint-1911"
OVERALL_JSON = Path("dataset/Overall.json").resolve()
IMAGE_DIR = Path("dataset/test").resolve()
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    args = BaseArguments.from_pretrained(str(RUN_DIR))

    config_path = Path("configs/qwen_dataset.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with config_path.open(encoding="utf-8") as cfg:
        config_data = yaml.safe_load(cfg) or {}
    user_prompt = config_data.get("user_prompt")
    if not user_prompt:
        raise ValueError("`user_prompt` not found in configs/qwen_dataset.yaml")

    prompt = user_prompt

    model, tokenizer = get_model_tokenizer(args.model)
    model = Swift.from_pretrained(model, str(ADAPTER_DIR))
    template = get_template(args.template, tokenizer, args.system)
    engine = PtEngine.from_model_template(model, template)

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in VALID_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No images found in {IMAGE_DIR}")

    output_dir = RUN_DIR / "inference_output"
    output_dir.mkdir(exist_ok=True)

    config = RequestConfig(max_tokens=512, temperature=0.0)

    for image_path in image_paths:
        request = InferRequest(
            messages=[{"role": "user", "content": prompt}],
            images=[str(image_path)],
        )

        start_time = time.time()
        response = engine.infer([request], config)[0].choices[0].message.content
        elapsed = time.time() - start_time

        print(f"=== {image_path.name} ===")
        print(response)
        print(f"Inference time: {elapsed:.2f} seconds\n")

        paragraph = response.strip()
        structured = None
        if "\n{" in response:
            paragraph, structured = response.split("\n{", 1)
            paragraph = paragraph.strip()
            structured = "{" + structured.strip()

        stem = image_path.stem
        (output_dir / f"{stem}.txt").write_text(paragraph + "\n", encoding="utf-8")
        if structured:
            try:
                parsed = json.loads(structured)
                (output_dir / f"{stem}.json").write_text(
                    json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except json.JSONDecodeError:
                (output_dir / f"{stem}.json").write_text(
                    structured, encoding="utf-8"
                )


if __name__ == "__main__":
    main()

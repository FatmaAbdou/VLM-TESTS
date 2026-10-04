import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "benchmark" / "assets" / "office" / "Images"
OUTPUT_DIR = ROOT / "results" / "qwen2.5-vl-3b-open-ended"
OUTPUT_FILE = OUTPUT_DIR / "open_ended_frame_033.json"

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 3136
MAX_PIXELS = 28 * 28 * 768
MAX_NEW_TOKENS = 180

IMAGE_NAME = (
    "ZED-X_SN45626933_٢٠٢٦-٠٩-٢١_١٥-٠١-١١_"
    "٢٠٢٦-٠٩-٢١_١٥-١٩-٤٠.png"
)

QUESTION = (
    "Describe what is visible in this office image. Identify the main "
    "objects, their approximate locations and spatial relationships, "
    "and any visible paths or obstacles. Only report details supported "
    "by the image."
)

def main():
    image_path = IMAGE_DIR / IMAGE_NAME
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Image: {image_path}")
    print(f"Model: {MODEL_NAME}")
    print(f"Pixel cap: {MAX_PIXELS:,}")
    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print("Loading 4-bit model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {
                "type": "text",
                "text": (
                    "Use only information supported by the image. "
                    "Do not invent details.\n\nQuestion: " + QUESTION
                ),
            },
        ],
    }]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = time.perf_counter() - start
    new_tokens = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], output_ids)
    ]
    answer = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    peak_vram = (
        round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        if torch.cuda.is_available()
        else None
    )

    result = {
        "model": MODEL_NAME,
        "quantization": "4-bit",
        "image": IMAGE_NAME,
        "question": QUESTION,
        "model_answer": answer,
        "latency_seconds": round(latency, 3),
        "peak_gpu_vram_mb": peak_vram,
        "evaluation": {
            "object_coverage": None,
            "attribute_correctness": None,
            "spatial_relationships": None,
            "hallucinations": None,
            "notes": "Manual evaluation pending.",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("OPEN-ENDED TEST ANSWER")
    print("=" * 60)
    print(answer)
    print(f"\nLatency: {latency:.3f}s")
    print(f"Peak allocated GPU VRAM: {peak_vram} MB")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

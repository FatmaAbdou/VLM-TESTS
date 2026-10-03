import argparse
import csv
import json
import statistics
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

from scoring import score_test


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_FILE = ROOT / "benchmark" / "office_v2_zed93.json"
MANIFEST_FILE = ROOT / "benchmark" / "office_manifest.csv"
IMAGE_DIR = ROOT / "benchmark" / "assets" / "office" / "Images"

# Keep full-resolution results separate from the capped run.
RESULTS_DIR = ROOT / "results" / "qwen2.5-vl-3b-zed-fullres-capped-test"


# ============================================================
# Model configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_NEW_TOKENS = 64

# Use the processor's default image preprocessing settings.
# No custom min_pixels or max_pixels are passed.


SYSTEM_PROMPT = """You are being evaluated on visual understanding.

Use only information supported by the provided image(s).

Answer briefly and directly.

Do not invent information that cannot be observed."""


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen2.5-VL-3B on the ZED office benchmark "
            "using default image preprocessing settings."
        )
    )

    parser.add_argument(
        "--test-id",
        type=str,
        default=None,
        help="Run only one benchmark case, e.g. OFFICE-009.",
    )

    return parser.parse_args()


# ============================================================
# Benchmark and manifest loading
# ============================================================

def load_benchmark():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        tests = data.get("tests", data.get("cases", []))
    else:
        tests = data

    if not tests:
        raise ValueError(
            f"No benchmark cases were found in {BENCHMARK_FILE}"
        )

    return tests


def load_manifest():
    manifest = {}

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            manifest[row["frame_id"]] = row["filename"]

    return manifest


# ============================================================
# Image loading
# ============================================================

def get_image_path(frame_id, manifest):
    if frame_id not in manifest:
        raise KeyError(
            f"Frame '{frame_id}' was not found in {MANIFEST_FILE}"
        )

    image_path = IMAGE_DIR / manifest[frame_id]

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image for {frame_id} does not exist:\n{image_path}"
        )

    return image_path


def load_images(input_media, manifest):
    records = []

    for frame_id in input_media:
        image_path = get_image_path(frame_id, manifest)

        # Verify the original image without resizing or modifying it.
        with Image.open(image_path) as image:
            image.verify()

        records.append(
            {
                "frame_id": frame_id,
                "path": image_path,
            }
        )

    return records


# ============================================================
# Prompt construction
# ============================================================

def build_prompt(test):
    question = test["question"]

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


# ============================================================
# Qwen input preparation
# ============================================================

def prepare_inputs(processor, model, image_records, prompt):
    content = []

    for record in image_records:
        content.append(
            {
                "type": "image",
                "image": str(record["path"]),
            }
        )

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    # Important: do not pass custom min_pixels or max_pixels.
    # The processor uses its own default image preprocessing.
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    return inputs


# ============================================================
# GPU memory metrics
# ============================================================

def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_vram_mb():
    if not torch.cuda.is_available():
        return None

    return round(
        torch.cuda.max_memory_allocated() / (1024 ** 2),
        1,
    )


# ============================================================
# Model generation
# ============================================================


def generate_answer(processor, model, inputs):
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]

    token_ids = generated_ids_trimmed[0].tolist()

    print("\n--- TOKEN DIAGNOSTICS ---")
    print("Generated token IDs:", token_ids)
    print(
        "Generated tokens:",
        processor.tokenizer.convert_ids_to_tokens(token_ids),
    )

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print("Decoded answer:", repr(answer))
    print("--- END DIAGNOSTICS ---\n")

    return answer


# ============================================================
# Model loading
# ============================================================

def load_model():
    print()
    print("Loading processor...")
    print("Image preprocessing: capped")
    print("Min pixels: 3136")
    print(f"Max pixels: {28 * 28 * 768}")

    processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=3136,
    max_pixels=28 * 28 * 768,
)

    image_processor = processor.image_processor

    print(
        "Processor default max_pixels:",
        getattr(image_processor, "max_pixels", "Not exposed"),
    )
    print(
        "Processor default min_pixels:",
        getattr(image_processor, "min_pixels", "Not exposed"),
    )

    print()
    print("Loading model with 4-bit quantization...")

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

    return processor, model


# ============================================================
# Warm-up
# ============================================================

def warm_up(processor, model, tests, manifest):
    if not tests:
        return

    test = tests[0]

    print()
    print("Running GPU warm-up...")

    image_records = load_images(
        test["input_media"],
        manifest,
    )

    prompt = build_prompt(test)

    reset_gpu_stats()
    start = time.perf_counter()

    inputs = prepare_inputs(
        processor,
        model,
        image_records,
        prompt,
    )

    generate_answer(
        processor,
        model,
        inputs,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"Warm-up latency: {elapsed:.3f}s")
    print(f"Warm-up peak VRAM: {get_peak_vram_mb()} MB")

    del inputs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Warm-up complete.")


# ============================================================
# Run one test
# ============================================================

def run_test(test, processor, model, manifest):
    test_id = test["test_id"]

    image_records = load_images(
        test["input_media"],
        manifest,
    )

    prompt = build_prompt(test)

    reset_gpu_stats()
    start = time.perf_counter()

    inputs = prepare_inputs(
        processor,
        model,
        image_records,
        prompt,
    )

    answer = generate_answer(
        processor,
        model,
        inputs,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = time.perf_counter() - start
    peak_vram = get_peak_vram_mb()

    # Preserve the existing scoring function's expected field.
    score_test_copy = dict(test)
    score_test_copy["expected_answer"] = test["ground_truth"]

    score = score_test(
        score_test_copy,
        answer,
    )

    result = {
        "test_id": test_id,
        "category": test.get("category"),
        "input_media": test.get("input_media", []),
        "question": test.get("question"),
        "ground_truth": test.get("ground_truth"),
        "accepted_answers": test.get("accepted_answers", []),
        "answer_type": test.get("answer_type"),
        "model_answer": answer,
        "correct": score,
        "latency_seconds": latency,
        "gpu_vram_mb": peak_vram,
        "gpu_utilization_percent": None,
    }

    del inputs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# Metrics
# ============================================================

def calculate_accuracy(results):
    scored = [
        result
        for result in results
        if result["correct"] is not None
    ]

    if not scored:
        return 0.0

    return sum(
        float(result["correct"])
        for result in scored
    ) / len(scored)


def calculate_capability_accuracy(results):
    grouped = {}

    for result in results:
        category = result.get("category", "Unknown")

        if result["correct"] is None:
            continue

        grouped.setdefault(category, []).append(
            float(result["correct"])
        )

    output = {}

    for category, scores in grouped.items():
        correct = sum(scores)
        total = len(scores)

        output[category] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }

    return output


def calculate_latency(results):
    values = [
        result["latency_seconds"]
        for result in results
        if result["latency_seconds"] is not None
    ]

    if not values:
        return {
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "p95_seconds": 0.0,
            "maximum_seconds": 0.0,
        }

    values_sorted = sorted(values)

    p95_index = min(
        len(values_sorted) - 1,
        int(0.95 * len(values_sorted)),
    )

    return {
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": values_sorted[p95_index],
        "maximum_seconds": max(values),
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    print()
    print("=" * 60)
    print("Qwen2.5-VL-3B ZED Full-Resolution Runner")
    print("=" * 60)

    print(f"Benchmark: {BENCHMARK_FILE}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Model: {MODEL_NAME}")
    print("Image preprocessing: default processor settings")

    if torch.cuda.is_available():
        print("Device: CUDA")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "GPU VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )
    else:
        print("Device: CPU")

    tests = load_benchmark()

    if args.test_id:
        tests = [
            test
            for test in tests
            if test["test_id"] == args.test_id
        ]

        if not tests:
            raise ValueError(
                f"Test ID '{args.test_id}' was not found."
            )

    manifest = load_manifest()

    print(f"Loaded {len(tests)} benchmark cases.")

    processor, model = load_model()

    warm_up(
        processor,
        model,
        tests,
        manifest,
    )

    results = []

    print()

    for index, test in enumerate(tests, start=1):
        print(f"[{index}/{len(tests)}] {test['test_id']}")
        print(f"  Category: {test.get('category', 'Unknown')}")

        try:
            result = run_test(
                test,
                processor,
                model,
                manifest,
            )

            results.append(result)

            print(f"  Answer: {result['model_answer']}")
            print(f"  Expected: {result['ground_truth']}")
            print(f"  Correct: {result['correct']}")
            print(f"  Latency: {result['latency_seconds']:.3f}s")
            print(f"  Peak VRAM: {result['gpu_vram_mb']} MB")

        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")

            results.append(
                {
                    "test_id": test["test_id"],
                    "category": test.get("category"),
                    "input_media": test.get("input_media", []),
                    "question": test.get("question"),
                    "ground_truth": test.get("ground_truth"),
                    "accepted_answers": test.get("accepted_answers", []),
                    "answer_type": test.get("answer_type"),
                    "model_answer": None,
                    "correct": None,
                    "latency_seconds": None,
                    "gpu_vram_mb": None,
                    "gpu_utilization_percent": None,
                    "error": str(exc),
                }
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    scored_results = [
        result
        for result in results
        if result["correct"] is not None
    ]

    peak_values = [
        result["gpu_vram_mb"]
        for result in results
        if result["gpu_vram_mb"] is not None
    ]

    summary = {
        "cases": len(results),
        "scored_cases": len(scored_results),
        "accuracy": calculate_accuracy(results),
        "latency": calculate_latency(results),
        "capability_accuracy": calculate_capability_accuracy(results),
        "peak_gpu_vram_mb": max(peak_values) if peak_values else None,
    }

    output = {
        "model": MODEL_NAME,
        "model_variant": "qwen3b-zed-fullres",
        "quantized": True,
        "quantization": "4-bit",
        "image_preprocessing": "processor_defaults",
        "custom_min_pixels": None,
        "custom_max_pixels": None,
        "benchmark_file": str(BENCHMARK_FILE),
        "image_directory": str(IMAGE_DIR),
        "summary": summary,
        "results": results,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = RESULTS_DIR / "results.json"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)

    print(f"Cases: {summary['cases']}")
    print(f"Scored cases: {summary['scored_cases']}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(
        "Mean latency: "
        f"{summary['latency']['mean_seconds']:.3f}s"
    )
    print(
        "Median latency: "
        f"{summary['latency']['median_seconds']:.3f}s"
    )
    print(
        "P95 latency: "
        f"{summary['latency']['p95_seconds']:.3f}s"
    )
    print(f"Peak GPU VRAM: {summary['peak_gpu_vram_mb']} MB")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
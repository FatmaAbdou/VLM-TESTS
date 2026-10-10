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
    Gemma3ForConditionalGeneration,
)

from scoring import score_test


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_FILE = ROOT / "benchmark" / "office_v2_zed93.json"
MANIFEST_FILE = ROOT / "benchmark" / "office_manifest.csv"
IMAGE_DIR = ROOT / "benchmark" / "assets" / "office" / "Images"

# Keep Gemma results separate from every Qwen run.
RESULTS_DIR = ROOT / "results" / "gemma-3-4b-zed-fullres-capped-test"


# ============================================================
# Model configuration
# ============================================================

MODEL_NAME = "google/gemma-3-4b-it"
MAX_NEW_TOKENS = 64

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
            "Run Gemma 3 4B IT on the ZED office benchmark "
            "using the Gemma processor's default image preprocessing."
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
        raise ValueError(f"No benchmark cases were found in {BENCHMARK_FILE}")

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
        raise KeyError(f"Frame '{frame_id}' was not found in {MANIFEST_FILE}")

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

        # Verify that the source image can be opened. Do not resize it here.
        with Image.open(image_path) as image:
            image.verify()

        records.append({"frame_id": frame_id, "path": image_path})

    return records


# ============================================================
# Prompt construction
# ============================================================

def build_prompt(test):
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {test['question']}\n\n"
        "Answer:"
    )


# ============================================================
# Gemma input preparation
# ============================================================

def prepare_inputs(processor, model, image_records, prompt):
    # Gemma 3 uses its own chat template and image processor.
    # Open images as RGB PIL images and pass them in the same order as input_media.
    images = []
    try:
        for record in image_records:
            with Image.open(record["path"]) as image:
                images.append(image.convert("RGB"))

        content = [{"type": "image", "url": image} for image in images]
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move tensor inputs to the model's device. device_map="auto" handles
        # model placement; the model may span more than one device.
        inputs = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        return inputs
    finally:
        for image in images:
            image.close()


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
    return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)


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

    # Gemma's chat-template inputs include the prompt tokens; score/decode only
    # the newly generated answer tokens.
    input_length = inputs["input_ids"].shape[-1]
    generated_ids_trimmed = generated_ids[:, input_length:]

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
    print("Loading Gemma processor...")
    print("Image preprocessing: Gemma processor defaults")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print()
    print("Loading Gemma 3 4B IT with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            else torch.float16
        ),
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
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

    image_records = load_images(test["input_media"], manifest)
    prompt = build_prompt(test)

    reset_gpu_stats()
    start = time.perf_counter()

    inputs = prepare_inputs(processor, model, image_records, prompt)
    generate_answer(processor, model, inputs)

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
    image_records = load_images(test["input_media"], manifest)
    prompt = build_prompt(test)

    reset_gpu_stats()
    start = time.perf_counter()

    inputs = prepare_inputs(processor, model, image_records, prompt)
    answer = generate_answer(processor, model, inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = time.perf_counter() - start
    peak_vram = get_peak_vram_mb()

    # Preserve the scoring function's expected field.
    score_test_copy = dict(test)
    score_test_copy["expected_answer"] = test["ground_truth"]
    score = score_test(score_test_copy, answer)

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
    scored = [result for result in results if result["correct"] is not None]
    if not scored:
        return 0.0
    return sum(float(result["correct"]) for result in scored) / len(scored)


def calculate_capability_accuracy(results):
    grouped = {}
    for result in results:
        category = result.get("category", "Unknown")
        if result["correct"] is None:
            continue
        grouped.setdefault(category, []).append(float(result["correct"]))

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
    p95_index = min(len(values_sorted) - 1, int(0.95 * len(values_sorted)))
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
    print("Gemma 3 4B IT ZED Office Benchmark Runner")
    print("=" * 60)
    print(f"Benchmark: {BENCHMARK_FILE}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Model: {MODEL_NAME}")
    print("Image preprocessing: Gemma processor defaults")

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
        tests = [test for test in tests if test["test_id"] == args.test_id]
        if not tests:
            raise ValueError(f"Test ID '{args.test_id}' was not found.")

    manifest = load_manifest()
    print(f"Loaded {len(tests)} benchmark cases.")

    processor, model = load_model()
    warm_up(processor, model, tests, manifest)

    results = []
    print()

    for index, test in enumerate(tests, start=1):
        print(f"[{index}/{len(tests)}] {test['test_id']}")
        print(f"  Category: {test.get('category', 'Unknown')}")

        try:
            result = run_test(test, processor, model, manifest)
            results.append(result)
            print(f"  Answer: {result['model_answer']}")
            print(f"  Expected: {result['ground_truth']}")
            print(f"  Correct: {result['correct']}")
            print(f"  Latency: {result['latency_seconds']:.3f}s")
            print(f"  Peak VRAM: {result['gpu_vram_mb']} MB")
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            results.append({
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
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    scored_results = [result for result in results if result["correct"] is not None]
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
        "model_variant": "gemma3-4b-zed-default-processor",
        "quantized": True,
        "quantization": "4-bit",
        "image_preprocessing": "gemma_processor_defaults",
        "custom_min_pixels": None,
        "custom_max_pixels": None,
        "benchmark_file": str(BENCHMARK_FILE),
        "image_directory": str(IMAGE_DIR),
        "summary": summary,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "results.json"

    # Never overwrite an earlier result file.
    if output_file.exists():
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = RESULTS_DIR / f"results-{timestamp}.json"
        if output_file.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing results: {output_file}"
            )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"Cases: {summary['cases']}")
    print(f"Scored cases: {summary['scored_cases']}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Mean latency: {summary['latency']['mean_seconds']:.3f}s")
    print(f"Median latency: {summary['latency']['median_seconds']:.3f}s")
    print(f"P95 latency: {summary['latency']['p95_seconds']:.3f}s")
    print(f"Peak GPU VRAM: {summary['peak_gpu_vram_mb']} MB")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

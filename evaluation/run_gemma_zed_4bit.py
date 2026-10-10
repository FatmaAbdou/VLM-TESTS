
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

RESULTS_DIR = ROOT / "results" / "gemma-3-4b-zed-4bit-test"

MODEL_NAME = "google/gemma-3-4b-it"
MAX_NEW_TOKENS = 64

SYSTEM_PROMPT = """You are being evaluated on visual understanding.

Use only information supported by the provided image(s).
Answer briefly and directly.
Do not invent information that cannot be observed."""


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Gemma 3 4B IT ZED benchmark in 4-bit NF4."
    )
    parser.add_argument(
        "--test-id",
        type=str,
        default=None,
        help="Run one test, e.g. OFFICE-009.",
    )
    return parser.parse_args()


# ============================================================
# Load benchmark and manifest
# ============================================================

def load_benchmark():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = data.get("tests", data.get("cases", [])) if isinstance(data, dict) else data

    if not tests:
        raise ValueError(f"No benchmark cases found in {BENCHMARK_FILE}")

    return tests


def load_manifest():
    manifest = {}

    with open(MANIFEST_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"frame_id", "filename"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{MANIFEST_FILE} must contain columns: frame_id, filename"
            )

        for row in reader:
            manifest[row["frame_id"]] = row["filename"]

    return manifest


# ============================================================
# Images and prompt
# ============================================================

def get_image_path(frame_id, manifest):
    if frame_id not in manifest:
        raise KeyError(f"Frame '{frame_id}' was not found in {MANIFEST_FILE}")

    image_path = IMAGE_DIR / manifest[frame_id]

    if not image_path.is_file():
        raise FileNotFoundError(f"Image for {frame_id} does not exist: {image_path}")

    return image_path


def load_images(input_media, manifest):
    records = []

    for frame_id in input_media:
        image_path = get_image_path(frame_id, manifest)

        with Image.open(image_path) as image:
            image.verify()

        records.append({"frame_id": frame_id, "path": image_path})

    return records


def build_prompt(test):
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {test['question']}\n\n"
        "Answer:"
    )


def prepare_inputs(processor, model, image_records, prompt):
    images = []

    try:
        for record in image_records:
            with Image.open(record["path"]) as image:
                images.append(image.convert("RGB"))

        content = [
            {"type": "image", "url": image}
            for image in images
        ]
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_device = model.get_input_embeddings().weight.device

        inputs = {
            key: value.to(input_device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        return inputs

    finally:
        for image in images:
            image.close()


# ============================================================
# GPU metrics
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
# Generation
# ============================================================

def generate_answer(processor, model, inputs, debug=False):
    input_length = inputs["input_ids"].shape[-1]

    pad_token_id = processor.tokenizer.pad_token_id
    eos_token_id = processor.tokenizer.eos_token_id

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    if generated_ids.shape[-1] < input_length:
        raise RuntimeError(
            "Generation returned fewer tokens than the input prompt."
        )

    generated_ids_trimmed = generated_ids[:, input_length:]

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    if debug:
        token_ids = generated_ids_trimmed[0].tolist()
        print("\n--- GENERATION DIAGNOSTICS ---")
        print("Input sequence length:", input_length)
        print("Output sequence length:", generated_ids.shape[-1])
        print("Generated token IDs:", token_ids)
        print(
            "Generated tokens:",
            processor.tokenizer.convert_ids_to_tokens(token_ids),
        )
        print("Decoded answer:", repr(answer))
        print("--- END DIAGNOSTICS ---\n")

    if not answer:
        token_ids = generated_ids_trimmed[0].tolist()
        raise RuntimeError(
            "Gemma generated an empty answer. "
            f"Generated token IDs: {token_ids}. "
            "The case will not be scored."
        )

    return answer


# ============================================================
# Load quantized model
# ============================================================

def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("The 4-bit runner requires a CUDA GPU.")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("\nModel:", MODEL_NAME)
    print("Quantization: 4-bit NF4")
    print("Compute dtype:", compute_dtype)
    print("Double quantization: disabled")
    print("Image preprocessing: Gemma processor defaults")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    
    model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    dtype=compute_dtype,
    device_map="auto",
    )


    model.eval()

    print("Model loaded successfully.")
    print("Padding token ID:", processor.tokenizer.pad_token_id)
    print("EOS token ID:", processor.tokenizer.eos_token_id)

    return processor, model


# ============================================================
# Warm-up
# ============================================================

def warm_up(processor, model, tests, manifest):
    if not tests:
        return

    test = tests[0]
    print("\nRunning GPU warm-up...")

    image_records = load_images(test["input_media"], manifest)
    prompt = build_prompt(test)

    reset_gpu_stats()
    start = time.perf_counter()

    inputs = prepare_inputs(processor, model, image_records, prompt)
    answer = generate_answer(processor, model, inputs, debug=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print("Warm-up answer:", answer)
    print(f"Warm-up latency: {elapsed:.3f}s")
    print(f"Warm-up peak VRAM: {get_peak_vram_mb()} MB")

    del inputs
    torch.cuda.empty_cache()
    print("Warm-up complete.")


# ============================================================
# Run one case
# ============================================================

def run_test(test, processor, model, manifest):
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

    score_test_copy = dict(test)
    score_test_copy["expected_answer"] = test["ground_truth"]
    score = score_test(score_test_copy, answer)

    result = {
        "test_id": test["test_id"],
        "category": test.get("category"),
        "input_media": test.get("input_media", []),
        "question": test.get("question"),
        "ground_truth": test.get("ground_truth"),
        "accepted_answers": test.get("accepted_answers", []),
        "answer_type": test.get("answer_type"),
        "model_answer": answer,
        "correct": score,
        "latency_seconds": round(latency, 6),
        "gpu_vram_mb": peak_vram,
        "gpu_utilization_percent": None,
    }

    del inputs
    torch.cuda.empty_cache()

    return result


# ============================================================
# Summary metrics
# ============================================================

def calculate_accuracy(results):
    scored = [r for r in results if r["correct"] is not None]
    return (
        sum(float(r["correct"]) for r in scored) / len(scored)
        if scored else 0.0
    )


def calculate_capability_accuracy(results):
    grouped = {}

    for result in results:
        category = result.get("category") or "Unknown"
        if result["correct"] is not None:
            grouped.setdefault(category, []).append(float(result["correct"]))

    return {
        category: {
            "correct": sum(scores),
            "total": len(scores),
            "accuracy": sum(scores) / len(scores),
        }
        for category, scores in grouped.items()
    }


def calculate_latency(results):
    values = [
        r["latency_seconds"]
        for r in results
        if r["latency_seconds"] is not None
    ]

    if not values:
        return {
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "p95_seconds": 0.0,
            "maximum_seconds": 0.0,
        }

    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))

    return {
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": ordered[p95_index],
        "maximum_seconds": max(values),
    }


def save_results(output):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / "results.json"

    if output_file.exists():
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = RESULTS_DIR / f"results-{timestamp}.json"

        if output_file.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing results: {output_file}"
            )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output_file


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("Gemma 3 4B IT ZED Office Benchmark - 4-bit NF4")
    print("=" * 60)
    print("Benchmark:", BENCHMARK_FILE)
    print("Manifest:", MANIFEST_FILE)
    print("Image directory:", IMAGE_DIR)
    print("Model:", MODEL_NAME)
    print("Quantization: 4-bit NF4")

    if not BENCHMARK_FILE.is_file():
        raise FileNotFoundError(f"Benchmark not found: {BENCHMARK_FILE}")
    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    print("Device: CUDA")
    print("GPU:", torch.cuda.get_device_name(0))
    total_vram = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    )
    print(f"GPU VRAM: {total_vram:.2f} GB")

    tests = load_benchmark()

    if args.test_id:
        tests = [t for t in tests if t["test_id"] == args.test_id]
        if not tests:
            raise ValueError(f"Test ID '{args.test_id}' was not found.")

    manifest = load_manifest()
    print(f"Loaded {len(tests)} benchmark cases.")

    processor, model = load_model()

    # Warm-up must succeed before benchmark results are recorded.
    warm_up(processor, model, tests, manifest)

    results = []

    for index, test in enumerate(tests, start=1):
        print(f"\n[{index}/{len(tests)}] {test['test_id']}")
        print("  Category:", test.get("category", "Unknown"))

        try:
            result = run_test(test, processor, model, manifest)
            results.append(result)

            print("  Answer:", result["model_answer"])
            print("  Expected:", result["ground_truth"])
            print("  Correct:", result["correct"])
            print(f"  Latency: {result['latency_seconds']:.3f}s")
            print("  Peak VRAM:", result["gpu_vram_mb"], "MB")

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

            torch.cuda.empty_cache()

    scored = [r for r in results if r["correct"] is not None]
    peak_values = [
        r["gpu_vram_mb"] for r in results if r["gpu_vram_mb"] is not None
    ]

    summary = {
        "cases": len(results),
        "scored_cases": len(scored),
        "accuracy": calculate_accuracy(results),
        "latency": calculate_latency(results),
        "capability_accuracy": calculate_capability_accuracy(results),
        "peak_gpu_vram_mb": max(peak_values) if peak_values else None,
    }

    output = {
        "model": MODEL_NAME,
        "model_variant": "gemma3-4b-zed-4bit-nf4",
        "quantized": True,
        "quantization": "4-bit NF4",
        "precision": "4-bit NF4 with BF16/FP16 compute",
        "image_preprocessing": "gemma_processor_defaults",
        "custom_min_pixels": None,
        "custom_max_pixels": None,
        "benchmark_file": str(BENCHMARK_FILE),
        "manifest_file": str(MANIFEST_FILE),
        "image_directory": str(IMAGE_DIR),
        "summary": summary,
        "results": results,
    }

    output_file = save_results(output)

    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print("Cases:", summary["cases"])
    print("Scored cases:", summary["scored_cases"])
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Mean latency: {summary['latency']['mean_seconds']:.3f}s")
    print(f"Median latency: {summary['latency']['median_seconds']:.3f}s")
    print(f"P95 latency: {summary['latency']['p95_seconds']:.3f}s")
    print("Peak GPU VRAM:", summary["peak_gpu_vram_mb"], "MB")
    print("Results saved to:", output_file)


if __name__ == "__main__":
    main()

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

# Keep the 4-bit experiment separate from other Gemma results.
RESULTS_DIR = ROOT / "results" / "gemma-3-4b-zed-4bit-test"


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
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Gemma 3 4B IT with 4-bit NF4 quantization "
            "on the ZED office benchmark."
        )
    )

    parser.add_argument(
        "--test-id",
        type=str,
        default=None,
        help="Run one test, for example OFFICE-009.",
    )

    return parser.parse_args()


# ============================================================
# Load benchmark and manifest
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
            f"No benchmark cases found in {BENCHMARK_FILE}"
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

        required = {"frame_id", "filename"}

        if not reader.fieldnames or not required.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                f"{MANIFEST_FILE} must contain columns: "
                "frame_id, filename"
            )

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

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image for {frame_id} does not exist:\n{image_path}"
        )

    return image_path


def load_images(input_media, manifest):
    records = []

    for frame_id in input_media:
        image_path = get_image_path(frame_id, manifest)

        with Image.open(image_path) as image:
            image.verify()

        records.append({
            "frame_id": frame_id,
            "path": image_path,
        })

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
# Prepare Gemma multimodal inputs
# ============================================================

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

        content.append({
            "type": "text",
            "text": prompt,
        })

        messages = [{
            "role": "user",
            "content": content,
        }]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_device = model.get_input_embeddings().weight.device

        inputs = {
            key: value.to(input_device)
            if isinstance(value, torch.Tensor)
            else value
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

    return round(
        torch.cuda.max_memory_allocated() / (1024 ** 2),
        1,
    )


# ============================================================
# Generate an answer
# ============================================================

def generate_answer(processor, model, inputs, debug=False):
    input_length = inputs["input_ids"].shape[-1]

    pad_token_id = processor.tokenizer.pad_token_id
    eos_token_id = processor.tokenizer.eos_token_id

    if debug:
        print("\n--- GENERATION DIAGNOSTICS ---")
        print("Input token length:", input_length)
        print("Padding token ID:", pad_token_id)
        print("EOS token ID:", eos_token_id)
        print("Input tensor device:", inputs["input_ids"].device)
        print("Input tensor dtype:", inputs["input_ids"].dtype)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "use_cache": True,
    }

    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

        # Prevent the padding token from being generated as answer text.
        # This is a diagnostic safeguard; it does not guarantee that
        # the model will generate a meaningful answer.
        generation_kwargs["bad_words_ids"] = [[pad_token_id]]

    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    with torch.inference_mode():
        generated_ids = model.generate(**generation_kwargs)

    if generated_ids.shape[-1] < input_length:
        raise RuntimeError(
            "Generation returned fewer tokens than the input prompt. "
            f"Input length: {input_length}; "
            f"output length: {generated_ids.shape[-1]}."
        )

    generated_ids_trimmed = generated_ids[:, input_length:]
    raw_ids = generated_ids_trimmed[0].tolist()

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    if debug:
        print("Output sequence length:", generated_ids.shape[-1])
        print("Generated token IDs:", raw_ids)

        print(
            "Generated tokens:",
            processor.tokenizer.convert_ids_to_tokens(raw_ids),
        )

        print("Decoded answer:", repr(answer))
        print("--- END GENERATION DIAGNOSTICS ---\n")

    if not answer:
        raise RuntimeError(
            "Gemma generated an empty answer after padding-token "
            "suppression. "
            f"Generated token IDs: {raw_ids}. "
            f"Input length: {input_length}. "
            f"Output length: {generated_ids.shape[-1]}. "
            "Inspect the generation diagnostics before changing "
            "the benchmark or scoring code."
        )

    return answer


# ============================================================
# Load processor and 4-bit model
# ============================================================

def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This 4-bit runner requires a CUDA GPU."
        )

    print("\nLoading Gemma processor...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    supports_bf16 = torch.cuda.is_bf16_supported()

    compute_dtype = (
        torch.bfloat16 if supports_bf16 else torch.float16
    )

    print(f"Model: {MODEL_NAME}")
    print("Quantization: 4-bit NF4")
    print(f"4-bit compute dtype: {compute_dtype}")
    print("Image preprocessing: Gemma processor defaults")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("\nLoading Gemma with 4-bit quantization...")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model.eval()

    # This is greedy generation, so sampling-only settings are unused.
    # Clear them to avoid misleading top_p/top_k warnings.
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    print("Model loaded successfully.")
    print("Padding token ID:", processor.tokenizer.pad_token_id)
    print("EOS token ID:", processor.tokenizer.eos_token_id)

    return processor, model, compute_dtype


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

    inputs = prepare_inputs(
        processor,
        model,
        image_records,
        prompt,
    )

    warmup_answer = generate_answer(
        processor,
        model,
        inputs,
        debug=True,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print("Warm-up answer:", warmup_answer)
    print(f"Warm-up latency: {elapsed:.3f}s")
    print(f"Warm-up peak VRAM: {get_peak_vram_mb()} MB")

    del inputs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Warm-up complete.")


# ============================================================
# Run one benchmark case
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
        debug=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = time.perf_counter() - start
    peak_vram = get_peak_vram_mb()

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
        "latency_seconds": round(latency, 6),
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
        category = result.get("category") or "Unknown"

        if result["correct"] is None:
            continue

        grouped.setdefault(category, []).append(
            float(result["correct"])
        )

    output = {}

    for category, scores in grouped.items():
        total = len(scores)
        correct = sum(scores)

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
        max(0, int(0.95 * len(values_sorted))),
    )

    return {
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": values_sorted[p95_index],
        "maximum_seconds": max(values),
    }


# ============================================================
# Save results without overwriting earlier runs
# ============================================================

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
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return output_file


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("Gemma 3 4B IT ZED Office Benchmark - 4-bit")
    print("=" * 60)

    print(f"Benchmark: {BENCHMARK_FILE}")
    print(f"Manifest: {MANIFEST_FILE}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Model: {MODEL_NAME}")
    print("Quantization: 4-bit NF4")

    if not BENCHMARK_FILE.is_file():
        raise FileNotFoundError(
            f"Benchmark not found: {BENCHMARK_FILE}"
        )

    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This 4-bit runner requires a CUDA GPU."
        )

    print("Device: CUDA")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    total_vram = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"GPU VRAM: {total_vram:.2f} GB")

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

    processor, model, compute_dtype = load_model()

    # Warm-up is excluded from benchmark results and summary metrics.
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
        "peak_gpu_vram_mb": (
            max(peak_values) if peak_values else None
        ),
    }

    output = {
        "model": MODEL_NAME,
        "model_variant": "gemma3-4b-zed-4bit-nf4",
        "quantized": True,
        "quantization": "4-bit NF4 with double quantization",
        "compute_dtype": str(compute_dtype),
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
    print(f"Cases: {summary['cases']}")
    print(f"Scored cases: {summary['scored_cases']}")
    print(f"Accuracy: {summary['accuracy']:.4f}")

    print(
        f"Mean latency: "
        f"{summary['latency']['mean_seconds']:.3f}s"
    )

    print(
        f"Median latency: "
        f"{summary['latency']['median_seconds']:.3f}s"
    )

    print(
        f"P95 latency: "
        f"{summary['latency']['p95_seconds']:.3f}s"
    )

    print(f"Peak GPU VRAM: {summary['peak_gpu_vram_mb']} MB")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

from metrics import (
    calculate_accuracy,
    calculate_capability_accuracy,
    calculate_latency_metrics,
)

from scoring import score_test


ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_FILE = (
    ROOT / "benchmark" / "office_v2_frozen.json"
)

RESULTS_DIR = ROOT / "results"

MODELS = {
    "500m": {
        "name": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "results_dir": "smolvlm-500m-instruct",
        "quantized": False,
    },
    "2b": {
        "name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "results_dir": "smolvlm2-2.2b-instruct",
        "quantized": True,
    },
}

MAX_NEW_TOKENS = 64

TEMPERATURE = 0.0

SYSTEM_PROMPT = """You are being evaluated on visual understanding.

Use only information supported by the provided image(s).

Answer briefly and directly.

Do not invent information that cannot be observed."""


def load_benchmark():
    with open(
        BENCHMARK_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return data.get("tests", [])


def get_image_path(frame_id):
    manifest_path = (
        ROOT
        / "benchmark"
        / "office_manifest.csv"
    )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as f:
        lines = f.read().splitlines()

    for line in lines[1:]:
        parts = line.split(",", 2)

        if (
            len(parts) >= 2
            and parts[0] == frame_id
        ):
            filename = (
                parts[1]
                .strip()
                .strip('"')
            )

            return (
                ROOT
                / "benchmark"
                / "assets"
                / "office"
                / filename
            )

    raise FileNotFoundError(
        f"Frame {frame_id} not found "
        "in office_manifest.csv"
    )


def load_images(case):
    images = []

    for frame_id in case["input_media"]:
        image_path = get_image_path(frame_id)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        images.append(
            Image.open(
                image_path
            ).convert("RGB")
        )

    return images


def build_prompt(case):
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n"
        f"{case['question']}\n\n"
        f"Answer:"
    )


def get_gpu_stats():
    if not torch.cuda.is_available():
        return {
            "gpu_vram_mb": None,
            "gpu_utilization_percent": None,
        }

    device = torch.cuda.current_device()

    allocated = (
        torch.cuda.memory_allocated(device)
    )

    reserved = (
        torch.cuda.memory_reserved(device)
    )

    return {
        "gpu_vram_mb": max(
            allocated,
            reserved,
        ) / (1024 * 1024),
        "gpu_utilization_percent": None,
    }


def load_model(model_config):
    model_name = model_config["name"]

    print("Loading model...")
    print(f"Model: {model_name}")

    processor = AutoProcessor.from_pretrained(
        model_name
    )

    if model_config["quantized"]:
        print("Quantization: 4-bit")

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        )

        model = (
            AutoModelForImageTextToText
            .from_pretrained(
                model_name,
                quantization_config=(
                    quantization_config
                ),
                device_map="auto",
            )
        )

    else:
        print("Quantization: None")

        model = (
            AutoModelForImageTextToText
            .from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
            )
        )

    model.eval()

    return processor, model


def prepare_inputs(
    processor,
    model,
    case,
):
    images = load_images(case)

    prompt = build_prompt(case)

    content = [
        {
            "type": "image",
        }
        for _ in images
    ]

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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    return inputs


def generate_answer(
    processor,
    model,
    inputs,
):
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_length = (
        inputs["input_ids"].shape[1]
    )

    generated_ids = (
        generated_ids[:, input_length:]
    )

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    return answer


def run_inference(
    processor,
    model,
    case,
):
    inputs = prepare_inputs(
        processor,
        model,
        case,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    answer = generate_answer(
        processor,
        model,
        inputs,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = (
        time.perf_counter() - start
    )

    return answer, latency


def warmup_model(
    processor,
    model,
    case,
):
    print("\nRunning GPU warm-up...")

    inputs = prepare_inputs(
        processor,
        model,
        case,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    generate_answer(
        processor,
        model,
        inputs,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    warmup_latency = (
        time.perf_counter() - start
    )

    print(
        f"Warm-up latency: "
        f"{warmup_latency:.3f}s"
    )

    print("Warm-up complete.")


def score_case(
    case,
    model_answer,
):
    scoring_case = {
        "expected_answer": case[
            "ground_truth"
        ],
        "accepted_answers": case.get(
            "accepted_answers",
            [],
        ),
        "partial_answers": case.get(
            "partial_answers",
            [],
        ),
        "scoring_mode": case.get(
            "scoring_mode",
            "exact_or_semantic",
        ),
        "answer_type": case[
            "answer_type"
        ],
    }

    correct = score_test(
        scoring_case,
        model_answer,
    )

    return correct

def evaluate_case(
    processor,
    model,
    case,
):
    gpu_before = get_gpu_stats()

    try:
        model_answer, latency = (
            run_inference(
                processor,
                model,
                case,
            )
        )

        correct = score_case(
            case,
            model_answer,
        )

        gpu_after = get_gpu_stats()

        return {
            "test_id": case[
                "test_id"
            ],
            "category": case[
                "category"
            ],
            "input_media": case[
                "input_media"
            ],
            "question": case[
                "question"
            ],
            "ground_truth": case[
                "ground_truth"
            ],
            "accepted_answers": case.get(
                "accepted_answers",
                [],
            ),
            "answer_type": case[
                "answer_type"
            ],
            "model_answer": model_answer,
            "correct": correct,
            "latency_seconds": latency,
            "gpu_vram_mb": (
                gpu_after.get(
                    "gpu_vram_mb"
                )
                or gpu_before.get(
                    "gpu_vram_mb"
                )
            ),
            "gpu_utilization_percent": None,
        }

    except Exception as e:
        return {
            "test_id": case[
                "test_id"
            ],
            "category": case[
                "category"
            ],
            "input_media": case[
                "input_media"
            ],
            "question": case[
                "question"
            ],
            "ground_truth": case[
                "ground_truth"
            ],
            "accepted_answers": case.get(
                "accepted_answers",
                [],
            ),
            "answer_type": case[
                "answer_type"
            ],
            "model_answer": None,
            "correct": None,
            "latency_seconds": None,
            "gpu_vram_mb": None,
            "gpu_utilization_percent": None,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=["500m", "2b"],
        default="500m",
        help="Model to evaluate",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of tests to run",
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Output JSON file",
    )

    parser.add_argument(
        "--test-id",
        type=str,
        nargs="+",
        default=None,
        help="Run one or more specific test IDs",
    )

    args = parser.parse_args()

    model_config = MODELS[args.model]

    model_name = model_config["name"]

    model_results_dir = (
        RESULTS_DIR
        / model_config["results_dir"]
    )

    tests = load_benchmark()

    if args.test_id is not None:
        requested_ids = set(
            args.test_id
        )

        all_test_ids = {
            test["test_id"]
            for test in tests
        }

        missing_ids = (
            requested_ids - all_test_ids
        )

        if missing_ids:
            raise ValueError(
                "Test ID(s) not found: "
                + ", ".join(
                    sorted(missing_ids)
                )
            )

        tests = [
            test
            for test in tests
            if test["test_id"]
            in requested_ids
        ]

    if args.limit is not None:
        tests = tests[:args.limit]

    print(
        f"Loaded {len(tests)} "
        "benchmark cases."
    )

    print(f"Model: {model_name}")

    if torch.cuda.is_available():
        print("Device: CUDA")

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        print("Device: CPU")

    processor, model = load_model(
        model_config
    )

    if tests:
        warmup_model(
            processor,
            model,
            tests[0],
        )

    results = []

    for index, case in enumerate(
        tests,
        start=1,
    ):
        print(
            f"\n[{index}/{len(tests)}] "
            f"{case['test_id']}"
        )

        result = evaluate_case(
            processor,
            model,
            case,
        )

        results.append(result)

        if result.get("error"):
            print(
                f"  ERROR: "
                f"{result['error']}"
            )

            continue

        print(
            f"  Answer: "
            f"{result['model_answer']}"
        )

        print(
            f"  Expected: "
            f"{result['ground_truth']}"
        )

        print(
            f"  Correct: "
            f"{result['correct']}"
        )

        print(
            f"  Latency: "
            f"{result['latency_seconds']:.3f}s"
        )

    scored_results = [
        r
        for r in results
        if r.get("correct") is not None
    ]

    accuracy = calculate_accuracy(
        scored_results
    )

    latency = calculate_latency_metrics(
        results
    )

    capability_results = []

    for result in scored_results:
        capability_results.append(
            {
                **result,
                "capability": result[
                    "category"
                ],
            }
        )

    capability_accuracy = (
        calculate_capability_accuracy(
            capability_results
        )
    )

    vram_values = [
        r["gpu_vram_mb"]
        for r in results
        if r.get("gpu_vram_mb")
        is not None
    ]

    summary = {
        "cases": len(results),
        "scored_cases": len(
            scored_results
        ),
        "accuracy": accuracy,
        "latency": latency,
        "capability_accuracy": (
            capability_accuracy
        ),
        "peak_gpu_vram_mb": (
            max(vram_values)
            if vram_values
            else None
        ),
    }

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    print(
        f"Cases: "
        f"{summary['cases']}"
    )

    print(
        f"Scored cases: "
        f"{summary['scored_cases']}"
    )

    print(
        f"Accuracy: "
        f"{summary['accuracy']}"
    )

    print(
        f"Mean latency: "
        f"{latency['mean_seconds']}"
    )

    print(
        f"Median latency: "
        f"{latency['median_seconds']}"
    )

    print(
        f"P95 latency: "
        f"{latency['p95_seconds']}"
    )

    print(
        f"Maximum latency: "
        f"{latency['maximum_seconds']}"
    )

    print(
        f"Peak GPU VRAM: "
        f"{summary['peak_gpu_vram_mb']} MB"
    )

    print("=" * 50)

    if args.results_file:
        results_file = args.results_file
    else:
        results_file = (
            model_results_dir
            / "results.json"
        )

    results_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "model": model_name,
        "model_variant": args.model,
        "quantized": model_config[
            "quantized"
        ],
        "quantization": (
            "4-bit"
            if model_config["quantized"]
            else None
        ),
        "benchmark_file": str(
            BENCHMARK_FILE
        ),
        "summary": summary,
        "results": results,
    }

    with open(
        results_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nResults saved to: "
        f"{results_file}"
    )


if __name__ == "__main__":
    main()
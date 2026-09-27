import argparse
import base64
import json
import re
import time
from pathlib import Path

import requests

from metrics import (
    calculate_accuracy,
    calculate_capability_accuracy,
    calculate_latency_metrics,
)
from scoring import score_test


ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_FILE = (
    ROOT / "benchmark" / "benchmark.json"
)

RESULTS_DIR = ROOT / "results"

MODEL_RESULTS_DIR = (
    RESULTS_DIR / "qwen2.5-vl-3b-instruct"
)

MODEL_NAME = (
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

API_BASE_URL = (
    "http://localhost:8000/v1"
)

MAX_TOKENS = 128
TEMPERATURE = 0.0


def file_to_data_url(path):
    path = Path(path)

    data = path.read_bytes()

    encoded = base64.b64encode(data).decode(
        "utf-8"
    )

    suffix = path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".mp4":
        mime = "video/mp4"
    elif suffix == ".mov":
        mime = "video/quicktime"
    else:
        mime = "application/octet-stream"

    return f"data:{mime};base64,{encoded}"


def build_question(case):
    question = case["question"]

    choices = case.get("choices")

    if choices:
        options = "\n".join(
            f"{i + 1}. {choice}"
            for i, choice in enumerate(choices)
        )

        question = (
            f"{question}\n\n"
            f"Choose one of the following options:\n"
            f"{options}\n\n"
            f"Answer with only the option text."
        )

    return question


def build_media_content(case):
    input_type = case.get("input_type")

    if input_type == "video":
        media_path = (
            ROOT / case["video"]
        )

        return {
            "type": "video_url",
            "video_url": {
                "url": file_to_data_url(
                    media_path
                )
            },
        }

    image_path = case.get(
        "image",
        case.get("image_path"),
    )

    if not image_path:
        raise ValueError(
            f"No image path found for "
            f"{case['test_id']}"
        )

    media_path = ROOT / image_path

    return {
        "type": "image_url",
        "image_url": {
            "url": file_to_data_url(
                media_path
            )
        },
    }


def query_model(case, model_name, api_url):
    question = build_question(case)

    media = build_media_content(case)

    content = [
        {
            "type": "text",
            "text": question,
        },
        media,
    ]

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    start = time.perf_counter()

    response = requests.post(
        f"{api_url}/chat/completions",
        json=payload,
        timeout=300,
    )

    latency = (
        time.perf_counter() - start
    )

    response.raise_for_status()

    data = response.json()

    answer = (
        data["choices"][0]
        ["message"]
        ["content"]
    )

    if isinstance(answer, list):
        answer = "".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )

    return str(answer).strip(), latency


def get_gpu_stats():
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu="
                "memory.used,"
                "utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        memory_values = []
        utilization_values = []

        for line in result.stdout.strip().splitlines():
            parts = [
                p.strip()
                for p in line.split(",")
            ]

            if len(parts) >= 2:
                memory_values.append(
                    float(parts[0])
                )

                utilization_values.append(
                    float(parts[1])
                )

        if not memory_values:
            return {
                "gpu_vram_mb": None,
                "gpu_utilization_percent": None,
            }

        return {
            "gpu_vram_mb": max(
                memory_values
            ),
            "gpu_utilization_percent": max(
                utilization_values
            ),
        }

    except Exception:
        return {
            "gpu_vram_mb": None,
            "gpu_utilization_percent": None,
        }


def score_case(case, model_answer):
    scored_answer = model_answer

    choices = case.get("choices")

    if choices:
        answer = model_answer.strip().lower()

        match = re.fullmatch(
            r"(?:option|choice)?\s*(\d+)",
            answer,
        )

        if match:
            index = (
                int(match.group(1)) - 1
            )

            if 0 <= index < len(choices):
                scored_answer = choices[
                    index
                ]

    correct = score_test(
        case,
        scored_answer,
    )

    return correct, scored_answer


def evaluate_case(case, model_name, api_url):
    gpu_before = get_gpu_stats()

    try:
        model_answer, latency = query_model(
            case,
            model_name,
            api_url,
        )

        correct, scored_answer = score_case(
            case,
            model_answer,
        )

        gpu_after = get_gpu_stats()

        gpu_vram = gpu_after.get(
            "gpu_vram_mb"
        )

        if gpu_vram is None:
            gpu_vram = gpu_before.get(
                "gpu_vram_mb"
            )

        gpu_utilization = (
            gpu_after.get(
                "gpu_utilization_percent"
            )
        )

        if gpu_utilization is None:
            gpu_utilization = (
                gpu_before.get(
                    "gpu_utilization_percent"
                )
            )

        result = {
            "test_id": case["test_id"],
            "dataset": case.get(
                "dataset"
            ),
            "capability": case.get(
                "capability"
            ),
            "input_type": case.get(
                "input_type"
            ),
            "question": case.get(
                "question"
            ),
            "expected_answer": case.get(
                "expected_answer"
            ),
            "model_answer": model_answer,
            "scored_answer": scored_answer,
            "correct": correct,
            "latency_seconds": latency,
            "gpu_vram_mb": gpu_vram,
            "gpu_utilization_percent": (
                gpu_utilization
            ),
        }

        return result

    except Exception as e:
        return {
            "test_id": case["test_id"],
            "dataset": case.get(
                "dataset"
            ),
            "capability": case.get(
                "capability"
            ),
            "input_type": case.get(
                "input_type"
            ),
            "question": case.get(
                "question"
            ),
            "expected_answer": case.get(
                "expected_answer"
            ),
            "model_answer": None,
            "scored_answer": None,
            "correct": None,
            "latency_seconds": None,
            "gpu_vram_mb": None,
            "gpu_utilization_percent": None,
            "error": str(e),
        }


def load_benchmark():
    with open(
        BENCHMARK_FILE,
        "r",
    ) as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("tests", [])

    return data


def check_server(model_name, api_url):
    print("Checking vLLM server...")

    response = requests.get(
        f"{api_url}/models",
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    models = [
        item["id"]
        for item in data.get("data", [])
    ]

    print(
        f"Available models: {models}"
    )

    if model_name not in models:
        raise RuntimeError(
            f"Model {model_name} "
            f"not found on server."
        )


def calculate_peak_gpu(results):
    values = [
        r["gpu_vram_mb"]
        for r in results
        if r.get("gpu_vram_mb")
        is not None
    ]

    if not values:
        return None

    return max(values)


def build_summary(results):
    accuracy = calculate_accuracy(
        results
    )

    latency = calculate_latency_metrics(
        results
    )

    capability_accuracy = (
        calculate_capability_accuracy(
            results
        )
    )

    peak_gpu = calculate_peak_gpu(
        results
    )

    gpu_utilization = [
        r["gpu_utilization_percent"]
        for r in results
        if r.get(
            "gpu_utilization_percent"
        ) is not None
    ]

    peak_gpu_utilization = (
        max(gpu_utilization)
        if gpu_utilization
        else None
    )

    return {
        "cases": len(results),
        "accuracy": accuracy,
        "latency": latency,
        "capability_accuracy": (
            capability_accuracy
        ),
        "peak_gpu_vram_mb": peak_gpu,
        "peak_gpu_utilization_percent": (
            peak_gpu_utilization
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Model ID exposed by the OpenAI-compatible server",
    )

    parser.add_argument(
        "--api-url",
        default=API_BASE_URL,
        help="Base URL for the server's OpenAI-compatible API",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for results (defaults to a model-specific folder)",
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Path for the JSON results file (overrides --results-dir)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of benchmark cases to run",
    )

    args = parser.parse_args()

    model_results_dir = args.results_dir
    if model_results_dir is None:
        if args.model == MODEL_NAME:
            model_results_dir = MODEL_RESULTS_DIR
        else:
            model_results_dir = (
                RESULTS_DIR / args.model.replace("/", "--")
            )

    results_file = args.results_file
    if results_file is None:
        results_file = model_results_dir / "results.json"

    check_server(args.model, args.api_url)

    tests = load_benchmark()

    if args.limit is not None:
        tests = tests[:args.limit]

    print(
        f"Loaded {len(tests)} benchmark cases."
    )

    print(
        f"Model: {args.model}"
    )

    results = []

    for index, case in enumerate(
        tests,
        start=1,
    ):
        print(
            f"\n[{index}/{len(tests)}] "
            f"Running {case['test_id']}..."
        )

        result = evaluate_case(
            case,
            args.model,
            args.api_url,
        )

        results.append(result)

        if result.get("error"):
            print(
                f"  ERROR: {result['error']}"
            )
            continue

        print(
            f"  Answer: "
            f"{result['model_answer']}"
        )

        if (
            result["scored_answer"]
            != result["model_answer"]
        ):
            print(
                f"  Scored as: "
                f"{result['scored_answer']}"
            )

        print(
            f"  Expected: "
            f"{result['expected_answer']}"
        )

        print(
            f"  Correct: "
            f"{result['correct']}"
        )

        print(
            f"  Latency: "
            f"{result['latency_seconds']:.3f}s"
        )

    summary = build_summary(
        results
    )

    print(
        "\n" + "=" * 50
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 50
    )

    print(
        f"Cases: "
        f"{summary['cases']}"
    )

    print(
        f"Accuracy: "
        f"{summary['accuracy']}"
    )

    latency = summary["latency"]

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

    print(
        f"Peak GPU utilization: "
        f"{summary['peak_gpu_utilization_percent']}%"
    )

    print(
        "=" * 50
    )

    results_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "model": args.model,
        "benchmark_file": str(
            BENCHMARK_FILE
        ),
        "summary": summary,
        "results": results,
    }

    with open(
        results_file,
        "w",
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
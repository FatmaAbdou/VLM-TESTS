import json
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / "benchmark/assets/office/Videos/recording_07/full.mp4"

OUTPUT_DIR = ROOT / "results/qwen2.5-vl-3b-video-test"
OUTPUT_FILE = OUTPUT_DIR / "recording_07_temporal_test_v5_5090.json"

FRAME_DIR = OUTPUT_DIR / "sampled_frames_v5"

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

MIN_PIXELS = 3136
MAX_PIXELS = 28 * 28 * 768
MAX_NEW_TOKENS = 80

# Focus on the approach to the doorway and the final frame.
TIMESTAMPS = [40.0, 44.0, 48.0, 51.0, 54.0, 56.0]


QUESTIONS = {
    "temporal_visibility": (
        "For EACH timestamp, answer exactly one line using this format:\n"
        "40s: Staircase=<clearly visible|partly visible/cropped|not visible>; "
        "EXIT=<clearly visible|partly visible/cropped|not visible>\n"
        "44s: Staircase=<clearly visible|partly visible/cropped|not visible>; "
        "EXIT=<clearly visible|partly visible/cropped|not visible>\n"
        "48s: Staircase=<clearly visible|partly visible/cropped|not visible>; "
        "EXIT=<clearly visible|partly visible/cropped|not visible>\n"
        "51s: Staircase=<clearly visible|partly visible/cropped|not visible>; "
        "EXIT=<clearly visible|partly visible/cropped|not visible>\n"
        "54s: Staircase=<clearly visible|partly visible/cropped|not visible>; "
        "EXIT=<clearly visible|partly visible/cropped|not visible>\n"
        "56s: Staircase=<clearly visible|partly visible/cropped|not visible>; "
        "EXIT=<clearly visible|partly visible/cropped|not visible>\n\n"
        "Use only what is actually visible in each image. "
        "Do not infer what happens between timestamps."
    ),
    "final_frame": (
        "Look ONLY at the 56.0-second frame. "
        "Answer exactly:\n"
        "Staircase=<Yes|No>\n"
        "EXIT sign=<Yes|No>\n"
        "Then give one short sentence describing what is visible around the doorway."
    ),
}


SYSTEM_PROMPT = (
    "You are being evaluated on visual understanding. "
    "Use only evidence visible in the supplied image frames. "
    "Do not invent details or assume an object is visible when it "
    "is outside the image. Follow the requested answer format exactly."
)


def extract_frames():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    records = []

    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for seconds in TIMESTAMPS:
            if seconds < 0 or seconds >= duration:
                raise ValueError(
                    f"Timestamp {seconds}s outside video duration "
                    f"({duration:.2f}s)"
                )

            cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)

            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Could not extract frame at {seconds}s"
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            frame_path = FRAME_DIR / f"frame_{seconds:.1f}s.jpg"
            image.save(frame_path, quality=95)

            records.append({
                "timestamp_seconds": seconds,
                "path": frame_path,
            })

    finally:
        cap.release()

    print(f"Video duration: {duration:.2f}s")
    print(f"Extracted {len(records)} frames:")

    for record in records:
        print(
            f"  {record['timestamp_seconds']}s: "
            f"{record['path']}"
        )

    return records


def load_model():
    print("\nLoading processor...")

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

    return processor, model


def build_messages(frames, question):
    content = []

    for record in frames:
        content.append({
            "type": "text",
            "text": (
                f"Frame at "
                f"{record['timestamp_seconds']:.1f} seconds:"
            ),
        })

        content.append({
            "type": "image",
            "image": str(record["path"]),
        })

    content.append({
        "type": "text",
        "text": f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}",
    })

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def ask_question(processor, model, frames, question):
    messages = build_messages(frames, question)

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
        key: value.to(model.device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    torch.cuda.synchronize()

    latency = time.perf_counter() - start

    peak_vram_mb = round(
        torch.cuda.max_memory_allocated() / 1024**2,
        1,
    )

    trimmed_ids = [
        output[len(input_ids):]
        for input_ids, output in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]

    answer = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return {
        "question": question,
        "model_answer": answer,
        "latency_seconds": round(latency, 3),
        "peak_gpu_memory_allocated_mb": peak_vram_mb,
    }


def main():
    # Refuse to overwrite an existing result.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite it:\n"
            f"{OUTPUT_FILE}\n"
            "Choose a new OUTPUT_FILE name before running again."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Activate the expected GPU environment."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Qwen2.5-VL Video Temporal Test — V5")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Timestamps: {TIMESTAMPS}")
    print(f"Output: {OUTPUT_FILE}")

    frames = extract_frames()
    processor, model = load_model()

    results = {}

    # ---------------------------------------------------------
    # QUESTION 1: Temporal visibility
    # Uses all six frames.
    # ---------------------------------------------------------
    name = "temporal_visibility"
    question = QUESTIONS[name]

    print(f"\n{'=' * 60}")
    print(f"QUESTION: {name}")
    print(question)

    result = ask_question(
        processor,
        model,
        frames,
        question,
    )

    results[name] = result

    print("\nMODEL ANSWER")
    print(result["model_answer"])

    print(f"Latency: {result['latency_seconds']}s")

    print(
        "Peak GPU memory allocated: "
        f"{result['peak_gpu_memory_allocated_mb']} MB"
    )

    # ---------------------------------------------------------
    # QUESTION 2: Final frame
    # IMPORTANT:
    # Give the model ONLY the 56-second frame.
    # ---------------------------------------------------------
    name = "final_frame"
    question = QUESTIONS[name]

    final_frame = frames[-1:]

    print(f"\n{'=' * 60}")
    print(f"QUESTION: {name}")
    print(question)

    result = ask_question(
        processor,
        model,
        final_frame,
        question,
    )

    results[name] = result

    print("\nMODEL ANSWER")
    print(result["model_answer"])

    print(f"Latency: {result['latency_seconds']}s")

    print(
        "Peak GPU memory allocated: "
        f"{result['peak_gpu_memory_allocated_mb']} MB"
    )

    # ---------------------------------------------------------
    # Save final result.
    # ---------------------------------------------------------
    output = {
        "model": MODEL_NAME,
        "experiment": "sampled_video_temporal_reasoning_v5",
        "video": str(VIDEO_PATH),
        "timestamps_seconds": TIMESTAMPS,
        "frame_paths": [
            str(record["path"])
            for record in frames
        ],
        "results": results,
        "note": (
            "Six sampled images focus on the approach to the "
            "doorway. The temporal visibility question uses all "
            "six sampled frames. The final-frame question uses "
            "only the 56.0-second frame. This is image-sequence "
            "reasoning, not native video input. The exact time "
            "an object enters or leaves the frame cannot be "
            "determined from sparse samples alone."
        ),
    }

    # Exclusive creation prevents overwriting an existing file.
    with OUTPUT_FILE.open("x", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved result: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
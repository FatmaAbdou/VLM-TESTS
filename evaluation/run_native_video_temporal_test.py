import json
import time
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

VIDEO_PATH = Path(
    "benchmark/assets/office/Videos/recording_07/full.mp4"
)

OUTPUT_PATH = Path(
    "results/qwen2.5-vl-3b-video-test/"
    "recording_07_native_video_temporal_test.json"
)

MIN_PIXELS = 3136
MAX_PIXELS = 28 * 28 * 768

MAX_NEW_TOKENS = 220


# ============================================================
# CHECKS
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT_PATH}\n"
        "This test will not overwrite an existing result."
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GPU
# ============================================================

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("NATIVE VIDEO TEMPORAL TEST")
print("=" * 60)

print(f"Model: {MODEL_NAME}")
print(f"Video: {VIDEO_PATH}")
print()

print("Loading model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)

model.eval()

print("Model loaded.")
print()


# ============================================================
# QUESTION
# ============================================================

question = """
Watch the entire video and answer the following about the
staircase and EXIT sign near the end.

Use only information visible in the video.

Answer in exactly this structure:

1. APPROACH:
Describe what happens to the staircase and EXIT sign as
the camera approaches the doorway near the end.

2. FINAL STATE:
At approximately 56 seconds, is the staircase visible?
Is the EXIT sign visible?

3. CHANGE:
State whether each object goes from visible to not visible,
remains visible, or is partly visible/cropped during the
approach.

4. EVIDENCE:
Briefly describe what is actually visible around the doorway
in the final part of the video.

Do not assume an object is visible just because it was visible
earlier in the video. Do not invent events between observations.
"""


# ============================================================
# NATIVE VIDEO INPUT
# ============================================================

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": str(VIDEO_PATH),
            },
            {
                "type": "text",
                "text": question,
            },
        ],
    }
]


# ============================================================
# PROCESS VIDEO
# ============================================================

print("Processing native video input...")
print("The MP4 itself is being passed to Qwen.")
print()

start_time = time.perf_counter()

image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages,
    return_video_kwargs=True,
    image_patch_size=processor.image_processor.patch_size,
)

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)

inputs = inputs.to(model.device)


# ============================================================
# GENERATE
# ============================================================

print("Generating answer...")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

elapsed = time.perf_counter() - start_time


# ============================================================
# DECODE
# ============================================================

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(
        inputs.input_ids,
        generated_ids,
    )
]

answer = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():
    peak_memory_mb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )
else:
    peak_memory_mb = None


# ============================================================
# RESULT
# ============================================================

result = {
    "model": MODEL_NAME,
    "experiment": "native_video_temporal_reasoning",
    "video": str(VIDEO_PATH.resolve()),
    "input_type": "native_video",
    "video_extraction_for_model": False,
    "question": question.strip(),
    "model_answer": answer.strip(),
    "latency_seconds": round(elapsed, 3),
    "peak_gpu_memory_allocated_mb": (
        round(peak_memory_mb, 1)
        if peak_memory_mb is not None
        else None
    ),
    "note": (
        "The MP4 was passed directly to Qwen2.5-VL through "
        "the native video input pathway. No OpenCV-extracted "
        "JPEG frames were supplied to the model. This test "
        "focuses on temporal reasoning about the staircase "
        "and EXIT sign near the end of the video."
    ),
}


# ============================================================
# SAVE
# ============================================================

with OUTPUT_PATH.open(
    "x",
    encoding="utf-8",
) as f:
    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print(answer)

print()
print(f"Latency: {elapsed:.3f} seconds")

if peak_memory_mb is not None:
    print(
        f"Peak GPU memory: "
        f"{peak_memory_mb:.1f} MB"
    )

print()
print(f"Saved to: {OUTPUT_PATH}")
print("=" * 60)
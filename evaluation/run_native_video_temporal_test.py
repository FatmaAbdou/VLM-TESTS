
import json
import time
from pathlib import Path

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
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
    "recording_07_native_video_temporal_test3.json"
)

MIN_PIXELS = 3136
MAX_PIXELS = 28 * 28 * 768

MAX_NEW_TOKENS = 500


# ============================================================
# CHECKS
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH.resolve()}"
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
print("NATIVE VIDEO TEMPORAL AND DOORWAY TRAVERSAL TEST")
print("=" * 60)

print(f"Model: {MODEL_NAME}")
print(f"Video: {VIDEO_PATH}")
print(f"Output: {OUTPUT_PATH}")
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
# QUESTIONS
# ============================================================

question = """
Watch the entire video and answer both questions below.

Use only information visible in the video. Pay particular
attention to the final approximately 10 seconds.

Do not assume that the camera crossed a doorway merely
because it approached the doorway or because objects
disappeared from view.

============================================================
QUESTION 1 — STAIRCASE AND EXIT SIGN
============================================================

Focus on the staircase and EXIT sign near the end of the video.

1. APPROACH:
Describe what happens to the staircase and EXIT sign as
the camera approaches the doorway near the end.

2. FINAL STATE:
At approximately 56 seconds, is the staircase visible?
Is the EXIT sign visible?
If the video ends before that timestamp, describe the
actual final view.

3. CHANGE:
For EACH object separately, state whether it:
- Goes from visible to not visible.
- Remains visible.
- Becomes partly visible or cropped.
- Cannot be determined from the available observations.

4. EVIDENCE:
Describe what is actually visible around the doorway
in the final part of the video.

============================================================
QUESTION 2 — WALKING THROUGH THE DOORWAY
============================================================

Focus on the final approximately 10 seconds.

1. ACTION:
Describe the camera's observed movement near the doorway.
Does it appear to cross the threshold, approach without
crossing, stop, move away, or remain uncertain?

2. BEFORE CROSSING:
Describe the visible scene immediately before the possible
crossing. Identify the doorway and surrounding features
that are actually visible.

3. DURING CROSSING:
Describe the observed visual changes that indicate whether
the camera passes through the doorway. Consider the position
of the doorframe, movement relative to the threshold, and
changes in the visible scene.

4. AFTER CROSSING:
Describe the scene visible after the possible crossing.
Does the viewpoint appear to be on the other side of the
doorway? Explain what supports this interpretation, or why
the evidence is insufficient.

5. CONCLUSION:
Choose exactly ONE label:

CROSSED
The video provides visual evidence that the camera passed
through the doorway.

APPROACHED_ONLY
The camera approaches the doorway, but crossing is not
visibly established.

UNCERTAIN
The available visual evidence is insufficient to determine
whether the camera crossed the doorway.

6. EVIDENCE:
Give specific visual evidence supporting the conclusion.
Do not use the disappearance of the staircase or EXIT sign
as proof of crossing unless other visual changes support it.

============================================================
REQUIRED OUTPUT FORMAT
============================================================

You MUST answer every section below in the exact order shown.
Do not omit, merge, or replace any section with a conclusion.

QUESTION 1 — STAIRCASE AND EXIT SIGN

1. APPROACH:
[Your answer]

2. FINAL STATE:
[Your answer]

3. CHANGE:
- Staircase: [Your answer]
- EXIT sign: [Your answer]

4. EVIDENCE:
[Your answer]

QUESTION 2 — WALKING THROUGH THE DOORWAY

1. ACTION:
[Your answer]

2. BEFORE CROSSING:
[Your answer]

3. DURING CROSSING:
[Your answer]

4. AFTER CROSSING:
[Your answer]

5. CONCLUSION:
[Exactly one label]

6. EVIDENCE:
[Your answer]

If a section cannot be answered from the video, write
"INSUFFICIENT VISUAL EVIDENCE" and briefly explain why.

Do not omit a section even when the answer is uncertain.
Do not invent intermediate events or claim to see details
that are not supported by the video.
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
print("No OpenCV-extracted JPEG frames are supplied.")
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

preprocessing_elapsed = time.perf_counter() - start_time

print(
    f"Video preprocessing completed in "
    f"{preprocessing_elapsed:.3f} seconds."
)
print()


# ============================================================
# GENERATE
# ============================================================

print("Generating answer...")

generation_start = time.perf_counter()

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

generation_elapsed = time.perf_counter() - generation_start

total_elapsed = time.perf_counter() - start_time


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

decode_start = time.perf_counter()

answer = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

decode_elapsed = time.perf_counter() - decode_start


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
    "experiment": (
        "native_video_temporal_reasoning_and_doorway_traversal"
    ),
    "video": str(VIDEO_PATH.resolve()),
    "input_type": "native_video",
    "video_extraction_for_model": False,
    "quantization": "4-bit",
    "max_new_tokens": MAX_NEW_TOKENS,
    "questions": {
        "question_1": (
            "Staircase and EXIT sign temporal visibility"
        ),
        "question_2": (
            "Doorway traversal recognition"
        ),
    },
    "question": question.strip(),
    "model_answer": answer.strip(),
    "timing_seconds": {
        "preprocessing": round(preprocessing_elapsed, 3),
        "generation": round(generation_elapsed, 3),
        "decoding": round(decode_elapsed, 3),
        "total": round(total_elapsed, 3),
    },
    "latency_seconds": round(total_elapsed, 3),
    "peak_gpu_memory_allocated_mb": (
        round(peak_memory_mb, 1)
        if peak_memory_mb is not None
        else None
    ),
    "note": (
        "The MP4 was passed directly to Qwen2.5-VL through "
        "the native video input pathway. No OpenCV-extracted "
        "JPEG frames were supplied to the model. The test "
        "evaluates temporal visibility of the staircase and "
        "EXIT sign, and recognition of camera traversal through "
        "a doorway during the final approximately 10 seconds. "
        "Every requested answer section is explicitly specified "
        "in the prompt. Prompt formatting does not guarantee "
        "that the model will comply or that its answer is correct."
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
print("-" * 60)
print("PERFORMANCE")
print("-" * 60)

print(
    f"Preprocessing: {preprocessing_elapsed:.3f} seconds"
)
print(
    f"Generation:    {generation_elapsed:.3f} seconds"
)
print(
    f"Decoding:      {decode_elapsed:.3f} seconds"
)
print(
    f"Total latency: {total_elapsed:.3f} seconds"
)

if peak_memory_mb is not None:
    print(
        f"Peak GPU memory: {peak_memory_mb:.1f} MB"
    )

print()
print(f"Saved to: {OUTPUT_PATH}")
print("=" * 60)

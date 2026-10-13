
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
    "recording_07_native_video_temporal_test2.json"
)

MIN_PIXELS = 3136
MAX_PIXELS = 28 * 28 * 768

MAX_NEW_TOKENS = 350


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
actual final frame instead.

3. CHANGE:
For each object, state whether it:
- Goes from visible to not visible.
- Remains visible.
- Becomes partly visible or cropped.
- Cannot be determined from the available observations.

4. EVIDENCE:
Briefly describe what is actually visible around the doorway
in the final part of the video.

============================================================
QUESTION 2 — WALKING THROUGH THE DOORWAY
============================================================

Focus on the final approximately 10 seconds of the video.

1. ACTION:
Does the camera appear to:
- Pass through the doorway.
- Approach the doorway without crossing it.
- Stop before the doorway.
- Move away from the doorway.
- Remain uncertain because the evidence is insufficient?

2. BEFORE CROSSING:
Describe the visible scene immediately before the possible
doorway crossing.

3. DURING CROSSING:
Describe the visual changes, if any, that indicate the
camera is moving through the doorway. Consider:
- The position and apparent size of the doorframe.
- Whether the doorframe moves past the camera's viewpoint.
- Whether the visible scene changes as the camera advances.
- Whether the camera appears to cross the doorway threshold.

4. AFTER CROSSING:
Describe the scene visible after the possible crossing.
Does the viewpoint appear to have moved to the other side
of the doorway, or is that not established by the footage?

5. CONCLUSION:
Choose exactly ONE of the following labels:

CROSSED:
The video provides visual evidence that the camera passed
through the doorway.

APPROACHED_ONLY:
The camera approaches the doorway, but crossing is not
visibly established.

UNCERTAIN:
The available visual evidence is insufficient to determine
whether the camera crossed the doorway.

6. EVIDENCE:
Give the specific visual evidence supporting your conclusion.
Distinguish observed motion from assumptions about what
happened.

============================================================
IMPORTANT RULES
============================================================

- Approaching a doorway does not necessarily mean crossing it.
- Seeing the space beyond a doorway does not prove that
  the camera entered that space.
- Do not claim that crossing occurred unless the visual
  evidence supports that conclusion.
- Distinguish actual viewpoint movement through the doorway
  from a stationary view or a change in camera direction.
- Do not assume an object remains visible just because it
  appeared earlier in the video.
- Do not invent events between observations.
- If the footage is ambiguous, explicitly acknowledge this.

Answer both questions using the numbered structure above.
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
    "questions": {
        "question_1": "Staircase and EXIT sign temporal visibility",
        "question_2": "Doorway traversal recognition",
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
        "JPEG frames were supplied to the model. The experiment "
        "evaluates temporal visibility of the staircase and EXIT "
        "sign, and whether the model can recognize and support "
        "a claim that the camera passes through a doorway during "
        "the final approximately 10 seconds. The model's conclusion "
        "must be evaluated against the actual video."
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

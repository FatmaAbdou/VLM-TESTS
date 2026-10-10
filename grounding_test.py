import json
import re
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from huggingface_hub import snapshot_download
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

IMAGE_DIR = Path("benchmark/assets/office/images")

MIN_PIXELS = 3136
MAX_PIXELS = 28 * 28 * 768

MAX_NEW_TOKENS = 64

IMAGE_NAME = (
    "ZED-X_SN45626933_٢٠٢٦-٠٩-٢١_١٥-١٠-٣٥_"
    "٢٠٢٦-٠٩-٢١_١٥-٢١-١٩.png"
)

OUTPUT_IMAGE = Path("stanchion_bbox_result.png")


# ============================================================
# FIND LOCAL CACHED MODEL
# ============================================================

print("=" * 60)
print("LOADING LOCAL MODEL CACHE")
print("=" * 60)

try:
    MODEL_PATH = snapshot_download(
        repo_id=MODEL_NAME,
        local_files_only=True,
    )

    print(f"Local model path: {MODEL_PATH}")

except Exception as exc:
    raise RuntimeError(
        "\nCould not find the Qwen model in the local "
        "Hugging Face cache.\n\n"
        "The model was previously loaded successfully, "
        "so the model should already exist locally.\n\n"
        f"Original error: {exc}"
    ) from exc


# ============================================================
# LOAD IMAGE
# ============================================================

image_path = IMAGE_DIR / IMAGE_NAME

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found:\n{image_path}"
    )

image = Image.open(image_path).convert("RGB")

print("\n" + "=" * 60)
print("IMAGE")
print("=" * 60)

print(f"Image: {image_path}")

print(
    f"Original image size: "
    f"{image.size[0]} x {image.size[1]}"
)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\nLoading processor from local cache...")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
    local_files_only=True,
)

print("Image preprocessing: capped")
print(f"Min pixels: {MIN_PIXELS}")
print(f"Max pixels: {MAX_PIXELS}")

print(
    f"Processor max_pixels: "
    f"{processor.image_processor.max_pixels}"
)

print(
    f"Processor min_pixels: "
    f"{processor.image_processor.min_pixels}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model with 4-bit quantization...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    local_files_only=True,
)

model.eval()

print("Model loaded.")


# ============================================================
# GROUNDING PROMPT
# ============================================================

prompt = """
Find the standalone stanchion nearest to the camera.

OUTPUT RULES:
1. Output ONLY valid JSON.
2. Do NOT use Markdown.
3. Do NOT use ``` or code fences.
4. Do NOT write any explanation.
5. Do NOT use '=' signs.
6. The four bbox values must be numbers.
7. x1 must be smaller than x2.
8. y1 must be smaller than y2.

Return exactly this JSON structure:

[
  {
    "bbox_2d": [x1, y1, x2, y2],
    "label": "standalone stanchion nearest to the camera"
  }
]

The coordinates must be normalized from 0 to 1000.
""".strip()


# ============================================================
# PREPARE VISION INPUT
# ============================================================

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(
    messages
)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)


# ============================================================
# INPUT DIAGNOSTICS
# ============================================================

print("\n" + "=" * 60)
print("INPUT DIAGNOSTICS")
print("=" * 60)

if "input_ids" in inputs:
    print(
        f"Input tokens: "
        f"{inputs['input_ids'].shape[-1]}"
    )

if "pixel_values" in inputs:
    print(
        f"Pixel tensor shape: "
        f"{tuple(inputs['pixel_values'].shape)}"
    )

if "image_grid_thw" in inputs:
    print(
        f"Image grid: "
        f"{inputs['image_grid_thw'].tolist()}"
    )

print("=" * 60)


# ============================================================
# GENERATE
# ============================================================

print("Generating...")

if torch.cuda.is_available():
    torch.cuda.synchronize()

generation_start = time.perf_counter()

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

generation_end = time.perf_counter()

generation_latency = (
    generation_end - generation_start
)


# ============================================================
# DECODE
# ============================================================

decode_start = time.perf_counter()

input_token_count = inputs["input_ids"].shape[-1]

generated_only = generated_ids[:, input_token_count:]

raw_output = processor.batch_decode(
    generated_only,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

decode_end = time.perf_counter()

decode_latency = (
    decode_end - decode_start
)


# ============================================================
# TOKEN DIAGNOSTICS
# ============================================================

print("\n" + "=" * 60)
print("TOKEN DIAGNOSTICS")
print("=" * 60)

token_ids = generated_only[0].tolist()

print(
    f"Generated token IDs: "
    f"{token_ids}"
)

try:
    token_strings = (
        processor.tokenizer.convert_ids_to_tokens(
            token_ids
        )
    )

    print(
        f"Generated tokens: "
        f"{token_strings}"
    )

except Exception as exc:
    print(
        f"Could not decode individual tokens: "
        f"{exc}"
    )

print("=" * 60)


# ============================================================
# RAW MODEL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("MODEL OUTPUT")
print("=" * 60)

print(repr(raw_output))

print("=" * 60)


# ============================================================
# STRICT JSON VALIDATION
# ============================================================

parse_start = time.perf_counter()

strict_json_valid = False
strict_bbox = None
json_error = None

try:

    parsed = json.loads(raw_output)

    strict_json_valid = True

    if (
        not isinstance(parsed, list)
        or len(parsed) == 0
        or not isinstance(parsed[0], dict)
        or "bbox_2d" not in parsed[0]
    ):
        raise ValueError(
            "JSON structure does not contain "
            "the expected bbox_2d field."
        )

    strict_bbox = parsed[0]["bbox_2d"]

except Exception as exc:

    json_error = str(exc)


# ============================================================
# EXTRACT BBOX FROM RAW OUTPUT
# ============================================================
#
# This does NOT repair the model output.
#
# It only lets us separately evaluate the visual grounding
# when the model gives a correct bbox but malformed JSON.
# ============================================================

bbox = None
bbox_extraction_error = None

if strict_bbox is not None:

    bbox = strict_bbox

else:

    # Look specifically for:
    #
    # "bbox_2d": [729, 173, 867, 486]
    #
    # even if the surrounding JSON is malformed.

    number_pattern = r"-?\d+(?:\.\d+)?"

    bbox_match = re.search(
        rf'"bbox_2d"\s*:\s*'
        rf'\[\s*'
        rf'({number_pattern})\s*,\s*'
        rf'({number_pattern})\s*,\s*'
        rf'({number_pattern})\s*,\s*'
        rf'({number_pattern})'
        rf'\s*\]',
        raw_output,
    )

    if bbox_match:

        bbox = [
            float(bbox_match.group(1)),
            float(bbox_match.group(2)),
            float(bbox_match.group(3)),
            float(bbox_match.group(4)),
        ]

    else:

        bbox_extraction_error = (
            "Could not extract bbox_2d from raw output."
        )


parse_end = time.perf_counter()

parse_latency = (
    parse_end - parse_start
)


# ============================================================
# VALIDATE BBOX
# ============================================================

valid_bbox = False
validation_error = None

if bbox is None:

    validation_error = (
        bbox_extraction_error
        or "No bounding box found."
    )

else:

    try:

        if len(bbox) != 4:
            raise ValueError(
                f"Expected 4 coordinates, "
                f"got {len(bbox)}"
            )

        x1, y1, x2, y2 = [
            float(value)
            for value in bbox
        ]

        # ----------------------------------------------------
        # Range validation
        # ----------------------------------------------------

        if not (0 <= x1 <= 1000):
            raise ValueError(
                f"x1 out of range: {x1}"
            )

        if not (0 <= y1 <= 1000):
            raise ValueError(
                f"y1 out of range: {y1}"
            )

        if not (0 <= x2 <= 1000):
            raise ValueError(
                f"x2 out of range: {x2}"
            )

        if not (0 <= y2 <= 1000):
            raise ValueError(
                f"y2 out of range: {y2}"
            )

        # ----------------------------------------------------
        # Coordinate ordering
        # ----------------------------------------------------

        if x1 >= x2:
            raise ValueError(
                f"x1 must be smaller than x2: "
                f"{x1} >= {x2}"
            )

        if y1 >= y2:
            raise ValueError(
                f"y1 must be smaller than y2: "
                f"{y1} >= {y2}"
            )

        valid_bbox = True

    except Exception as exc:

        validation_error = str(exc)


# ============================================================
# DRAW BOUNDING BOX
# ============================================================

drawing_start = time.perf_counter()

if valid_bbox:

    x1, y1, x2, y2 = [
        float(value)
        for value in bbox
    ]

    image_width, image_height = image.size

    pixel_x1 = int(
        x1 / 1000 * image_width
    )

    pixel_y1 = int(
        y1 / 1000 * image_height
    )

    pixel_x2 = int(
        x2 / 1000 * image_width
    )

    pixel_y2 = int(
        y2 / 1000 * image_height
    )

    annotated = image.copy()

    draw = ImageDraw.Draw(annotated)

    draw.rectangle(
        [
            pixel_x1,
            pixel_y1,
            pixel_x2,
            pixel_y2,
        ],
        outline="red",
        width=6,
    )

    draw.text(
        (
            pixel_x1,
            max(0, pixel_y1 - 25),
        ),
        "Qwen detection",
        fill="red",
    )

    annotated.save(
        OUTPUT_IMAGE
    )

drawing_end = time.perf_counter()

drawing_latency = (
    drawing_end - drawing_start
)


# ============================================================
# LATENCY
# ============================================================

total_model_latency = (
    generation_latency
    + decode_latency
)

total_end_to_end = (
    generation_latency
    + decode_latency
    + parse_latency
    + drawing_latency
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("GROUNDING RESULT")
print("=" * 60)

if strict_json_valid:

    print(
        "Strict JSON:                  PASS"
    )

else:

    print(
        "Strict JSON:                  FAIL"
    )

    print(
        f"JSON error:                  "
        f"{json_error}"
    )


if bbox is not None:

    print(
        f"Extracted bbox:              "
        f"{bbox}"
    )


if valid_bbox:

    print(
        "Bounding box validation:     PASS"
    )

    print(
        f"Annotated image:             "
        f"{OUTPUT_IMAGE}"
    )

else:

    print(
        "Bounding box validation:     FAIL"
    )

    print(
        f"Reason:                      "
        f"{validation_error}"
    )


# ------------------------------------------------------------
# Important interpretation
# ------------------------------------------------------------

if valid_bbox and not strict_json_valid:

    print()
    print(
        "INTERPRETATION:"
    )

    print(
        "The model produced a valid bounding box,"
    )

    print(
        "but the surrounding JSON was malformed."
    )

    print(
        "Grounding and output-format compliance"
    )

    print(
        "are therefore reported separately."
    )

print("=" * 60)


# ============================================================
# LATENCY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LATENCY")
print("=" * 60)

print(
    f"Detection / generation latency: "
    f"{generation_latency:.4f} seconds"
)

print(
    f"Output decoding latency:        "
    f"{decode_latency:.4f} seconds"
)

print(
    f"BBox parsing latency:           "
    f"{parse_latency:.4f} seconds"
)

print(
    f"Drawing latency:                "
    f"{drawing_latency:.4f} seconds"
)

print(
    f"Total model latency:            "
    f"{total_model_latency:.4f} seconds"
)

print(
    f"Total end-to-end latency:       "
    f"{total_end_to_end:.4f} seconds"
)

print("=" * 60)
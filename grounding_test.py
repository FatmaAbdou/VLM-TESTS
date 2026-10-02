import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

MANIFEST = Path("benchmark/office_manifest.csv")
IMAGE_DIR = Path("benchmark/assets/office")


# ============================================================
# Find frame_035
# ============================================================

with open(MANIFEST, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

row = next(
    r for r in rows
    if r["frame_id"] == "frame_035"
)

image_path = IMAGE_DIR / row["filename"]

image = Image.open(image_path).convert("RGB")

width, height = image.size

print("Image:", image_path)
print("Image size:", width, "x", height)


# ============================================================
# Load model
# ============================================================

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("Loading model...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Model loaded.")


# ============================================================
# Prompt
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
               "text": (
    "Locate the standalone stanchion nearest to the camera.\n\n"
    "The target is the single freestanding stanchion post, "
    "not the rope and not any other stanchion.\n"
    "The bounding box must cover the ENTIRE visible stanchion post, "
    "including its top, full vertical pole, and circular base at the bottom.\n"
    "Do not crop the top or bottom of the stanchion.\n"
    "Do not include the floor area beyond its base.\n"
    "Do not include the rope unless it is unavoidable because it overlaps "
    "the stanchion.\n\n"
    "Return ONLY this JSON format:\n"
    "[{\"bbox_2d\":[x1,y1,x2,y2],"
    "\"label\":\"standalone stanchion nearest to the camera\"}]\n\n"
    "Use pixel coordinates based on the original image.\n"
    f"The image dimensions are {width} x {height} pixels.\n"
    "x1 is left, y1 is top, x2 is right, y2 is bottom."
),
            },
        ],
    }
]


# ============================================================
# Prepare input
# ============================================================

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)


# ============================================================
# Generate
# ============================================================

print("Generating...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )


# Only decode newly generated tokens
generated_ids = outputs[:, inputs.input_ids.shape[1]:]

answer = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0].strip()


print()
print("=" * 60)
print("MODEL OUTPUT:")
print(answer)
print("=" * 60)


# ============================================================
# Parse bounding box
# ============================================================

try:
    # Remove Markdown code fences
    cleaned = (
        answer
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    data = json.loads(cleaned)

    # The model returns a list containing an object
    if isinstance(data, list):
        if not data:
            raise ValueError("Empty bounding-box list")
        data = data[0]

    # Extract bbox_2d
    if "bbox_2d" in data:
        bbox = data["bbox_2d"]
    elif "bbox" in data:
        bbox = data["bbox"]
    else:
        raise ValueError("No bbox_2d or bbox field found")

    if len(bbox) != 4:
        raise ValueError("Bounding box must contain 4 values")

    x1, y1, x2, y2 = [float(v) for v in bbox]

    print("Model bbox (0-1000):", bbox)

    # Validate normalized coordinates
    if not all(0 <= v <= 1000 for v in bbox):
        raise ValueError("Coordinates outside 0-1000")

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid or reversed bounding box")

    # Convert normalized coordinates to pixels
    px1 = int(x1 / 1000 * width)
    py1 = int(y1 / 1000 * height)
    px2 = int(x2 / 1000 * width)
    py2 = int(y2 / 1000 * height)

    print("Pixel bbox:", [px1, py1, px2, py2])

    # Draw the box
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    draw.rectangle(
        [px1, py1, px2, py2],
        outline="red",
        width=5,
    )

    label = data.get("label", "stanchion")

    draw.text(
        (px1 + 5, py1 + 5),
        label,
        fill="red",
    )

    output_path = Path("stanchion_bbox_result.png")
    result_image.save(output_path)

    print("Saved:", output_path)
    result_image.show()

except Exception as e:
    print()
    print("NO VALID BOUNDING BOX")
    print("Reason:", e)
   
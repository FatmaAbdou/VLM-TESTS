import csv
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

MANIFEST = Path("benchmark/office_manifest.csv")
IMAGE_DIR = Path("benchmark/assets/office")


# --------------------------------------------------
# Load image
# --------------------------------------------------

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


# --------------------------------------------------
# Load processor
# --------------------------------------------------

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading model...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quant_config,
)

print("Model loaded.")


# --------------------------------------------------
# Prompt
# --------------------------------------------------

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": str(image_path),
            },
            {
                "type": "text",
                "text": (
                    "Locate the standalone stanchion closest to "
                    "the camera in the image.\n\n"
                    "Return ONLY four integers in this exact order:\n"
                    "x1,y1,x2,y2\n\n"
                    "Use the original image pixel coordinates.\n"
                    f"Image width: {width} pixels.\n"
                    f"Image height: {height} pixels.\n"
                    "x1,y1 is the top-left corner.\n"
                    "x2,y2 is the bottom-right corner.\n"
                    "Do not return JSON.\n"
                    "Do not return words.\n"
                    "Do not return markdown."
                ),
            },
        ],
    }
]


# --------------------------------------------------
# Qwen native preprocessing
# --------------------------------------------------

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


# --------------------------------------------------
# Generate
# --------------------------------------------------

print("Generating...")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )


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
)[0].strip()


# --------------------------------------------------
# Show raw output
# --------------------------------------------------

print()
print("MODEL OUTPUT:")
print(answer)


# --------------------------------------------------
# Extract coordinates
# --------------------------------------------------

numbers = re.findall(
    r"\b\d+(?:\.\d+)?\b",
    answer,
)

print()
print("Numbers found:", numbers)


if len(numbers) < 4:

    print()
    print("NO VALID BOUNDING BOX")
    print("Reason: Fewer than four coordinates found")

else:

    try:

        x1, y1, x2, y2 = [
            float(value)
            for value in numbers[:4]
        ]

        print()
        print(
            "Candidate bbox:",
            [x1, y1, x2, y2]
        )

        # --------------------------------------------------
        # Validate coordinates
        # --------------------------------------------------

        if not (0 <= x1 < x2 <= width):
            raise ValueError(
                "Invalid x coordinates"
            )

        if not (0 <= y1 < y2 <= height):
            raise ValueError(
                "Invalid y coordinates"
            )

        pixel_box = [
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        ]

        print()
        print(
            "VALID PIXEL BBOX:",
            pixel_box
        )


        # --------------------------------------------------
        # Draw bounding box
        # --------------------------------------------------

        result_image = image.copy()

        draw = ImageDraw.Draw(
            result_image
        )

        draw.rectangle(
            pixel_box,
            outline="red",
            width=5,
        )

        draw.text(
            (
                pixel_box[0],
                pixel_box[1],
            ),
            "standalone stanchion",
            fill="red",
        )


        # --------------------------------------------------
        # Save result
        # --------------------------------------------------

        output_path = Path(
            "qwen_stanchion_result.png"
        )

        result_image.save(
            output_path
        )

        print()
        print(
            "Saved:",
            output_path
        )


        # --------------------------------------------------
        # Open result
        # --------------------------------------------------

        result_image.show()


    except Exception as e:

        print()
        print("NO VALID BOUNDING BOX")
        print("Reason:", e)
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parent

MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
]


IMAGE_DIR = (
    ROOT
    / "benchmark"
    / "assets"
    / "office"
    / "Images"
)


# Use the first real ZED PNG we found.
image_files = sorted(
    IMAGE_DIR.glob("*.png")
)

if not image_files:
    raise FileNotFoundError(
        f"No PNG images found in {IMAGE_DIR}"
    )


image_path = image_files[0]

image = Image.open(
    image_path
).convert("RGB")


print()
print("=" * 70)
print("ORIGINAL ZED IMAGE")
print("=" * 70)

print(
    f"File: {image_path.name}"
)

print(
    f"Original size: {image.size}"
)

print(
    f"Original pixels: "
    f"{image.width * image.height:,}"
)


for model_name in MODELS:

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print("Loading processor...")

    processor = (
        AutoProcessor.from_pretrained(
            model_name
        )
    )

    image_processor = (
        processor.image_processor
    )

    print(
        f"Processor: "
        f"{type(image_processor).__name__}"
    )

    print(
        f"do_resize: "
        f"{getattr(image_processor, 'do_resize', 'N/A')}"
    )

    print(
        f"size: "
        f"{getattr(image_processor, 'size', 'N/A')}"
    )

    print(
        f"max_image_size: "
        f"{getattr(image_processor, 'max_image_size', 'N/A')}"
    )

    try:

        processed = image_processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = (
            processed.get(
                "pixel_values"
            )
        )

        if pixel_values is not None:

            print(
                f"Processed pixel_values "
                f"shape: "
                f"{tuple(pixel_values.shape)}"
            )

        else:

            print(
                "No pixel_values returned."
            )

    except Exception as e:

        print(
            f"PROCESSING ERROR: {e}"
        )


print()
print("=" * 70)
print("DONE")
print("=" * 70)
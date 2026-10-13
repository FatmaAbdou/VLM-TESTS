import argparse
import time
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

MIN_PIXELS = 3136
MAX_PIXELS = 28 * 28 * 768
DEFAULT_FPS = 1.0

FIRST_QUESTION = (
    "Describe the video from beginning to end. What kind of room or area "
    "is shown? Identify visible objects, spatial relationships, and any "
    "changes across the sampled frames. Distinguish direct observations "
    "from inferences. Do not claim to observe events that are not supported "
    "by the video frames."
)

SYSTEM_PROMPT = (
    "Answer questions about the supplied video using visible evidence. "
    "Distinguish observations from inferences and express uncertainty "
    "when appropriate. The video is represented by sampled frames, so do "
    "not claim to know what happened between those frames. Do not provide "
    "hidden chain-of-thought; give a concise explanation grounded in "
    "visible clues."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ask Qwen2.5-VL questions about a local video "
            "using native video input."
        )
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=ROOT / "recording_09" / "full.mp4",
        help="Video path (default: recording_09/full.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Target video sampling rate in frames per second (default: 1).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
        help="Maximum answer length in generated tokens.",
    )
    return parser.parse_args()


def make_initial_messages(video_path: Path, fps: float):
    video_uri = video_path.as_uri()

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_uri,
                    "fps": fps,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": (
                        f"{SYSTEM_PROMPT}\n\n"
                        f"Question:\n{FIRST_QUESTION}"
                    ),
                },
            ],
        }
    ]


def prepare_inputs(processor, messages, model):
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Native video processing. The utility decodes the video and
    # selects frames according to the video settings.
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
    )

    if video_inputs is None:
        raise RuntimeError(
            "No video frames were produced. Check the installed "
            "qwen-vl-utils version and video reader dependencies."
        )

    print("\nPreparing video inputs...")

    processor_kwargs = {
        "text": [prompt],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
        **video_kwargs,
    }

    inputs = processor(**processor_kwargs)
    inputs = inputs.to(model.device)

    # Report the number of temporal frames when available.
    video_grid = inputs.get("video_grid_thw")
    if video_grid is not None:
        grids = video_grid.detach().cpu().tolist()
        temporal_frames = sum(int(grid[0]) for grid in grids)
        print(
            "Video temporal grid values: "
            f"{grids}"
        )
        print(
            "Sum of temporal grid dimensions: "
            f"{temporal_frames}"
        )
        print(
            "Note: temporal grid dimensions are not guaranteed "
            "to equal the exact number of decoded source frames."
        )
    else:
        print(
            "The processor did not expose video_grid_thw; "
            "exact sampled-frame count is unavailable here."
        )

    return inputs


def answer_question(
    processor,
    model,
    messages,
    max_new_tokens,
    initial_video=False,
):
    start = time.perf_counter()

    if initial_video:
        # Prepare the full video context for the initial question.
        inputs = prepare_inputs(processor, messages, model)
    else:
        # Follow-up turns reuse the video in the conversation.
        # Reprocessing the conversation is required by this
        # simple Transformers chat implementation.
        inputs = prepare_inputs(processor, messages, model)

    preparation_seconds = time.perf_counter() - start

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_start = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start

    generated = [
        output[len(input_ids):]
        for input_ids, output in zip(inputs.input_ids, generated)
    ]

    answer = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    print(
        f"\nInput preparation: {preparation_seconds:.2f} seconds"
    )
    print(
        f"Generation: {generation_seconds:.2f} seconds"
    )

    return answer


def main():
    args = parse_args()
    video_path = args.video.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")

    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Activate the project's GPU environment."
        )

    print("=" * 60)
    print("QWEN2.5-VL NATIVE VIDEO QUESTION-ANSWERING")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Target sampling rate: {args.fps} FPS")
    print(f"Model: {MODEL_NAME}")

    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print("Loading model in 4-bit...")
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

    messages = make_initial_messages(video_path, args.fps)

    print("\nPreparing the video and asking the initial question...")
    print(f"\nYou: {FIRST_QUESTION}")

    try:
        answer = answer_question(
            processor,
            model,
            messages,
            args.max_new_tokens,
            initial_video=True,
        )
        print(f"\nQwen: {answer}\n")
    except Exception as exc:
        raise RuntimeError(
            "Video processing or initial inference failed. "
            "Check the installed Transformers and qwen-vl-utils versions."
        ) from exc

    print(
        "Ask a follow-up, type 'new' to reset the conversation, "
        "or 'quit' to exit.\n"
    )

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if question.lower() == "new":
            messages = make_initial_messages(video_path, args.fps)
            print(
                "Conversation reset. The video will be processed "
                "again for the next initial question.\n"
            )
            continue

        messages.append(
            {
                "role": "user",
                "content": (
                    f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}"
                ),
            }
        )

        try:
            answer = answer_question(
                processor,
                model,
                messages,
                args.max_new_tokens,
                initial_video=False,
            )
            print(f"\nQwen: {answer}\n")
        except Exception as exc:
            # Remove the failed user turn so the conversation remains usable.
            messages.pop()
            print(f"\nQuestion failed: {exc}\n")


if __name__ == "__main__":
    main()
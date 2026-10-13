import argparse
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
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 3136
MAX_PIXELS = 28 * 28 * 768
FIRST_QUESTION = (
    "What kind of room or area is shown in this video? Explain which "
    "visible features support that classification, and distinguish what "
    "you can directly see from what you infer. If the frames are not "
    "enough to identify the room confidently, say so."
)
SYSTEM_PROMPT = (
    "Answer questions about the supplied video frames using visible "
    "evidence. Distinguish observations from inferences and express "
    "uncertainty when appropriate. The frames are sparse samples, so do not "
    "claim to know what happened between them. Do not provide hidden "
    "chain-of-thought; give a concise explanation grounded in visible clues."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat with Qwen2.5-VL about a video using sampled frames."
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=ROOT / "recording_09" / "full.mp4",
        help="Video path (default: recording_09/full.mp4).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=6,
        help="Number of frames sampled evenly across the video (default: 6).",
    )
    return parser.parse_args()


def sample_video(video_path: Path, num_frames: int):
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if num_frames < 1:
        raise ValueError("--num-frames must be at least 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if frame_count < 1 or fps <= 0:
        capture.release()
        raise RuntimeError(f"Could not read video duration: {video_path}")

    if num_frames == 1:
        frame_indices = [frame_count // 2]
    else:
        frame_indices = [
            round(index * (frame_count - 1) / (num_frames - 1))
            for index in range(num_frames)
        ]

    frames = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read video frame {frame_index}: {video_path}"
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(
                {
                    "timestamp": frame_index / fps,
                    "image": Image.fromarray(frame),
                }
            )
    finally:
        capture.release()

    return frames


def make_video_message(frames, question):
    content = []
    for frame in frames:
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"Video frame sampled at "
                        f"{frame['timestamp']:.1f} seconds:"
                    ),
                },
                {"type": "image", "image": frame["image"]},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}",
        }
    )
    return {"role": "user", "content": content}


def answer_question(processor, model, messages):
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
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    generated = [
        output[len(input_ids):]
        for input_ids, output in zip(inputs.input_ids, generated)
    ]
    answer = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    messages.append({"role": "assistant", "content": answer})
    return answer


def main():
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    frames = sample_video(video_path, args.num_frames)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Activate the project's GPU environment."
        )

    print(f"Video: {video_path}")
    print(
        "Sampled frames at: "
        + ", ".join(f"{frame['timestamp']:.1f}s" for frame in frames)
    )
    print(f"Loading {MODEL_NAME} in 4-bit...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
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

    messages = [
        make_video_message(frames, FIRST_QUESTION),
    ]

    print(f"\nYou: {FIRST_QUESTION}")
    answer = answer_question(processor, model, messages)
    print(f"\nQwen: {answer}\n")
    print("Ask a follow-up, or type 'new' to reset / 'quit' to exit.\n")

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
            messages = [
                make_video_message(frames, FIRST_QUESTION),
            ]
            print("Conversation cleared; the video is still loaded.\n")
            continue

        messages.append(
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}",
            }
        )
        answer = answer_question(processor, model, messages)
        print(f"\nQwen: {answer}\n")


if __name__ == "__main__":
    main()

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import csv

root = Path("benchmark/assets/office")
image_dir = root / "Images"
videos_dir = root / "Videos"
output = root / "frame_match_candidates.csv"

def load_small(path):
    with Image.open(path) as im:
        return np.asarray(
            im.convert("RGB").resize((96, 60), Image.Resampling.BILINEAR),
            dtype=np.int16
        )

targets = [
    (p, load_small(p))
    for p in sorted(image_dir.iterdir())
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
]

rows = []
for video_dir in sorted(videos_dir.glob("recording_*")):
    video_path = video_dir / "full.mp4"
    if not video_path.exists():
        continue

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        continue

    samples = []
    for idx in range(0, frame_count, max(1, round(fps))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (96, 60), interpolation=cv2.INTER_AREA)
        samples.append((idx, frame.astype(np.int16)))
    cap.release()

    for target_path, target in targets:
        best_score, best_frame = float("inf"), None
        for idx, sample in samples:
            score = float(np.abs(target - sample).mean())
            if score < best_score:
                best_score, best_frame = score, idx
        rows.append({
            "benchmark_image": target_path.name,
            "recording": video_dir.name,
            "video_frame_index": best_frame,
            "elapsed_seconds": round(best_frame / fps, 3) if best_frame is not None else "",
            "mae": round(best_score, 3),
        })

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "benchmark_image", "recording", "video_frame_index",
        "elapsed_seconds", "mae"
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Done. Saved {len(rows)} candidates to: {output}")
print("No benchmark or result files were changed.")

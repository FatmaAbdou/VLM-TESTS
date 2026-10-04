from pathlib import Path
import csv
import cv2
import numpy as np
from PIL import Image

video_path = Path("benchmark/assets/office/Videos/recording_04/full.mp4")
manifest_path = Path("benchmark/office_manifest.csv")
output_path = Path("benchmark/assets/office/recording_04_fine_matches.csv")
image_dir = Path("benchmark/assets/office/Images")

with manifest_path.open(encoding="utf-8-sig", newline="") as f:
    manifest = list(csv.DictReader(f))

targets = []
for row in manifest:
    n = int(row["frame_id"].split("_")[1])
    if 33 <= n <= 54:
        path = image_dir / row["filename"]
        if path.exists():
            image = Image.open(path).convert("RGB").resize((96, 60))
            targets.append((row["frame_id"], row["filename"],
                            np.asarray(image, dtype=np.int16)))

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
print(f"Video: {fps:.3f} FPS, {duration:.2f} seconds")
print(f"Target images loaded: {len(targets)}")

results = []
step = max(1, round(fps / 2))  # scan every 0.5 seconds, not every 0.1

for index in range(0, frame_count, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        continue

    seconds = index / fps
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (96, 60)).astype(np.int16)

    for frame_id, filename, target in targets:
        mae = float(np.abs(frame - target).mean())
        results.append({
            "frame_id": frame_id,
            "filename": filename,
            "recording": "recording_04",
            "elapsed_seconds": f"{seconds:.2f}",
            "video_frame_index_approx": index,
            "mae": f"{mae:.3f}",
        })

    if index % (step * 10) == 0:
        print(f"Scanned {seconds:.1f}s / {duration:.1f}s")

cap.release()

if not results:
    raise RuntimeError("No matches were calculated; check the video and image paths.")

with output_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved {len(results)} comparisons to {output_path}")
print("\nBest candidate per benchmark frame:")

for frame_id, _, _ in targets:
    candidates = [r for r in results if r["frame_id"] == frame_id]
    best = min(candidates, key=lambda r: float(r["mae"]))
    print(f'{frame_id}: {best["elapsed_seconds"]}s, MAE={best["mae"]}')

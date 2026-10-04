import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark/office_v2_zed93.json"
MANIFEST = ROOT / "benchmark/office_manifest.csv"
CANDIDATES = ROOT / "benchmark/assets/office/frame_match_candidates.csv"
OUTPUT = ROOT / "results/video_case_shortlist.txt"

# Map frame IDs (frame_001, etc.) to original image filenames.
with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    manifest = {
        row["frame_id"]: Path(row["filename"]).name
        for row in csv.DictReader(f)
    }

# Group candidate video matches by original image filename.
matches = {}
with CANDIDATES.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        name = Path(row["benchmark_image"]).name
        matches.setdefault(name, []).append(row)

with BENCHMARK.open(encoding="utf-8") as f:
    benchmark = json.load(f)

tests = benchmark["tests"]
lines = [
    "VIDEO MATCH SHORTLIST",
    "Best available recording/time matches for benchmark test frames.",
    "MAE is a visual similarity score; lower is better.",
    "These are candidate matches, not confirmed mappings.",
    "",
]

for test in tests:
    test_id = test.get("test_id", "UNKNOWN")
    question = test.get("question", "")
    frames = test.get("input_media", [])

    lines.append(f"{test_id} | {test.get('category', '')}")
    lines.append(f"Question: {question}")

    for frame_id in frames:
        filename = manifest.get(frame_id)
        lines.append(f"  {frame_id} -> {filename or 'NOT IN MANIFEST'}")

        candidates = matches.get(filename, []) if filename else []
        candidates = sorted(candidates, key=lambda r: float(r["mae"]))

        if not candidates:
            lines.append("    No candidate match found.")
            continue

        # Show the best two distinct recording/time candidates.
        shown = []
        seen = set()
        for row in candidates:
            key = (row["recording"], row["elapsed_seconds"])
            if key in seen:
                continue
            seen.add(key)
            shown.append(row)
            if len(shown) == 2:
                break

        for row in shown:
            lines.append(
                f"    {row['recording']} at {float(row['elapsed_seconds']):.1f}s"
                f" | MAE {float(row['mae']):.3f}"
            )

    lines.append("")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved shortlist to: {OUTPUT}")
print(f"Benchmark cases processed: {len(tests)}")
print(f"Frames mapped in manifest: {len(manifest)}")

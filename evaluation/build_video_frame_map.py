import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark/office_manifest.csv"
CANDIDATES = ROOT / "benchmark/assets/office/frame_match_candidates.csv"
OUTPUT = ROOT / "benchmark/video_frame_map_draft.csv"

with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    manifest_rows = list(csv.DictReader(f))

by_image = {}
with CANDIDATES.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        filename = Path(row["benchmark_image"]).name
        by_image.setdefault(filename, []).append(row)

output_rows = []

for item in manifest_rows:
    filename = Path(item["filename"]).name
    candidates = sorted(
        by_image.get(filename, []),
        key=lambda row: float(row["mae"])
    )

    row = {
        "frame_id": item["frame_id"],
        "sequence_order": item["sequence_order"],
        "benchmark_image": filename,
        "mapping_status": "UNVERIFIED",
        "candidate_1_recording": "",
        "candidate_1_seconds": "",
        "candidate_1_mae": "",
        "candidate_2_recording": "",
        "candidate_2_seconds": "",
        "candidate_2_mae": "",
        "candidate_3_recording": "",
        "candidate_3_seconds": "",
        "candidate_3_mae": "",
        "verified_recording": "",
        "verified_seconds": "",
        "verification_notes": "",
    }

    for rank, candidate in enumerate(candidates[:3], start=1):
        row[f"candidate_{rank}_recording"] = candidate["recording"]
        row[f"candidate_{rank}_seconds"] = candidate["elapsed_seconds"]
        row[f"candidate_{rank}_mae"] = candidate["mae"]

    if not candidates:
        row["mapping_status"] = "NO_CANDIDATE"

    output_rows.append(row)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
    writer.writeheader()
    writer.writerows(output_rows)

print(f"Created: {OUTPUT}")
print(f"Frames mapped: {len(output_rows)}")
print("All candidate matches remain UNVERIFIED.")
print("No frozen benchmark or model results were changed.")

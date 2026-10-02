import json
from pathlib import Path

benchmark = Path("benchmark")

with open(benchmark / "office_v2_frozen.json", encoding="utf-8") as f:
    original = json.load(f)

with open(benchmark / "office_v2_zed93.json", encoding="utf-8") as f:
    remapped = json.load(f)

original_tests = {t["test_id"]: t for t in original["tests"]}
remapped_tests = {t["test_id"]: t for t in remapped["tests"]}

errors = []

# ------------------------------------------------------------
# Basic structure
# ------------------------------------------------------------

if set(original_tests) != set(remapped_tests):
    errors.append("Test IDs changed!")

if len(remapped_tests) != 49:
    errors.append("Unexpected test count!")

# ------------------------------------------------------------
# Fields that must remain unchanged
# ------------------------------------------------------------

preserved_fields = [
    "question",
    "ground_truth",
    "scoring_mode",
]

for test_id, old in original_tests.items():
    new = remapped_tests[test_id]

    for field in preserved_fields:
        if old.get(field) != new.get(field):
            errors.append(f"{test_id}: {field} changed")

# ------------------------------------------------------------
# Accepted answers
#
# Accepted answers are preserved except for known stale frame
# references that had to be updated after remapping to ZED-93.
# ------------------------------------------------------------

for test_id, old in original_tests.items():
    new = remapped_tests[test_id]

    old_answers = old.get("accepted_answers")
    new_answers = new.get("accepted_answers")

    if test_id == "OFFICE-044":
        expected_answers = list(old_answers)

        # Update only the stale frame references.
        expected_answers[2] = (
            "yes, it is visible in frame_079 and frame_080 "
            "but not frame_082"
        )

        if new_answers != expected_answers:
            errors.append(
                "OFFICE-044: accepted_answers do not match the "
                "approved ZED-93 correction"
            )
    else:
        if old_answers != new_answers:
            errors.append(f"{test_id}: accepted_answers changed")

# ------------------------------------------------------------
# OFFICE-028 mapping correction
# ------------------------------------------------------------

test_028 = remapped_tests["OFFICE-028"]

if test_028.get("input_media") != ["frame_005", "frame_006"]:
    errors.append(
        "OFFICE-028: input_media must be ['frame_005', 'frame_006']"
    )

notes_028 = test_028.get("note", test_028.get("notes", "")).lower()

if "duplicate-image comparison" in notes_028:
    errors.append(
        "OFFICE-028: stale duplicate-image comparison note remains"
    )

# ------------------------------------------------------------
# OFFICE-044 mapping correction
# ------------------------------------------------------------

test_044 = remapped_tests["OFFICE-044"]

if test_044.get("input_media") != [
    "frame_079",
    "frame_080",
    "frame_082",
]:
    errors.append(
        "OFFICE-044: input_media must use frame_079, "
        "frame_080, and frame_082"
    )

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

if errors:
    print("VALIDATION FAILED:")
    for error in errors:
        print(f" - {error}")
else:
    print("PASS: all 49 test IDs are preserved.")
    print("PASS: questions and ground-truth fields are unchanged.")
    print("PASS: accepted answers are unchanged except for the approved OFFICE-044 frame-reference correction.")
    print("PASS: OFFICE-028 uses frame_005 and frame_006.")
    print("PASS: OFFICE-044 uses frame_079, frame_080, and frame_082.")
    print("PASS: original frozen benchmark was not modified by this check.")
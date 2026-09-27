import json
import re
from pathlib import Path


ROOT = Path("/home/lyth2/VLM-TESTS")
NAV_DIR = ROOT / "datasets" / "navigation"

AUDIT_FILE = NAV_DIR / "candidate_audit.json"
METADATA_FILE = NAV_DIR / "metadata.json"
OUTPUT_FILE = NAV_DIR / "benchmark_candidates.json"


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def get_first(d, *keys, default=None):
    for key in keys:
        if isinstance(d, dict) and key in d and d[key] is not None:
            return d[key]
    return default


def classify_candidate(task, context, categories):
    """
    Conservative classification.

    Important:
    This does NOT attempt to infer every benchmark capability.
    NaviTrace is primarily being used for navigation/instruction tasks.
    """

    text = f"{task} {context}".lower()
    categories_lower = {str(x).lower() for x in categories}

    # Explicit obstacle / avoidance evidence.
    obstacle_terms = [
        "blocked",
        "barrier",
        "construction",
        "obstacle",
        "puddle",
        "avoid",
        "blocks",
        "blocking",
        "high curb",
        "stairs",
        "ramp",
    ]

    if (
        "stationary obstacle" in categories_lower
        or any(term in text for term in obstacle_terms)
    ):
        return "Obstacle / Navigation"

    # Visibility is only assigned when visibility is actually relevant.
    if "visibility" in categories_lower:
        visibility_terms = [
            "blocked",
            "hidden",
            "visibility",
            "obscured",
            "cannot see",
            "not visible",
        ]

        if any(term in text for term in visibility_terms):
            return "Visibility / Occlusion"

    # Dynamic obstacles only when the actual navigation situation involves
    # people/cars/traffic/etc.
    dynamic_terms = [
        "people",
        "person",
        "crowd",
        "car",
        "cars",
        "traffic",
        "bicycle",
        "scooter",
        "tram",
    ]

    if (
        "dynamic obstacle" in categories_lower
        and any(term in text for term in dynamic_terms)
    ):
        return "Obstacle / Navigation"

    # Default for NaviTrace.
    return "Instruction Following"


def build_question(task, context, capability):
    """
    Conservative candidate wording.

    We intentionally preserve the original NaviTrace task rather than
    inventing a new visual question.
    """

    task = clean_text(task)

    if not task:
        return None

    if capability == "Instruction Following":
        return task

    if capability == "Obstacle / Navigation":
        return task

    if capability == "Visibility / Occlusion":
        return task

    return task


def extract_records(audit):
    """
    Supports either:
      - a list
      - {"samples": [...]}
      - {"candidates": [...]}
      - {"records": [...]}
    """

    if isinstance(audit, list):
        return audit

    if isinstance(audit, dict):
        for key in ["samples", "candidates", "records", "items", "data"]:
            if isinstance(audit.get(key), list):
                return audit[key]

    raise ValueError(
        "Could not find a list of samples in candidate_audit.json"
    )


def main():
    if not AUDIT_FILE.exists():
        raise FileNotFoundError(AUDIT_FILE)

    with open(AUDIT_FILE, "r", encoding="utf-8") as f:
        audit = json.load(f)

    records = extract_records(audit)

    metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    candidates = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue

        source_id = get_first(
            record,
            "id",
            "source_id",
            "sample_id",
            "uuid",
        )

        task = get_first(
            record,
            "task",
            "source_task",
            "instruction",
        )

        context = get_first(
            record,
            "context",
            "source_context",
            default="",
        )

        categories = get_first(
            record,
            "categories",
            "category",
            "source_categories",
            default=[],
        )

        if isinstance(categories, str):
            categories = [categories]

        task_type = get_first(
            record,
            "task_type",
            default=None,
        )

        city = get_first(
            record,
            "city",
            default=None,
        )

        # Some audit formats keep metadata nested.
        nested_metadata = record.get("metadata", {})

        if isinstance(nested_metadata, dict):
            task_type = task_type or nested_metadata.get("task_type")
            city = city or nested_metadata.get("city")

        if not source_id or not task:
            continue

        capability = classify_candidate(
            task,
            context,
            categories,
        )

        question = build_question(
            task,
            context,
            capability,
        )

        if not question:
            continue

        # Image filename convention used by our navigation subset.
        image_path = f"images/{source_id}.jpg"

        candidates.append(
            {
                "candidate_id": f"NAVI-{len(candidates) + 1:03d}",
                "source": "NaviTrace",
                "source_split": "validation",
                "source_id": source_id,
                "image": image_path,

                "candidate_capability": capability,

                "question": question,

                # Preserve source evidence.
                "source_task": task,
                "source_context": context,
                "source_categories": categories,

                "task_type": task_type,
                "city": city,

                # We do NOT invent an answer here.
                "candidate_answer": None,

                # Original NaviTrace annotation remains the source GT.
                "ground_truth_status": "source_annotation_available",

                # Critical: this is not yet a validated benchmark item.
                "validation_status": "needs_visual_validation",

                "validation_notes": [
                    "Question is preserved from the original NaviTrace task.",
                    "Image must be visually inspected before benchmark inclusion.",
                    "Answer must be validated against the image and source annotation.",
                    "Do not infer unsupported visual facts from metadata alone."
                ]
            }
        )

    output = {
        "benchmark": "VLM Evaluation Benchmark",
        "dataset": "NaviTrace",
        "source": "leggedrobotics/navitrace",
        "split": "validation",

        "purpose": (
            "Candidate pool for Instruction Following and "
            "Obstacle / Navigation evaluation."
        ),

        "candidate_count": len(candidates),

        "validation_policy": {
            "status_meanings": {
                "needs_visual_validation": (
                    "Candidate has not yet been manually/visually validated."
                ),
                "validated": (
                    "Image, question, answer, and evidence have been checked."
                ),
                "rejected": (
                    "Candidate is unsuitable or ambiguous."
                )
            }
        },

        "candidates": candidates
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Created: {OUTPUT_FILE}")
    print(f"Candidates: {len(candidates)}")

    counts = {}

    for item in candidates:
        capability = item["candidate_capability"]
        counts[capability] = counts.get(capability, 0) + 1

    print("\nCandidate capabilities:")
    for capability, count in sorted(counts.items()):
        print(f"  {capability}: {count}")


if __name__ == "__main__":
    main()
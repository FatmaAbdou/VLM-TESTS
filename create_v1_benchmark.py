import json
import random
import shutil
from pathlib import Path

from datasets import load_dataset, load_from_disk
from PIL import Image


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

BENCHMARK_DIR = ROOT / "benchmark"
ASSETS_DIR = BENCHMARK_DIR / "assets"

TEMPORAL_ASSETS = ASSETS_DIR / "temporal"
GROUNDING_ASSETS = ASSETS_DIR / "grounding"
NAVIGATION_ASSETS = ASSETS_DIR / "navigation"

OUTPUT_FILE = BENCHMARK_DIR / "benchmark.json"

SEED = 42
random.seed(SEED)


# ============================================================
# BENCHMARK DISTRIBUTION
# ============================================================

MVBENCH_CASES_PER_CATEGORY = 8

REFCOCOG_CASES = 10
REFCOCOG_GROUNDING_CASES = 6
REFCOCOG_VISIBILITY_CASES = 4

NAVITRACE_CASES = 10
NAVITRACE_INSTRUCTION_CASES = 5
NAVITRACE_NAVIGATION_CASES = 5


# ============================================================
# DIRECTORY SETUP
# ============================================================

for directory in [
    BENCHMARK_DIR,
    TEMPORAL_ASSETS,
    GROUNDING_ASSETS,
    NAVIGATION_ASSETS,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# GENERAL HELPERS
# ============================================================

def save_image(image, output_path):
    """
    Save a HuggingFace/PIL image to a normal local file.
    """
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        image.save(output_path)
        return

    if isinstance(image, dict):
        if image.get("bytes") is not None:
            from io import BytesIO

            img = Image.open(BytesIO(image["bytes"]))
            img.convert("RGB").save(output_path)
            return

        if image.get("path"):
            shutil.copy2(image["path"], output_path)
            return

    raise TypeError(
        f"Unsupported image type: {type(image)}"
    )


def reset_directory(directory):
    """
    Remove previously generated benchmark assets.
    """
    directory.mkdir(parents=True, exist_ok=True)

    for item in directory.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def json_safe(value):
    """
    Convert common dataset values into JSON-serializable values.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


# ============================================================
# MVBench Temporal Conflict
# ============================================================

def load_mvbench():
    print("\nLoading MVBench Temporal Conflict...")

    dataset = load_dataset(
        "shivank21/mvbench-temporal-conflict-subset",
        split="train",
    )

    print(f"MVBench rows: {len(dataset)}")

    return dataset


def find_video(row):
    """
    Locate a downloaded MVBench video.
    """
    candidates = []

    if row.get("video_path"):
        candidates.append(
            Path(str(row["video_path"]))
        )

    if row.get("video"):
        candidates.append(
            Path(str(row["video"]))
        )

    video_name = Path(
        str(row.get("video", ""))
    ).name

    local_video_dir = (
        ROOT
        / "datasets"
        / "temporal"
        / "videos"
    )

    if video_name:
        candidates.append(
            local_video_dir / video_name
        )

    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate

        if not candidate.is_absolute():
            project_candidate = ROOT / candidate

            if project_candidate.exists():
                return project_candidate

    if video_name and local_video_dir.exists():
        matches = list(
            local_video_dir.rglob(video_name)
        )

        if matches:
            return matches[0]

    return None


def mvbench_capability(group):
    """
    Map MVBench source categories to the agreed
    benchmark taxonomy.
    """
    mapping = {
        "moving_attribute": "Attribute Recognition",
        "moving_count": "Counting",
        "object_existence": "Change Detection",
        "moving_direction": "Spatial Reasoning",
        "object_interaction": "Temporal Consistency",
    }

    return mapping[group]


def build_mvbench_cases():
    dataset = load_mvbench()

    groups = [
        "moving_attribute",
        "moving_count",
        "object_existence",
        "moving_direction",
        "object_interaction",
    ]

    cases = []

    for group in groups:
        capability = mvbench_capability(group)

        candidates = []

        for index in range(len(dataset)):
            row = dataset[index]

            row_id = str(
                row.get("id", "")
            )

            split_value = str(
                row.get("split", "")
            )

            if (
                split_value == group
                or row_id.startswith(group)
            ):
                candidates.append(row)

        if len(candidates) < MVBENCH_CASES_PER_CATEGORY:
            raise RuntimeError(
                f"MVBench category '{group}' has only "
                f"{len(candidates)} usable rows; "
                f"need {MVBENCH_CASES_PER_CATEGORY}."
            )

        candidates = sorted(
            candidates,
            key=lambda row: str(row.get("id", "")),
        )

        selected = candidates[
            :MVBENCH_CASES_PER_CATEGORY
        ]

        print(
            f"MVBench {group}: "
            f"{len(selected)}/{MVBENCH_CASES_PER_CATEGORY}"
        )

        for row in selected:
            video = find_video(row)

            if video is None:
                raise FileNotFoundError(
                    "MVBench video not found: "
                    f"{row.get('video')}"
                )

            test_number = len(cases) + 1

            test_id = (
                f"MVBench-{test_number:03d}"
            )

            destination = (
                TEMPORAL_ASSETS
                / f"{test_id}.mp4"
            )

            shutil.copy2(
                video,
                destination,
            )

            candidates_value = row.get(
                "candidates"
            )

            cases.append(
                {
                    "test_id": test_id,
                    "dataset": (
                        "MVBench Temporal Conflict"
                    ),
                    "source_category": group,
                    "capability": capability,
                    "input_type": "video",
                    "video": str(
                        destination.relative_to(ROOT)
                    ),
                    "question": str(
                        row["question"]
                    ),
                    "choices": json_safe(
                        candidates_value
                    ),
                    "expected_answer": str(
                        row["answer"]
                    ),
                    "answer_type": (
                        "multiple_choice"
                        if candidates_value
                        else "text"
                    ),
                    "scoring_mode": "choice",
                    "source_id": str(
                        row.get("id")
                    ),
                    "source_dataset": str(
                        row.get("source_dataset", "")
                    ),
                    "conflict_mechanism": str(
                        row.get(
                            "conflict_mechanism",
                            "",
                        )
                    ),
                }
            )

    print(
        f"MVBench selected: {len(cases)}"
    )

    return cases


# ============================================================
# RefCOCOg / SoM
# ============================================================

def load_grounding():
    print("\nLoading RefCOCOg...")

    path = (
        ROOT
        / "datasets"
        / "grounding"
        / "dataset"
    )

    dataset = load_from_disk(
        str(path)
    )

    print(
        f"RefCOCOg rows: {len(dataset)}"
    )

    return dataset


def normalize_expression(value):
    """
    RefCOCOg obj_text entries can be strings or lists.
    """
    if isinstance(value, list):
        if not value:
            return ""

        return str(value[0])

    return str(value)


def extract_referring_expressions(row):
    """
    Return individual referring expressions paired
    with their actual source ref_id.
    """
    obj_text = row.get("obj_text", [])
    ref_ids = row.get("ref_ids", [])

    expressions = []

    if not isinstance(obj_text, list):
        obj_text = [obj_text]

    for index, text_value in enumerate(obj_text):
        if index >= len(ref_ids):
            break

        expression = normalize_expression(
            text_value
        )

        if not expression:
            continue

        expressions.append(
            {
                "expression": expression,
                "ref_id": json_safe(
                    ref_ids[index]
                ),
            }
        )

    return expressions


def is_visibility_expression(expression):
    """
    Identify expressions that explicitly describe
    visibility or partial occlusion.
    """
    text = expression.lower()

    visibility_terms = [
        "visible",
        "barely",
        "half visible",
        "partially",
        "occluded",
        "blocked",
        "hidden",
        "showing",
        "barely showing",
        "only half",
    ]

    return any(
        term in text
        for term in visibility_terms
    )


def build_grounding_cases():
    dataset = load_grounding()

    all_references = []
    visibility_references = []

    for index in range(len(dataset)):
        row = dataset[index]

        expressions = extract_referring_expressions(
            row
        )

        for reference in expressions:
            item = {
                "row_index": index,
                "row": row,
                "expression": reference[
                    "expression"
                ],
                "ref_id": reference[
                    "ref_id"
                ],
            }

            all_references.append(item)

            if is_visibility_expression(
                reference["expression"]
            ):
                visibility_references.append(
                    item
                )

    # Grounding cases
    grounding_cases = []

    seen_images = set()

    for item in all_references:
        if len(grounding_cases) >= (
            REFCOCOG_GROUNDING_CASES
        ):
            break

        row_index = item["row_index"]

        if row_index in seen_images:
            continue

        if is_visibility_expression(
            item["expression"]
        ):
            continue

        seen_images.add(row_index)
        grounding_cases.append(item)

    # Visibility cases
    visibility_cases = []

    seen_visibility_images = set()

    for item in visibility_references:
        if len(visibility_cases) >= (
            REFCOCOG_VISIBILITY_CASES
        ):
            break

        row_index = item["row_index"]

        if row_index in seen_visibility_images:
            continue

        seen_visibility_images.add(row_index)
        visibility_cases.append(item)

    if len(grounding_cases) < (
        REFCOCOG_GROUNDING_CASES
    ):
        raise RuntimeError(
            "Could not find enough RefCOCOg "
            "grounding cases."
        )

    if len(visibility_cases) < (
        REFCOCOG_VISIBILITY_CASES
    ):
        raise RuntimeError(
            "Could not find enough RefCOCOg "
            "visibility/occlusion cases. "
            f"Found {len(visibility_cases)}, "
            f"need {REFCOCOG_VISIBILITY_CASES}."
        )

    selected = []

    for item in grounding_cases:
        selected.append(
            (
                "Grounding / Instance Selection",
                item,
            )
        )

    for item in visibility_cases:
        selected.append(
            (
                "Visibility / Occlusion",
                item,
            )
        )

    cases = []

    for index, (
        capability,
        item,
    ) in enumerate(
        selected,
        start=1,
    ):
        row = item["row"]

        test_id = (
            f"REFCOCOG-{index:03d}"
        )

        image_path = (
            GROUNDING_ASSETS
            / f"{test_id}.png"
        )

        save_image(
            row["image"],
            image_path,
        )

        expression = item[
            "expression"
        ]

        cases.append(
            {
                "test_id": test_id,
                "dataset": "SoM RefCOCOg",
                "capability": capability,
                "input_type": "image",
                "image": str(
                    image_path.relative_to(ROOT)
                ),
                "question": (
                    "Which object in the image "
                    "does the following referring "
                    "expression identify?\n\n"
                    f"{expression}"
                ),
                "reference_expression": expression,
                "ground_truth_ref_id": item[
                    "ref_id"
                ],
                "answer_type": "grounding",
                "scoring_mode": "manual_grounding",
                "source_id": str(
                    row.get("id", "")
                ),
            }
        )

    print(
        f"RefCOCOg selected: {len(cases)}"
    )

    print(
        "  Grounding / Instance Selection: "
        f"{REFCOCOG_GROUNDING_CASES}"
    )

    print(
        "  Visibility / Occlusion: "
        f"{REFCOCOG_VISIBILITY_CASES}"
    )

    return cases
# ============================================================
# NaviTrace
# ============================================================

def load_navigation_candidates():
    path = (
        ROOT
        / "datasets"
        / "navigation"
        / "benchmark_candidates.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    candidates = data["candidates"]

    if len(candidates) != int(
        data["candidate_count"]
    ):
        raise ValueError(
            "NaviTrace candidate manifest is inconsistent: "
            f"candidate_count={data['candidate_count']} "
            f"but candidates={len(candidates)}"
        )

    return candidates


def load_navigation_dataset():
    print("\nLoading NaviTrace...")

    dataset = load_dataset(
        "leggedrobotics/navitrace",
        split="validation",
    )

    print(
        f"NaviTrace rows: {len(dataset)}"
    )

    return dataset


def build_navigation_cases():
    candidates = (
        load_navigation_candidates()
    )

    dataset = load_navigation_dataset()

    by_id = {}

    for index in range(len(dataset)):
        row = dataset[index]

        source_id = str(
            row.get("sample_id", "")
        )

        if source_id:
            by_id[source_id] = row

    instruction_candidates = [
        candidate
        for candidate in candidates
        if candidate.get(
            "candidate_capability"
        ) == "Instruction Following"
    ]

    navigation_candidates = [
        candidate
        for candidate in candidates
        if candidate.get(
            "candidate_capability"
        ) == "Obstacle / Navigation"
    ]

    if len(instruction_candidates) < (
        NAVITRACE_INSTRUCTION_CASES
    ):
        raise RuntimeError(
            "Not enough NaviTrace Instruction Following "
            "candidates."
        )

    if len(navigation_candidates) < (
        NAVITRACE_NAVIGATION_CASES
    ):
        raise RuntimeError(
            "Not enough NaviTrace Obstacle / Navigation "
            "candidates."
        )

    # Preserve the ordering of the validated candidate
    # manifest.
    selected = (
        instruction_candidates[
            :NAVITRACE_INSTRUCTION_CASES
        ]
        + navigation_candidates[
            :NAVITRACE_NAVIGATION_CASES
        ]
    )

    cases = []

    for index, candidate in enumerate(
        selected,
        start=1,
    ):
        source_id = str(
            candidate["source_id"]
        )

        row = by_id.get(source_id)

        if row is None:
            raise RuntimeError(
                "NaviTrace source ID not found: "
                f"{source_id}"
            )

        test_id = (
            f"NAVITRACE-{index:03d}"
        )

        image_path = (
            NAVIGATION_ASSETS
            / f"{test_id}.png"
        )

        save_image(
            row["image"],
            image_path,
        )

        ground_truth = json_safe(
            row.get("ground_truth")
        )

        cases.append(
            {
                "test_id": test_id,
                "dataset": "NaviTrace",
                "capability": candidate[
                    "candidate_capability"
                ],
                "input_type": "image",
                "image": str(
                    image_path.relative_to(ROOT)
                ),
                "question": str(
                    candidate["question"]
                ),
                "expected_answer": ground_truth,
                "answer_type": (
                    "navigation_grounding"
                ),
                "scoring_mode": "manual_navigation",
                "source_id": source_id,
                "source_context": candidate.get(
                    "source_context"
                ),
                "source_categories": (
                    candidate.get(
                        "source_categories"
                    )
                ),
                "task_type": candidate.get(
                    "task_type"
                ),
                "city": candidate.get(
                    "city"
                ),
                "ground_truth_status": candidate.get(
                    "ground_truth_status"
                ),
                "validation_status": candidate.get(
                    "validation_status"
                ),
            }
        )

    print(
        f"NaviTrace selected: {len(cases)}"
    )

    print(
        "  Instruction Following: "
        f"{NAVITRACE_INSTRUCTION_CASES}"
    )

    print(
        "  Obstacle / Navigation: "
        f"{NAVITRACE_NAVIGATION_CASES}"
    )

    return cases


# ============================================================
# BENCHMARK VALIDATION
# ============================================================

def validate_benchmark(benchmark):
    """
    Validate the generated V1 manifest before writing it.
    """

    expected_total = (
        (
            MVBENCH_CASES_PER_CATEGORY
            * 5
        )
        + REFCOCOG_CASES
        + NAVITRACE_CASES
    )

    if len(benchmark) != expected_total:
        raise RuntimeError(
            "Unexpected benchmark size: "
            f"{len(benchmark)}; "
            f"expected {expected_total}."
        )

    test_ids = [
        case["test_id"]
        for case in benchmark
    ]

    if len(test_ids) != len(set(test_ids)):
        raise RuntimeError(
            "Duplicate test IDs detected."
        )

    for case in benchmark:
        required_fields = [
            "test_id",
            "dataset",
            "capability",
            "input_type",
            "question",
            "scoring_mode",
        ]

        for field in required_fields:
            if field not in case:
                raise RuntimeError(
                    f"{case.get('test_id', '<unknown>')} "
                    f"is missing required field '{field}'."
                )

        if case["input_type"] == "video":
            path = ROOT / case["video"]

            if not path.exists():
                raise RuntimeError(
                    f"Missing video asset: {path}"
                )

        if case["input_type"] == "image":
            path = ROOT / case["image"]

            if not path.exists():
                raise RuntimeError(
                    f"Missing image asset: {path}"
                )

    # Capability counts.
    capability_counts = {}

    for case in benchmark:
        capability = case["capability"]

        capability_counts[capability] = (
            capability_counts.get(
                capability,
                0,
            )
            + 1
        )

    print("\nCapability coverage:")

    for capability in sorted(
        capability_counts
    ):
        print(
            f"  {capability}: "
            f"{capability_counts[capability]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CREATING VLM BENCHMARK V1")
    print("=" * 60)

    print("\nV1 distribution:")
    print(
        "  MVBench: "
        f"{MVBENCH_CASES_PER_CATEGORY * 5}"
    )
    print(
        "  RefCOCOg: "
        f"{REFCOCOG_CASES}"
    )
    print(
        "  NaviTrace: "
        f"{NAVITRACE_CASES}"
    )
    print(
        "  Total: "
        f"{(MVBENCH_CASES_PER_CATEGORY * 5) + REFCOCOG_CASES + NAVITRACE_CASES}"
    )

    # Remove assets from previous benchmark generations.
    print("\nCleaning previous benchmark assets...")

    reset_directory(TEMPORAL_ASSETS)
    reset_directory(GROUNDING_ASSETS)
    reset_directory(NAVIGATION_ASSETS)

    benchmark = []

    # --------------------------------------------------------
    # MVBench
    # --------------------------------------------------------

    mvbench_cases = build_mvbench_cases()
    benchmark.extend(mvbench_cases)

    # --------------------------------------------------------
    # RefCOCOg
    # --------------------------------------------------------

    grounding_cases = build_grounding_cases()
    benchmark.extend(grounding_cases)

    # --------------------------------------------------------
    # NaviTrace
    # --------------------------------------------------------

    navigation_cases = build_navigation_cases()
    benchmark.extend(navigation_cases)

    # --------------------------------------------------------
    # Stable global IDs
    # --------------------------------------------------------

    for index, case in enumerate(
        benchmark,
        start=1,
    ):
        case["benchmark_index"] = index

    # --------------------------------------------------------
    # Validate before writing
    # --------------------------------------------------------

    validate_benchmark(
        benchmark
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {
        "benchmark_name": (
            "VLM Virtual Office Evaluation"
        ),
        "benchmark_version": "V1",
        "seed": SEED,
        "total_tests": len(benchmark),
        "datasets": {
            "MVBench Temporal Conflict": 40,
            "SoM RefCOCOg": 10,
            "NaviTrace": 10,
        },
        "notes": [
            (
                "CLEVR Sample 500 was excluded because "
                "the downloaded dataset contains images "
                "without question/answer ground truth."
            ),
            (
                "MVBench videos are retained. Their "
                "source_dataset metadata may identify "
                "CLEVRER; this does not depend on the "
                "excluded CLEVR image dataset."
            ),
            (
                "RefCOCOg grounding and visibility cases "
                "use source reference IDs as ground truth."
            ),
            (
                "NaviTrace ground truth is geometric/"
                "segmentation-style annotation and is "
                "therefore marked for manual navigation "
                "scoring rather than textual exact match."
            ),
        ],
        "tests": benchmark,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("BENCHMARK CREATED")
    print("=" * 60)

    print(
        "File:",
        OUTPUT_FILE,
    )

    print(
        "Total tests:",
        len(benchmark),
    )

    print("\nBy dataset:")

    dataset_counts = {}

    for case in benchmark:
        dataset = case["dataset"]

        dataset_counts[dataset] = (
            dataset_counts.get(
                dataset,
                0,
            )
            + 1
        )

    for dataset in sorted(
        dataset_counts
    ):
        print(
            f"  {dataset}: "
            f"{dataset_counts[dataset]}"
        )

    print("\nBy capability:")

    capability_counts = {}

    for case in benchmark:
        capability = case["capability"]

        capability_counts[capability] = (
            capability_counts.get(
                capability,
                0,
            )
            + 1
        )

    for capability in sorted(
        capability_counts
    ):
        print(
            f"  {capability}: "
            f"{capability_counts[capability]}"
        )

    print("\nV1 is ready for model evaluation.")


if __name__ == "__main__":
    main()
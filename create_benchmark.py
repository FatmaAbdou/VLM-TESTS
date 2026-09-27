from huggingface_hub import snapshot_download
from pathlib import Path
import json
import random
from collections import Counter


# ============================================================
# Configuration
# ============================================================

SEED = 42
QUESTIONS_PER_CATEGORY = 20

CATEGORIES = [
    "object_attribute",
    "counting",
    "spatial",
    "existence",
    "comparison",
    "compositional",
]


# ============================================================
# Load the already-downloaded dataset
# ============================================================

root = Path(
    snapshot_download(
        "maujim/CLEVR_sample_500",
        repo_type="dataset"
    )
)

questions_file = root / "questions" / "CLEVR_train_questions.json"

with questions_file.open() as f:
    questions = json.load(f)["questions"]

print(f"Total available questions: {len(questions)}")


# ============================================================
# Helpers
# ============================================================

ATTRIBUTE_OPS = {
    "query_color",
    "query_size",
    "query_material",
    "query_shape",
}

COMPARISON_OPS = {
    "greater_than",
    "less_than",
    "equal_integer",
    "equal_color",
    "equal_shape",
    "equal_size",
    "equal_material",
}

SPATIAL_OPS = {
    "relate",
}


def get_operations(question):
    """Return the reasoning operations used by a question."""
    return [
        step["function"]
        for step in question["program"]
    ]


def final_operation(question):
    """Return the final reasoning operation."""
    return get_operations(question)[-1]


def program_length(question):
    """Number of reasoning steps."""
    return len(question["program"])


# ============================================================
# Category rules
#
# Each question belongs to ONE category.
# The rules are deliberately ordered and mutually exclusive.
# ============================================================

def get_category(question):

    ops = get_operations(question)
    final_op = ops[-1]
    length = len(ops)

    # --------------------------------------------------------
    # 1. Counting
    #
    # Questions whose final answer is a number produced by
    # the count operation.
    # --------------------------------------------------------

    if final_op == "count":
        return "counting"


    # --------------------------------------------------------
    # 2. Existence
    #
    # Questions asking whether something exists.
    # --------------------------------------------------------

    if final_op == "exist":
        return "existence"


    # --------------------------------------------------------
    # 3. Comparison
    #
    # Greater-than, less-than, or equality questions.
    # --------------------------------------------------------

    if final_op in COMPARISON_OPS:
        return "comparison"


    # --------------------------------------------------------
    # 4. Spatial reasoning
    #
    # Questions that explicitly use a spatial relation.
    #
    # We exclude very long programs so that the dedicated
    # compositional category can contain genuinely multi-step
    # questions.
    # --------------------------------------------------------

    if "relate" in ops and length < 8:
        return "spatial"


    # --------------------------------------------------------
    # 5. Object / Attribute recognition
    #
    # Questions whose final operation asks for:
    # color, size, material, or shape.
    #
    # We keep these relatively simple and exclude spatial
    # questions.
    # --------------------------------------------------------

    if (
        final_op in ATTRIBUTE_OPS
        and "relate" not in ops
        and length < 8
    ):
        return "object_attribute"


    # --------------------------------------------------------
    # 6. Compositional reasoning
    #
    # Longer reasoning chains that require multiple operations.
    #
    # These are deliberately selected last so that simpler
    # capabilities above are not swallowed by this category.
    # --------------------------------------------------------

    if length >= 8:
        return "compositional"


    return None


# ============================================================
# Build candidate groups
# ============================================================

groups = {
    category: []
    for category in CATEGORIES
}

for question in questions:

    category = get_category(question)

    if category is not None:
        groups[category].append(question)


# ============================================================
# Show candidate counts
# ============================================================

print("\nAvailable candidates:")

for category in CATEGORIES:
    print(
        f"  {category:20s}: "
        f"{len(groups[category])}"
    )


# ============================================================
# Select the benchmark
#
# We use a fixed random seed so that the benchmark is
# reproducible. The same script will always select the
# same questions.
# ============================================================

random.seed(SEED)

selected = []

# Keep track of questions already selected.
selected_question_ids = set()

# Keep the benchmark reasonably distributed across images.
image_counts = Counter()


for category in CATEGORIES:

    candidates = groups[category].copy()

    random.shuffle(candidates)

    chosen = []

    for question in candidates:

        question_id = question["question_index"]
        image_id = question["image_index"]

        # Don't select the same question twice.
        if question_id in selected_question_ids:
            continue

        # Maximum two benchmark questions from one image.
        if image_counts[image_id] >= 2:
            continue

        chosen.append(question)

        selected_question_ids.add(question_id)
        image_counts[image_id] += 1

        if len(chosen) == QUESTIONS_PER_CATEGORY:
            break


    # Make sure we actually got 20.
    if len(chosen) < QUESTIONS_PER_CATEGORY:

        raise RuntimeError(
            f"Could only find {len(chosen)} questions "
            f"for category '{category}'. "
            f"Need {QUESTIONS_PER_CATEGORY}."
        )


    # Add benchmark metadata.
    for question in chosen:

        question_copy = dict(question)

        question_copy["benchmark_category"] = category

        selected.append(question_copy)


# ============================================================
# Sort benchmark
# ============================================================

category_order = {
    category: index
    for index, category in enumerate(CATEGORIES)
}

selected.sort(
    key=lambda question: (
        category_order[
            question["benchmark_category"]
        ],
        question["question_index"]
    )
)


# ============================================================
# Create output directory
# ============================================================

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)


# ============================================================
# Save benchmark JSON
# ============================================================

output_file = (
    output_dir /
    "clevr_120_benchmark.json"
)

with output_file.open("w") as f:

    json.dump(
        selected,
        f,
        indent=2
    )


# ============================================================
# Save a human-readable summary
# ============================================================

summary_file = (
    output_dir /
    "clevr_120_summary.txt"
)

with summary_file.open("w") as f:

    f.write("CLEVR VLM Evaluation Benchmark\n")
    f.write("=" * 40 + "\n\n")

    f.write(
        f"Total questions: {len(selected)}\n"
    )

    f.write(
        f"Random seed: {SEED}\n\n"
    )

    for category in CATEGORIES:

        category_questions = [
            q for q in selected
            if q["benchmark_category"] == category
        ]

        f.write(
            f"\n[{category}]\n"
        )

        f.write("-" * 40 + "\n")

        for q in category_questions:

            f.write(
                f"ID: {q['question_index']}\n"
            )

            f.write(
                f"Image: {q['image_index']}\n"
            )

            f.write(
                f"Question: {q['question']}\n"
            )

            f.write(
                f"Answer: {q['answer']}\n"
            )

            f.write(
                f"Program: "
                f"{' -> '.join(get_operations(q))}\n\n"
            )


# ============================================================
# Final summary
# ============================================================

print("\n========================================")
print("BENCHMARK CREATED")
print("========================================")

print(
    f"\nTotal benchmark questions: "
    f"{len(selected)}"
)

print("\nQuestions per category:")

for category in CATEGORIES:

    count = sum(
        1
        for q in selected
        if q["benchmark_category"] == category
    )

    print(
        f"  {category:20s}: {count}"
    )

print("\nFiles created:")

print(
    f"  {output_file.resolve()}"
)

print(
    f"  {summary_file.resolve()}"
)

print("\nBenchmark is ready for inspection.")
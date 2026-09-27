from huggingface_hub import snapshot_download
from pathlib import Path
import json
from collections import Counter

# Find the already-downloaded dataset
root = Path(
    snapshot_download(
        "maujim/CLEVR_sample_500",
        repo_type="dataset"
    )
)

# Load training questions
questions_file = root / "questions" / "CLEVR_train_questions.json"

with questions_file.open() as f:
    data = json.load(f)

questions = data["questions"]

print(f"\nTotal training questions: {len(questions)}")

# Count answer types
answers = Counter(q["answer"] for q in questions)

print("\nMost common answers:")
for answer, count in answers.most_common(20):
    print(f"  {answer}: {count}")

# Count operations used in reasoning programs
operations = Counter()

for q in questions:
    for step in q["program"]:
        operations[step["function"]] += 1

print("\nReasoning operations:")
for operation, count in operations.most_common():
    print(f"  {operation}: {count}")

# Show a few examples
print("\nExample questions:\n")

for q in questions[:5]:
    print("Question:", q["question"])
    print("Answer:", q["answer"])
    print("Program:", " -> ".join(step["function"] for step in q["program"]))
    print()

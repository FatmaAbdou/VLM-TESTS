from huggingface_hub import snapshot_download
import json
from pathlib import Path

print("Downloading CLEVR Sample 500...")

root = Path(
    snapshot_download(
        repo_id="maujim/CLEVR_sample_500",
        repo_type="dataset"
    )
)

print(f"Dataset downloaded to: {root}")

questions_file = root / "questions" / "CLEVR_train_questions.json"

with questions_file.open() as f:
    questions = json.load(f)["questions"]

print(f"Training questions available: {len(questions)}")
print("First question:")
print(questions[0]["question"])
print("Answer:")
print(questions[0]["answer"])

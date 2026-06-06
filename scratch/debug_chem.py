import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from src.generation.reconstructor import reconstruct_exam, ReconstructorConfig

def main():
    exam_path = Path("output/exams/exam_chemistry_g10_20260601_195703_5d1bd3bd.json")
    if not exam_path.exists():
        print("Exam file not found!")
        return

    with open(exam_path, "r", encoding="utf-8") as f:
        exam_data = json.load(f)

    # Reconstruct exam with default config (or similar to what dataset preparation does)
    config = ReconstructorConfig(
        typo_rate=0.02,
        space_noise_rate=0.15,
        latex_mask_prob=0.5,
        latex_placeholder="[LATEX]",
        casing_noise_prob=0.10,
        synonym_swap_prob=0.10,
        formatting_noise_prob=0.10
    )
    config.seed = "5d1bd3bd"
    rec = reconstruct_exam(exam_data, config)
    raw_text = rec["raw_text"]
    spans = rec["spans"]

    print("\nSearching dataset splits for '$' labeled as 'option_label':")
    splits = ["train.jsonl", "val.jsonl", "test.jsonl"]
    for split in splits:
        p = Path("output/dataset") / split
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                sample = json.loads(line)
                metadata = sample.get("metadata", {})
                if "5d1bd3bd" in str(metadata):
                    tokens = sample.get("tokens", [])
                    tags = sample.get("tags", [])
                    for t, tag in zip(tokens, tags):
                        if "$" in t and "option_label" in tag:
                            print(f"Found in {split} line {idx}: token {repr(t)} has tag {tag} | Metadata: {metadata}")

if __name__ == "__main__":
    main()

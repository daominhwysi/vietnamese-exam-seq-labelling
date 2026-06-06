import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from src.generation.reconstructor import reconstruct_exam, ReconstructorConfig

def main():
    exam_path = Path("output/exams/exam_history_g10_20260602_025223_c37b23d6.json")
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
    config.seed = "c37b23d6"
    rec = reconstruct_exam(exam_data, config)
    raw_text = rec["raw_text"]
    spans = rec["spans"]

    print(f"Total spans: {len(spans)}")
    print("\nSpans containing 'Question' or 'Câu' or 'Ông':")
    for i, s in enumerate(spans):
        start = s["start"]
        end = s["end"]
        label = s["label"]
        text = raw_text[start:end]
        
        # If the text has '5' or 'Ông' or label is question_label
        if "5" in text or "Ông" in text or label == "question_label":
            print(f"Span {i} | ({start}, {end}) | {label:<15} | Expected: {repr(s.get('text'))} | Actual: {repr(text)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Count real XLM-RoBERTa tokens across all exam files using the actual reconstructor pipeline.
"""
import sys
import json
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.reconstructor import reconstruct_question, ReconstructorConfig

LATEX_REGEX = re.compile(r'\$[^$]+\$')

def replace_latex(text):
    return LATEX_REGEX.sub("[LATEX]", text) if text else text

def replace_latex_in_q(q_data):
    import copy
    q = copy.deepcopy(q_data)
    if q.get("is_group"):
        q["context"] = replace_latex(q.get("context", ""))
        for sub in q.get("questions", []):
            sub["stem"] = replace_latex(sub.get("stem", ""))
            sub["options"] = [replace_latex(o) for o in sub.get("options", [])]
    else:
        q["stem"] = replace_latex(q.get("stem", ""))
        q["options"] = [replace_latex(o) for o in q.get("options", [])]
    q.pop("raw_text", None)
    q.pop("spans", None)
    return q

def main():
    from transformers import AutoTokenizer

    exam_dir = Path(__file__).parent.parent / "output" / "exams"
    exam_files = list(exam_dir.glob("exam_*.json"))
    print(f"Found {len(exam_files)} exam files")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[LATEX]", "<blank />", "<blank/>", "[BLANK]"]})

    config = ReconstructorConfig()

    total_questions = 0
    total_tokens = 0
    total_chars = 0
    token_lengths = []
    skipped = 0

    for exam_path in exam_files:
        try:
            with open(exam_path, "r", encoding="utf-8") as f:
                exam = json.load(f)
        except Exception as e:
            print(f"  SKIP {exam_path.name}: {e}")
            skipped += 1
            continue

        sections = exam.get("sections", {})
        for section_title, questions in sections.items():
            for q_data in questions:
                try:
                    q = dict(q_data)
                    if "subject" not in q and "subject" in exam:
                        q["subject"] = exam["subject"]
                    if "grade" not in q and "grade" in exam:
                        q["grade"] = exam["grade"]

                    q = replace_latex_in_q(q)
                    reconstructed = reconstruct_question(q, config)
                    raw_text = reconstructed["raw_text"]

                    tokens = tokenizer(raw_text, truncation=False, add_special_tokens=True)
                    n_tokens = len(tokens["input_ids"])

                    total_questions += 1
                    total_tokens += n_tokens
                    total_chars += len(raw_text)
                    token_lengths.append(n_tokens)

                except Exception as e:
                    skipped += 1

    if not token_lengths:
        print("No questions processed!")
        return

    token_lengths.sort()
    n = len(token_lengths)
    avg = total_tokens / n
    median = token_lengths[n // 2]
    p90 = token_lengths[int(n * 0.90)]
    p99 = token_lengths[int(n * 0.99)]
    max_tok = max(token_lengths)
    min_tok = min(token_lengths)

    train_tokens = int(total_tokens * 0.8)
    val_tokens = int(total_tokens * 0.1)
    test_tokens = total_tokens - train_tokens - val_tokens

    print("\n" + "="*60)
    print("  REAL TOKEN COUNT REPORT (XLM-RoBERTa)")
    print("="*60)
    print(f"  Exam files processed    : {len(exam_files) - skipped}")
    print(f"  Questions processed     : {total_questions:,}")
    print(f"  Questions skipped       : {skipped}")
    print(f"  Total chars (raw text)  : {total_chars:,}")
    print()
    print(f"  Total tokens (all)      : {total_tokens:,}")
    print(f"  Train tokens (80%)      : {train_tokens:,}")
    print(f"  Val tokens   (10%)      : {val_tokens:,}")
    print(f"  Test tokens  (10%)      : {test_tokens:,}")
    print()
    print(f"  Avg tokens / question   : {avg:.1f}")
    print(f"  Median tokens           : {median}")
    print(f"  90th percentile         : {p90}")
    print(f"  99th percentile         : {p99}")
    print(f"  Min / Max tokens        : {min_tok} / {max_tok}")
    print()
    print("  TRAINING TIME ESTIMATES (3 epochs, batch=8, fp16)")
    print("-"*60)
    train_steps = (int(total_questions * 0.8) // 8) * 3
    print(f"  Train samples           : {int(total_questions * 0.8):,}")
    print(f"  Steps (3 epochs, bs=8)  : {train_steps:,}")
    for gpu, steps_per_sec in [("RTX 3060/3070", 6), ("RTX 3090/4090", 15), ("A100", 35), ("T4 (Colab)", 3), ("CPU only", 0.15)]:
        secs = train_steps / steps_per_sec
        mins = secs / 60
        hrs = mins / 60
        if hrs >= 1:
            print(f"  {gpu:<22}: ~{hrs:.1f} hours")
        else:
            print(f"  {gpu:<22}: ~{mins:.0f} minutes")
    print("="*60)

if __name__ == "__main__":
    main()

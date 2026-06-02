import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
from src.generation.generator import Subject
from src.generation.exam_compiler import generate_single_exam

def test_generation():
    print("Generating English Exam (Grade 12)...")
    try:
        exam_english = generate_single_exam(Subject.ENGLISH, 12, concurrency=4)
        if exam_english:
            with open("output/exams/test_english_exam.json", "w", encoding="utf-8") as f:
                json.dump(exam_english, f, ensure_ascii=False, indent=2)
            print("Successfully generated English exam!")
        else:
            print("Failed to generate English exam (returned None).")
    except Exception as e:
        print(f"Error generating English exam: {e}")

    print("\nGenerating Literature Exam (Grade 11)...")
    try:
        exam_lit = generate_single_exam(Subject.LITERATURE, 11, concurrency=4)
        if exam_lit:
            with open("output/exams/test_literature_exam.json", "w", encoding="utf-8") as f:
                json.dump(exam_lit, f, ensure_ascii=False, indent=2)
            print("Successfully generated Literature exam!")
        else:
            print("Failed to generate Literature exam (returned None).")
    except Exception as e:
        print(f"Error generating Literature exam: {e}")

if __name__ == "__main__":
    test_generation()

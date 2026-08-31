import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.exam_compiler import run_batch_exams_generator

def main():
    print("=" * 60)
    print("STARTING BATCH GENERATION: 25 LITERATURE + 25 BIOLOGY EXAMS")
    print("Model: gpt-5.6-luna | Provider: codex | Thinking: low | Concurrency: 2")
    print("=" * 60)

    # 1. Generate 25 Literature exams
    print("\n>>> STAGE 1: Generating 25 Literature (Ngữ văn) Exams...")
    run_batch_exams_generator(
        num_exams=25,
        output_dir="output/exams",
        model="gpt-5.6-luna",
        thinking="low",
        concurrency=2,
        subject="literature",
        provider="codex",
    )

    # 2. Generate 25 Biology exams
    print("\n>>> STAGE 2: Generating 25 Biology (Sinh học) Exams...")
    run_batch_exams_generator(
        num_exams=25,
        output_dir="output/exams",
        model="gpt-5.6-luna",
        thinking="low",
        concurrency=2,
        subject="biology",
        provider="codex",
    )

    print("\n" + "=" * 60)
    print("ALL 50 EXAMS (25 LITERATURE + 25 BIOLOGY) GENERATED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()

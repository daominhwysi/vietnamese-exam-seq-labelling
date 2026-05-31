import argparse
import sys
import concurrent.futures
from sequence_labelling_data_generator.curriculum import generate_curriculum
from sequence_labelling_data_generator.generator import Subject

# Hardcoded default values
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_THINKING = "high"
DEFAULT_CONCURRENCY = 4

def main():
    parser = argparse.ArgumentParser(description="Concurrently Generate Curricula for all Subjects & Grades [6-12]")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: '{DEFAULT_MODEL}')"
    )
    parser.add_argument(
        "--thinking",
        type=str,
        default=DEFAULT_THINKING,
        choices=["high", "max", "low", "medium", "none"],
        help=f"Thinking effort level (default: '{DEFAULT_THINKING}')"
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of parallel workers for concurrent generation (default: {DEFAULT_CONCURRENCY})"
    )
    
    args = parser.parse_args()
    
    # Resolve thinking parameter
    thinking_val = None if args.thinking == "none" else args.thinking
    
    # List of subjects
    subjects = [s.value for s in Subject]
    # Grades 6 to 12
    grades = list(range(10, 13))
    
    print("=" * 60)
    print(f"Orchestrating concurrent curriculum generation for all subjects and grades [6-12]")
    print(f"Model: {args.model}")
    print(f"Thinking Level: {args.thinking}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Grades: {', '.join(map(str, grades))}")
    print("=" * 60)
    
    # Concurrent Curriculum Generation
    print("\n--- Starting Curriculum Generation (Concurrent) ---")
    
    # Prepare list of curriculum tasks
    tasks = []
    for subject in subjects:
        for grade in grades:
            tasks.append((subject, grade))
            
    def generate_single_curriculum_task(subj, grd):
        try:
            print(f"[Curriculum Start] Subject={subj}, Grade={grd}")
            generate_curriculum(
                subject=subj,
                grade=grd,
                model=args.model,
                thinking=thinking_val
            )
            print(f"[Curriculum Success] Subject={subj}, Grade={grd}")
            return True
        except Exception as e:
            print(f"[Curriculum Error] Subject={subj}, Grade={grd}: {e}")
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(generate_single_curriculum_task, s, g): (s, g) for s, g in tasks}
        concurrent.futures.wait(futures.keys())
        
    print("\nCurriculum generation completed successfully.")

if __name__ == "__main__":
    main()

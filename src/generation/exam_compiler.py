import os
import re
import json
import uuid
import random
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from src.generation.generator import (
    Subject,
    QuestionType,
    Difficulty,
    generate_single_question
)

# Section Headers as defined by real Vietnamese exam layouts
SECTION_MC = "PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn"
SECTION_TF = "PHẦN II. Trắc nghiệm đúng-sai"
SECTION_SA = "PHẦN III. Trắc nghiệm trả lời ngắn"
SECTION_ESSAY = "PHẦN IV. Tự luận"

def get_available_curricula() -> List[Tuple[str, int]]:
    """
    Scans the output/curriculum directory to find pre-generated curriculum files.
    Returns a list of tuples: [(subject, grade), ...]
    """
    curriculum_dir = Path("output") / "curriculum"
    if not curriculum_dir.exists():
        return []
    
    pairs = []
    for file in curriculum_dir.glob("*.json"):
        match = re.match(r"^([a-zA-Z0-9_]+)_(\d+)\.json$", file.name)
        if match:
            pairs.append((match.group(1), int(match.group(2))))
    return pairs

def generate_exam_tasks() -> List[Tuple[str, QuestionType, Optional[Difficulty]]]:
    """
    Builds the question tasks (Section, QuestionType, Difficulty) for a single exam
    based on the user's randomized section count requirements.
    """
    tasks = []
    
    # Section 1: 10 - 20 questions
    num_mc = random.randint(10, 20)
    for _ in range(num_mc):
        # Mostly standard multiple choice, with small probabilities of ordering and grouped questions
        qtype = random.choices(
            [QuestionType.MULTIPLE_CHOICE, QuestionType.ORDERING, QuestionType.GROUP_MULTIPLE_CHOICE],
            weights=[0.90, 0.05, 0.05]
        )[0]
        tasks.append((SECTION_MC, qtype, None))
        
    # Section 2: 1 - 6 questions
    num_tf = random.randint(1, 6)
    for _ in range(num_tf):
        tasks.append((SECTION_TF, QuestionType.TRUE_FALSE, None))
        
    # Section 3: 50% 6 questions, 50% random(0 - 6) questions
    if random.random() < 0.5:
        num_sa = 6
    else:
        num_sa = random.randint(0, 6)
    for _ in range(num_sa):
        # Standard short answer, with a small probability of grouped short answer questions
        qtype = random.choices(
            [QuestionType.SHORT_ANSWER, QuestionType.GROUP_SHORT_ANSWER],
            weights=[0.90, 0.10]
        )[0]
        tasks.append((SECTION_SA, qtype, None))
        
    # Section 4: 0 - 4 questions
    num_essay = random.randint(0, 4)
    for _ in range(num_essay):
        # Essay questions are short_answers of higher cognitive level
        diff = random.choice([Difficulty.APPLICATION, Difficulty.HIGH_APPLICATION])
        tasks.append((SECTION_ESSAY, QuestionType.SHORT_ANSWER, diff))
        
    return tasks

def generate_single_exam(
    subject: Subject,
    grade: int,
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
    concurrency: int = 8
) -> Optional[Dict[str, Any]]:
    """
    Generates a single exam JSON by calling curriculum-based generation in parallel.
    """
    tasks = generate_exam_tasks()
    print(f"Compiling exam for Subject={subject.value}, Grade={grade} with {len(tasks)} questions...")
    
    results = [None] * len(tasks)
    pending_indices = list(range(len(tasks)))
    
    max_attempts = 3
    attempt = 0
    
    # Retry loop for failed generation calls (e.g. LLM errors/timeouts)
    while pending_indices and attempt < max_attempts:
        attempt += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {}
            for idx in pending_indices:
                section_name, qtype, diff = tasks[idx]
                future_to_idx[executor.submit(
                    generate_single_question,
                    subject=subject,
                    grade=grade,
                    model=model,
                    thinking=thinking,
                    question_type=qtype,
                    difficulty=diff
                )] = idx
                
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    if res is not None:
                        results[idx] = res
                except Exception as e:
                    print(f"Error generating question {idx} on attempt {attempt}: {e}")
                    
        pending_indices = [i for i, r in enumerate(results) if r is None]
        if pending_indices and attempt < max_attempts:
            print(f"Attempt {attempt} completed. {len(pending_indices)} questions failed. Retrying in next attempt...")
            
    # Group results by section name
    sections = {
        SECTION_MC: [],
        SECTION_TF: [],
        SECTION_SA: [],
        SECTION_ESSAY: []
    }
    
    success_count = 0
    for idx, q_data in enumerate(results):
        if q_data is not None:
            # Clean up redundant/duplicate keys
            clean_q = dict(q_data)
            clean_q.pop("raw_text", None)
            clean_q.pop("spans", None)
            clean_q.pop("problem_type_level", None)
            
            section_name = tasks[idx][0]
            sections[section_name].append(clean_q)
            success_count += 1
            
    if success_count == 0:
        return None
        
    exam_id = uuid.uuid4().hex[:8]
    return {
        "exam_id": exam_id,
        "subject": subject.value,
        "grade": grade,
        "created_at": datetime.now().isoformat(),
        "sections": sections
    }

def run_batch_exams_generator(
    num_exams: int,
    output_dir: str = "output/exams",
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
    concurrency: int = 8
):
    """
    Generates num_exams mock exams across randomly selected subjects and grades.
    Saves each exam to output_dir/exam_{subject}_g{grade}_{timestamp}_{uuid}.json.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Resolve available curricula
    available_pairs = get_available_curricula()
    if not available_pairs:
        # Fallback if no curricula generated yet
        print("Warning: No curricula files found in output/curriculum. Using standard fallbacks.")
        available_pairs = [(s.value, g) for s in Subject for g in [10, 11, 12]]
        
    print("=" * 60)
    print(f"Starting batch generation of {num_exams} mock exam JSON files...")
    print(f"Output Directory: {out_path.absolute()}")
    print(f"Concurrency per exam: {concurrency}")
    print("=" * 60)
    
    success_exams = 0
    
    for i in range(num_exams):
        print(f"\n--- Generating Exam {i+1} / {num_exams} ---")
        
        # Pick random subject and grade
        subj_str, grade = random.choice(available_pairs)
        subject = Subject(subj_str)
        
        try:
            exam_data = generate_single_exam(
                subject=subject,
                grade=grade,
                model=model,
                thinking=thinking,
                concurrency=concurrency
            )
            
            if exam_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                exam_uuid = exam_data["exam_id"]
                file_name = f"exam_{subj_str}_g{grade}_{timestamp}_{exam_uuid}.json"
                file_path = out_path / file_name
                
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(exam_data, f, ensure_ascii=False, indent=2)
                    
                print(f"[Success] Exam {i+1} saved to: {file_path.name}")
                success_exams += 1
            else:
                print(f"[Error] Failed to generate any questions for Exam {i+1}")
        except Exception as e:
            print(f"[Exception] Failed to compile Exam {i+1}: {e}")
            
    print("\n" + "=" * 60)
    print(f"Batch generation completed: {success_exams}/{num_exams} successfully compiled and saved.")
    print("=" * 60)

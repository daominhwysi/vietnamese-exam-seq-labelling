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

# New Section Headers for Literature
SECTION_LIT_READING = "I. ĐỌC HIỂU"
SECTION_LIT_WRITING = "II. VIẾT"

# English Section Instructions
ENG_INST_EXCHANGE = "Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that best completes each of the following exchanges."
ENG_INST_STRESS = "Mark the letter A, B, C, or D on your answer sheet to indicate the word that differs from the other three in the position of stress in each of the following questions."
ENG_INST_PRONUNCIATION = "Mark the letter A, B, C, or D on your answer sheet to indicate the word whose underlined part differs from the other three in pronunciation in each of the following questions."
ENG_INST_CLOSEST = "Mark the letter A, B, C, or D on your answer sheet to indicate the word CLOSEST in meaning to the underlined word in each of the following questions."
ENG_INST_OPPOSITE = "Mark the letter A, B, C, or D on your answer sheet to indicate the word(s) OPPOSITE in meaning to the underlined word(s) in each of the following questions."
ENG_INST_GRAMMAR_VOCAB = "Mark the letter A, B, C, or D on your answer sheet to indicate the correct answer to each of the following questions."
ENG_INST_CLOZE_OLD = "Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct word or phrase that best fits each of the numbered blanks."
ENG_INST_READING = "Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct answer to each of the questions."
ENG_INST_COMBINATION = "Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that best combines each pair of sentences in the following questions."
ENG_INST_ERROR = "Mark the letter A, B, C, or D on your answer sheet to indicate the underlined part that needs correction in each of the following questions."
ENG_INST_TRANSFORMATION = "Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that is closest in meaning to each of the following questions."

ENG_INST_CLOZE_NEW = "Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct option that best fits each of the numbered blanks."
ENG_INST_REORDERING = "Mark the letter A, B, C, or D on your answer sheet to indicate the correct option that best reorders the sentences/paragraphs to build a coherent dialogue/letter/text."

def generate_exam_tasks(subject: Subject = Subject.CHEMISTRY) -> List[Tuple[str, QuestionType, Optional[Difficulty], dict]]:
    """
    Builds the question tasks (Section, QuestionType, Difficulty, extra_filters) for a single exam
    based on the subject's structure and layout requirements.
    """
    tasks = []
    
    if subject == Subject.ENGLISH:
        # Decide whether to generate the new (2018 GDPT) or old format
        # 60% new format (CEFR B2-C1), 40% old format (CEFR A2-B1)
        if random.random() < 0.6:
            # --- NEW FORMAT (2018 GDPT, 40 questions, 10 tasks) ---
            # 1. Cloze Sentence (1 group question with 5 sub-questions)
            tasks.append((ENG_INST_CLOZE_NEW, QuestionType.GROUP_MULTIPLE_CHOICE, None, {"problem_type_filter": "cloze_sentence"}))
            
            # 2. Reading 1 (1 group question with 8 sub-questions)
            tasks.append((ENG_INST_READING, QuestionType.GROUP_MULTIPLE_CHOICE, None, {"problem_type_filter": "reading_comprehension_8"}))
            
            # 3. Cloze Word News (1 group question with 6 sub-questions, Q14-19)
            tasks.append((ENG_INST_CLOZE_NEW, QuestionType.GROUP_MULTIPLE_CHOICE, None, {"problem_type_filter": "cloze_word_news"}))
            
            # 4. Cloze Word Leaflet (1 group question with 6 sub-questions, Q20-25)
            tasks.append((ENG_INST_CLOZE_NEW, QuestionType.GROUP_MULTIPLE_CHOICE, None, {"problem_type_filter": "cloze_word_leaflet"}))
            
            # 5. Reading 2 (1 group question with 10 sub-questions, Q26-35)
            tasks.append((ENG_INST_READING, QuestionType.GROUP_MULTIPLE_CHOICE, None, {"problem_type_filter": "reading_comprehension_10"}))
            
            # 6. Sentence/Dialogue Reordering (5 standalone MC questions, Q36-40)
            tasks.append((ENG_INST_REORDERING, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "reordering_dialogue"}))
            tasks.append((ENG_INST_REORDERING, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "reordering_dialogue"}))
            tasks.append((ENG_INST_REORDERING, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "reordering_letter"}))
            tasks.append((ENG_INST_REORDERING, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "reordering_text"}))
            tasks.append((ENG_INST_REORDERING, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "reordering_text"}))
        else:
            # --- OLD FORMAT (2006 GDPT, 50 questions, 36 tasks) ---
            # 1. Spoken Exchange (2 standalone MC questions)
            tasks.append((ENG_INST_EXCHANGE, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "exchange"}))
            tasks.append((ENG_INST_EXCHANGE, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "exchange"}))
            
            # 2. Stress position (2 standalone MC questions)
            tasks.append((ENG_INST_STRESS, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "stress"}))
            tasks.append((ENG_INST_STRESS, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "stress"}))
            
            # 3. Pronunciation (2 standalone MC questions)
            tasks.append((ENG_INST_PRONUNCIATION, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "pronunciation"}))
            tasks.append((ENG_INST_PRONUNCIATION, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "pronunciation"}))
            
            # 4. Closest meaning (2 standalone MC questions)
            tasks.append((ENG_INST_CLOSEST, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "closest_meaning"}))
            tasks.append((ENG_INST_CLOSEST, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "closest_meaning"}))
            
            # 5. Opposite meaning (2 standalone MC questions)
            tasks.append((ENG_INST_OPPOSITE, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "opposite_meaning"}))
            tasks.append((ENG_INST_OPPOSITE, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "opposite_meaning"}))
            
            # 6. Grammar & Vocabulary (15 standalone MC questions)
            for _ in range(15):
                tasks.append((ENG_INST_GRAMMAR_VOCAB, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "grammar_vocabulary"}))
                
            # 7. Cloze Word (1 group question with 5 sub-questions)
            tasks.append((ENG_INST_CLOZE_OLD, QuestionType.GROUP_MULTIPLE_CHOICE, None, {"problem_type_filter": "cloze_word_old"}))
            
            # 8. Reading 1 (1 group question with 5 sub-questions)
            tasks.append((ENG_INST_READING, QuestionType.GROUP_MULTIPLE_CHOICE, None, {"problem_type_filter": "reading_comprehension_5"}))
            
            # 9. Reading 2 (1 group question with 7 sub-questions)
            tasks.append((ENG_INST_READING, QuestionType.GROUP_MULTIPLE_CHOICE, None, {"problem_type_filter": "reading_comprehension_7"}))
            
            # 10. Sentence Combination (2 standalone MC questions)
            tasks.append((ENG_INST_COMBINATION, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "sentence_combination"}))
            tasks.append((ENG_INST_COMBINATION, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "sentence_combination"}))
            
            # 11. Error Correction (3 standalone MC questions)
            tasks.append((ENG_INST_ERROR, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "error_correction"}))
            tasks.append((ENG_INST_ERROR, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "error_correction"}))
            tasks.append((ENG_INST_ERROR, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "error_correction"}))
            
            # 12. Sentence Transformation (3 standalone MC questions)
            tasks.append((ENG_INST_TRANSFORMATION, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "sentence_transformation"}))
            tasks.append((ENG_INST_TRANSFORMATION, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "sentence_transformation"}))
            tasks.append((ENG_INST_TRANSFORMATION, QuestionType.MULTIPLE_CHOICE, None, {"problem_type_filter": "sentence_transformation"}))
            
    elif subject == Subject.LITERATURE:
        # Section I: Đọc hiểu (1 group question with 4-5 free-response questions)
        tasks.append((SECTION_LIT_READING, QuestionType.GROUP_SHORT_ANSWER, None, {"problem_type_filter": "reading_comprehension_literature"}))
        
        # Section II: Viết (2 standalone essay prompts)
        tasks.append((SECTION_LIT_WRITING, QuestionType.SHORT_ANSWER, Difficulty.APPLICATION, {"problem_type_filter": "social_argumentation_essay"}))
        tasks.append((SECTION_LIT_WRITING, QuestionType.SHORT_ANSWER, Difficulty.HIGH_APPLICATION, {"problem_type_filter": "literary_analysis_essay"}))
        
    else:
        # Section 1: 10 - 20 questions
        num_mc = random.randint(10, 20)
        for _ in range(num_mc):
            # Mostly standard multiple choice, with small probabilities of ordering and grouped questions
            qtype = random.choices(
                [QuestionType.MULTIPLE_CHOICE, QuestionType.ORDERING, QuestionType.GROUP_MULTIPLE_CHOICE],
                weights=[0.90, 0.05, 0.05]
            )[0]
            tasks.append((SECTION_MC, qtype, None, {}))
            
        # Section 2: 1 - 6 questions
        num_tf = random.randint(1, 6)
        for _ in range(num_tf):
            tasks.append((SECTION_TF, QuestionType.TRUE_FALSE, None, {}))
            
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
            tasks.append((SECTION_SA, qtype, None, {}))
            
        # Section 4: 0 - 4 questions
        num_essay = random.randint(0, 4)
        for _ in range(num_essay):
            # Essay questions are short_answers of higher cognitive level
            diff = random.choice([Difficulty.APPLICATION, Difficulty.HIGH_APPLICATION])
            tasks.append((SECTION_ESSAY, QuestionType.SHORT_ANSWER, diff, {}))
            
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
    tasks = generate_exam_tasks(subject)
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
                section_name, qtype, diff, extra_filters = tasks[idx]
                future_to_idx[executor.submit(
                    generate_single_question,
                    subject=subject,
                    grade=grade,
                    model=model,
                    thinking=thinking,
                    question_type=qtype,
                    difficulty=diff,
                    **extra_filters
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
    if subject == Subject.LITERATURE:
        sections = {
            SECTION_LIT_READING: [],
            SECTION_LIT_WRITING: []
        }
    elif subject == Subject.ENGLISH:
        sections = {}
    else:
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
            if section_name not in sections:
                sections[section_name] = []
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
    concurrency: int = 8,
    subject: Optional[str] = None,
    grade: Optional[int] = None
):
    """
    Generates num_exams mock exams across randomly selected subjects and grades.
    Saves each exam to output_dir/exam_{subject}_g{grade}_{timestamp}_{uuid}.json.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Resolve available curricula
    available_pairs = get_available_curricula()
    # Always ensure english and literature grades 10-12 are available (they bypass curriculum loading)
    for g in [10, 11, 12]:
        available_pairs.append(("english", g))
        available_pairs.append(("literature", g))
        
    print("=" * 60)
    print(f"Starting batch generation of {num_exams} mock exam JSON files...")
    print(f"Output Directory: {out_path.absolute()}")
    print(f"Concurrency per exam: {concurrency}")
    if subject:
        print(f"Filter Subject: {subject}")
    if grade:
        print(f"Filter Grade: {grade}")
    print("=" * 60)
    
    success_exams = 0
    
    for i in range(num_exams):
        print(f"\n--- Generating Exam {i+1} / {num_exams} ---")
        
        # Pick random subject and grade respecting filters
        if subject and grade:
            subj_str, gr = subject, grade
        elif subject:
            matching = [p for p in available_pairs if p[0] == subject]
            if not matching:
                print(f"Error: No matching curriculum/pair found for subject '{subject}'. Using random.")
                subj_str, gr = random.choice(available_pairs)
            else:
                subj_str, gr = random.choice(matching)
        elif grade:
            matching = [p for p in available_pairs if p[1] == grade]
            if not matching:
                print(f"Error: No matching curriculum/pair found for grade {grade}. Using random.")
                subj_str, gr = random.choice(available_pairs)
            else:
                subj_str, gr = random.choice(matching)
        else:
            subj_str, gr = random.choice(available_pairs)
            
        subject_enum = Subject(subj_str)
        
        try:
            exam_data = generate_single_exam(
                subject=subject_enum,
                grade=gr,
                model=model,
                thinking=thinking,
                concurrency=concurrency
            )
            
            if exam_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                exam_uuid = exam_data["exam_id"]
                file_name = f"exam_{subj_str}_g{gr}_{timestamp}_{exam_uuid}.json"
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

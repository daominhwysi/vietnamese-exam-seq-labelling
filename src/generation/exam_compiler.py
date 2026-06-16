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
    generate_single_question,
)

# Các định nghĩa phần thi tiêu chuẩn
SECTION_MC = "PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn"
SECTION_TF = "PHẦN II. Trắc nghiệm đúng-sai"
SECTION_SA = "PHẦN III. Trắc nghiệm trả lời ngắn"
SECTION_ESSAY = "PHẦN IV. Tự luận"

# Các định nghĩa phần thi Ngữ văn
SECTION_LIT_READING = "I. ĐỌC HIỂU"
SECTION_LIT_WRITING = "II. VIẾT"

# Chỉ dẫn chi tiết cho từng phần thi Tiếng Anh (Đóng vai trò là Section Headers)
ENG_INST_EXCHANGE = "Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that best completes each of the following exchanges."
ENG_INST_STRESS = "Mark the letter A, B, C, or D on your answer sheet to indicate the word that differs from the other three in the position of stress in each of the following questions."
ENG_INST_PRONUNCIATION = "Mark the letter A, B, C, or D on your answer sheet to indicate the word whose underlined part differs from the other three in pronunciation in each of the following questions."
ENG_INST_CLOSEST = "Mark the letter A, B, C, or D on your answer sheet to indicate the word CLOSEST in meaning to the underlined word in each of the following questions."
ENG_INST_OPPOSITE = "Mark the letter A, B, C, or D on your answer sheet to indicate the word(s) OPPOSITE in meaning to the underlined word(s) in each of the following questions."
ENG_INST_GRAMMAR_VOCAB = "Mark the letter A, B, C, or D on your answer sheet to indicate the correct answer to each of the following questions."
ENG_INST_CLOZE_OLD = "Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct word or phrase that best fits each of the numbered blanks from 26 to 30."
ENG_INST_READING = "Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct answer to each of the questions."
ENG_INST_COMBINATION = "Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that best combines each pair of sentences in the following questions."
ENG_INST_ERROR = "Mark the letter A, B, C, or D on your answer sheet to indicate the underlined part that needs correction in each of the following questions."
ENG_INST_TRANSFORMATION = "Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that is closest in meaning to each of the following questions."

ENG_INST_CLOZE_NEW = "Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct option that best fits each of the numbered blanks."
ENG_INST_REORDERING = "Mark the letter A, B, C, or D on your answer sheet to indicate the correct option that best reorders the sentences/paragraphs to build a coherent dialogue/letter/text."


def get_available_curricula() -> List[Tuple[str, int]]:
    curriculum_dir = Path("output") / "curriculum"
    if not curriculum_dir.exists():
        return []

    pairs = []
    for file in curriculum_dir.glob("*.json"):
        match = re.match(r"^([a-zA-Z0-9_]+)_(\d+)\.json$", file.name)
        if match:
            pairs.append((match.group(1), int(match.group(2))))
    return pairs


def generate_exam_tasks(
    subject: Subject = Subject.CHEMISTRY,
) -> List[Tuple[str, QuestionType, Optional[Difficulty], dict]]:
    """
    Xây dựng danh sách tác vụ tạo câu hỏi (Phần thi, Loại câu hỏi, Độ khó, Bộ lọc bổ sung) cho một đề thi.
    """
    tasks = []

    if subject == Subject.ENGLISH:
        # 60% lựa chọn cấu trúc mới (GDPT 2018) , 40% cấu trúc cũ
        if random.random() < 0.6:
            # --- CẤU TRÚC MỚI (40 câu hỏi) ---
            tasks.append(
                (
                    ENG_INST_CLOZE_NEW,
                    QuestionType.GROUP_MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "cloze_sentence"},
                )
            )
            tasks.append(
                (
                    ENG_INST_READING,
                    QuestionType.GROUP_MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reading_comprehension_8"},
                )
            )
            tasks.append(
                (
                    ENG_INST_CLOZE_NEW,
                    QuestionType.GROUP_MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "cloze_word_news"},
                )
            )
            tasks.append(
                (
                    ENG_INST_CLOZE_NEW,
                    QuestionType.GROUP_MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "cloze_word_leaflet"},
                )
            )
            tasks.append(
                (
                    ENG_INST_READING,
                    QuestionType.GROUP_MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reading_comprehension_10"},
                )
            )

            # 4 dialogue ordering questions
            tasks.append(
                (
                    ENG_INST_REORDERING,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reordering_dialogue"},
                )
            )
            tasks.append(
                (
                    ENG_INST_REORDERING,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reordering_dialogue"},
                )
            )
            tasks.append(
                (
                    ENG_INST_REORDERING,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reordering_dialogue"},
                )
            )
            tasks.append(
                (
                    ENG_INST_REORDERING,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reordering_dialogue"},
                )
            )
            # 2 letter ordering questions
            tasks.append(
                (
                    ENG_INST_REORDERING,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reordering_letter"},
                )
            )
            tasks.append(
                (
                    ENG_INST_REORDERING,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reordering_letter"},
                )
            )
            # 4 text/paragraph ordering questions
            tasks.append(
                (
                    ENG_INST_REORDERING,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reordering_text"},
                )
            )
            tasks.append(
                (
                    ENG_INST_REORDERING,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reordering_text"},
                )
            )
            tasks.append(
                (
                    ENG_INST_REORDERING,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reordering_text"},
                )
            )
            tasks.append(
                (
                    ENG_INST_REORDERING,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reordering_text"},
                )
            )
        else:
            # --- CẤU TRÚC CŨ (50 câu hỏi) ---
            tasks.append(
                (
                    ENG_INST_EXCHANGE,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "exchange"},
                )
            )
            tasks.append(
                (
                    ENG_INST_EXCHANGE,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "exchange"},
                )
            )

            tasks.append(
                (
                    ENG_INST_STRESS,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "stress"},
                )
            )
            tasks.append(
                (
                    ENG_INST_STRESS,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "stress"},
                )
            )

            tasks.append(
                (
                    ENG_INST_PRONUNCIATION,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "pronunciation"},
                )
            )
            tasks.append(
                (
                    ENG_INST_PRONUNCIATION,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "pronunciation"},
                )
            )

            tasks.append(
                (
                    ENG_INST_CLOSEST,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "closest_meaning"},
                )
            )
            tasks.append(
                (
                    ENG_INST_CLOSEST,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "closest_meaning"},
                )
            )

            tasks.append(
                (
                    ENG_INST_OPPOSITE,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "opposite_meaning"},
                )
            )
            tasks.append(
                (
                    ENG_INST_OPPOSITE,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "opposite_meaning"},
                )
            )

            for _ in range(15):
                tasks.append(
                    (
                        ENG_INST_GRAMMAR_VOCAB,
                        QuestionType.MULTIPLE_CHOICE,
                        None,
                        {"problem_type_filter": "grammar_vocabulary"},
                    )
                )

            tasks.append(
                (
                    ENG_INST_CLOZE_OLD,
                    QuestionType.GROUP_MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "cloze_word_old"},
                )
            )
            tasks.append(
                (
                    ENG_INST_READING,
                    QuestionType.GROUP_MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reading_comprehension_5"},
                )
            )
            tasks.append(
                (
                    ENG_INST_READING,
                    QuestionType.GROUP_MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "reading_comprehension_7"},
                )
            )

            tasks.append(
                (
                    ENG_INST_COMBINATION,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "sentence_combination"},
                )
            )
            tasks.append(
                (
                    ENG_INST_COMBINATION,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "sentence_combination"},
                )
            )

            tasks.append(
                (
                    ENG_INST_ERROR,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "error_correction"},
                )
            )
            tasks.append(
                (
                    ENG_INST_ERROR,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "error_correction"},
                )
            )
            tasks.append(
                (
                    ENG_INST_ERROR,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "error_correction"},
                )
            )

            tasks.append(
                (
                    ENG_INST_TRANSFORMATION,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "sentence_transformation"},
                )
            )
            tasks.append(
                (
                    ENG_INST_TRANSFORMATION,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "sentence_transformation"},
                )
            )
            tasks.append(
                (
                    ENG_INST_TRANSFORMATION,
                    QuestionType.MULTIPLE_CHOICE,
                    None,
                    {"problem_type_filter": "sentence_transformation"},
                )
            )

    elif subject == Subject.LITERATURE:
        tasks.append(
            (
                SECTION_LIT_READING,
                QuestionType.GROUP_SHORT_ANSWER,
                None,
                {"problem_type_filter": "reading_comprehension_literature"},
            )
        )
        tasks.append(
            (
                SECTION_LIT_WRITING,
                QuestionType.SHORT_ANSWER,
                Difficulty.APPLICATION,
                {"problem_type_filter": "social_argumentation_essay"},
            )
        )
        tasks.append(
            (
                SECTION_LIT_WRITING,
                QuestionType.SHORT_ANSWER,
                Difficulty.HIGH_APPLICATION,
                {"problem_type_filter": "literary_analysis_essay"},
            )
        )

    else:
        num_mc = random.randint(10, 20)
        for _ in range(num_mc):
            qtype = random.choices(
                [
                    QuestionType.MULTIPLE_CHOICE,
                    QuestionType.ORDERING,
                    QuestionType.GROUP_MULTIPLE_CHOICE,
                ],
                weights=[0.90, 0.05, 0.05],
            )[0]
            tasks.append((SECTION_MC, qtype, None, {}))

        num_tf = random.randint(1, 6)
        for _ in range(num_tf):
            tasks.append((SECTION_TF, QuestionType.TRUE_FALSE, None, {}))

        num_sa = 6 if random.random() < 0.5 else random.randint(0, 6)
        for _ in range(num_sa):
            qtype = random.choices(
                [QuestionType.SHORT_ANSWER, QuestionType.GROUP_SHORT_ANSWER],
                weights=[0.90, 0.10],
            )[0]
            tasks.append((SECTION_SA, qtype, None, {}))

        num_essay = random.randint(0, 4)
        for _ in range(num_essay):
            diff = random.choice([Difficulty.APPLICATION, Difficulty.HIGH_APPLICATION])
            tasks.append((SECTION_ESSAY, QuestionType.SHORT_ANSWER, diff, {}))

    return tasks


def generate_single_exam(
    subject: Subject,
    grade: int,
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
    concurrency: int = 8,
    provider: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Biên dịch và điều phối tạo đề thi hoàn chỉnh.
    """
    tasks = generate_exam_tasks(subject)
    print(
        f"Compiling exam for Subject={subject.value}, Grade={grade} with {len(tasks)} questions..."
    )

    results = [None] * len(tasks)
    pending_indices = list(range(len(tasks)))

    max_attempts = 3
    attempt = 0

    while pending_indices and attempt < max_attempts:
        attempt += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {}
            for idx in pending_indices:
                section_name, qtype, diff, extra_filters = tasks[idx]
                future_to_idx[
                    executor.submit(
                        generate_single_question,
                        subject=subject,
                        grade=grade,
                        model=model,
                        thinking=thinking,
                        question_type=qtype,
                        difficulty=diff,
                        provider=provider,
                        **extra_filters,
                    )
                ] = idx

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
            print(
                f"Attempt {attempt} completed. {len(pending_indices)} questions failed. Retrying..."
            )

    # Phân nhóm động bảo toàn thứ tự xuất hiện của phần thi
    sections = {}
    if subject == Subject.LITERATURE:
        sections = {SECTION_LIT_READING: [], SECTION_LIT_WRITING: []}
    elif subject not in [Subject.ENGLISH, Subject.LITERATURE]:
        sections = {SECTION_MC: [], SECTION_TF: [], SECTION_SA: [], SECTION_ESSAY: []}

    success_count = 0
    for idx, q_data in enumerate(results):
        if q_data is not None:
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
        "sections": sections,
    }


def run_batch_exams_generator(
    num_exams: int,
    output_dir: str = "output/exams",
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
    concurrency: int = 8,
    subject: Optional[str] = None,
    grade: Optional[int] = None,
    provider: Optional[str] = None,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    available_pairs = get_available_curricula()
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
        print(f"\n--- Generating Exam {i + 1} / {num_exams} ---")

        if subject and grade:
            subj_str, gr = subject, grade
        elif subject:
            matching = [p for p in available_pairs if p[0] == subject]
            subj_str, gr = (
                random.choice(matching) if matching else random.choice(available_pairs)
            )
        elif grade:
            matching = [p for p in available_pairs if p[1] == grade]
            subj_str, gr = (
                random.choice(matching) if matching else random.choice(available_pairs)
            )
        else:
            subj_str, gr = random.choice(available_pairs)

        subject_enum = Subject(subj_str)

        try:
            exam_data = generate_single_exam(
                subject=subject_enum,
                grade=gr,
                model=model,
                thinking=thinking,
                concurrency=concurrency,
                provider=provider,
            )

            if exam_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                exam_uuid = exam_data["exam_id"]
                file_name = f"exam_{subj_str}_g{gr}_{timestamp}_{exam_uuid}.json"
                file_path = out_path / file_name

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(exam_data, f, ensure_ascii=False, indent=2)

                print(f"[Success] Exam {i + 1} saved to: {file_path.name}")
                success_exams += 1
            else:
                print(f"[Error] Failed to generate questions for Exam {i + 1}")
        except Exception as e:
            print(f"[Exception] Failed to compile Exam {i + 1}: {e}")

    print("\n" + "=" * 60)
    print(
        f"Batch generation completed: {success_exams}/{num_exams} successfully compiled."
    )
    print("=" * 60)

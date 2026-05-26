import os
import re
import json
import random
import uuid
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
from enum import Enum

from deepseek_client import chat
from sequence_labelling_data_generator.parser import parse_question_xml
from sequence_labelling_data_generator.reconstructor import reconstruct_question

class Subject(str, Enum):
    ECONOMICS_LAW = "economics_law"
    GEOGRAPHY = "geography"
    HISTORY = "history"
    MATH_ALGEBRA = "math_algebra"
    MATH_GEOMETRY = "math_geometry"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"

class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ORDERING = "ordering"
    GROUP_MULTIPLE_CHOICE = "group_multiple_choice"
    GROUP_SHORT_ANSWER = "group_short_answer"

class Difficulty(str, Enum):
    RECOGNIZE = "recognize"
    COMPREHEND = "comprehend"
    LOW_APPLICATION = "low_application"
    APPLICATION = "application"
    HIGH_APPLICATION = "high_application"

SUBJECT_DISPLAY = {
    Subject.ECONOMICS_LAW: 'Kinh tế pháp luật',
    Subject.GEOGRAPHY: 'Địa lý',
    Subject.HISTORY: 'Lịch sử',
    Subject.MATH_ALGEBRA: 'Toán Đại số',
    Subject.MATH_GEOMETRY: 'Toán hình học',
    Subject.PHYSICS: 'Vật lý',
    Subject.CHEMISTRY: 'Hóa học'
}

QUESTION_TYPE_DISPLAY = {
    QuestionType.MULTIPLE_CHOICE: 'trắc nghiệm nhiều phương án',
    QuestionType.TRUE_FALSE: 'đúng sai',
    QuestionType.SHORT_ANSWER: 'trả lời ngắn',
    QuestionType.ORDERING: 'sắp xếp thứ tự',
    QuestionType.GROUP_MULTIPLE_CHOICE: 'trắc nghiệm nhiều phương án',
    QuestionType.GROUP_SHORT_ANSWER: 'trả lời ngắn'
}

DIFFICULTY_DISPLAY = {
    Difficulty.RECOGNIZE: "Nhận biết",
    Difficulty.COMPREHEND: "Thông hiểu",
    Difficulty.LOW_APPLICATION: "Vận dụng thấp",
    Difficulty.APPLICATION: "Vận dụng thường",
    Difficulty.HIGH_APPLICATION: "Vận dụng cao"
}

GRADES = [8, 9, 10, 11, 12]


def make_standard_prompt(subject: Subject, grade: int, q_type: QuestionType, difficulty: Difficulty) -> tuple[str, str]:
    if q_type == QuestionType.MULTIPLE_CHOICE:
        length_desc = "trung bình phần dẫn (stem) là 30 từ"
        format_example = """<question>
<stem>Câu hỏi của bạn...</stem>
<option>Phương án A (chỉ ghi nội dung phương án, không ghi 'A.' hay 'A. ' ở đầu)</option>
<option>Phương án B...</option>
<option>Phương án C...</option>
<option>Phương án D...</option>
</question>"""
    elif q_type == QuestionType.TRUE_FALSE:
        length_desc = "trung bình phần dẫn (stem) là 100 từ"
        format_example = """<question>
<stem>Câu hỏi/Ngữ cảnh của bạn...</stem>
<option>Khẳng định a (chỉ ghi nội dung, không ghi 'a)' hay 'a) ' ở đầu)</option>
<option>Khẳng định b...</option>
<option>Khẳng định c...</option>
<option>Khẳng định d...</option>
</question>"""
    elif q_type == QuestionType.ORDERING:
        length_desc = "trung bình phần dẫn (stem) là 50 từ, yêu cầu sắp xếp thứ tự các sự kiện/bước"
        format_example = """<question>
<stem>Câu hỏi yêu cầu sắp xếp thứ tự các sự kiện hoặc các bước (ví dụ: Sắp xếp các sự kiện lịch sử sau theo thứ tự thời gian từ trước đến sau:)</stem>
<option>Sự kiện/Bước thứ nhất (chỉ ghi nội dung, không ghi số thứ tự ở đầu)</option>
<option>Sự kiện/Bước thứ hai...</option>
<option>Sự kiện/Bước thứ ba...</option>
<option>Sự kiện/Bước thứ tư...</option>
</question>"""
    else:  # QuestionType.SHORT_ANSWER
        length_desc = "trung bình phần dẫn (stem) là 150 từ"
        format_example = """<question>
<stem>Câu hỏi của bạn...</stem>
</question>"""

    system = "Bạn là một AI chuyên môn cao về biên soạn câu hỏi đề thi và tài liệu học tập tại Việt Nam."
    
    prompt = f"""Hãy tạo ngẫu nhiên một câu hỏi cho:
- Môn học: {SUBJECT_DISPLAY[subject]}
- Lớp: {grade}
- Dạng câu hỏi: {QUESTION_TYPE_DISPLAY[q_type]}
- Mức độ nhận thức: {DIFFICULTY_DISPLAY[difficulty]}
- Độ dài phần dẫn (stem): {length_desc}

Yêu cầu định dạng:
Chỉ xuất ra cấu trúc XML sau đây, không kèm theo bất kỳ lời thoại, văn bản giải thích hay bọc định dạng nào ngoài XML.
{format_example}

Quy tắc quan trọng:
1. Công thức toán học, ký hiệu vật lý/hóa học bắt buộc phải sử dụng LaTeX và được bọc trong cặp dấu $...$ (ví dụ: $f(x) = x^2$ hoặc $\\vec{{u}} = (a;b;c)$).
2. Hãy sáng tạo ra câu hỏi độc lạ, ngẫu nhiên nhất có thể, tránh trùng lặp với các dạng câu hỏi cơ bản.
3. KHÔNG cần cung cấp lời giải hay đáp án.
4. Câu hỏi không cần đúng thực tế hay hoàn hảo tuyệt đối (chỉ cần sinh ra định dạng đề thi chuẩn, các biểu thức toán học hoặc thông tin có thể giả định tự do vì đây là dữ liệu giả lập phục vụ cho huấn luyện mô hình sequence labeling, không dùng để kiểm tra học sinh).
5. Hãy trả lời NGAY LẬP TỨC dưới dạng XML, bỏ qua các bước phân tích sâu hay lập luận chi tiết. Nghĩ càng ít càng tốt, đi thẳng vào cấu trúc XML.
6. KHÔNG được ghi các ký tự đề mục (như "A.", "B.", "C.", "D." hoặc "a)", "b)", "c)", "d)") vào đầu nội dung thẻ <option>. Thẻ <option> chỉ chứa trực tiếp phần chữ/công thức của phương án.
"""
    return system, prompt

def make_group_prompt(subject: Subject, grade: int, q_type: QuestionType, difficulty: Difficulty) -> tuple[str, str]:
    if q_type == QuestionType.GROUP_MULTIPLE_CHOICE:
        num_sub = "3 đến 4 câu trắc nghiệm"
        format_example = """<group_question>
  <context>Đoạn thông tin/Ngữ cảnh dùng chung cho các câu hỏi...</context>
  <question>
    <stem>Câu hỏi 1...</stem>
    <option>Phương án A (chỉ ghi nội dung phương án, không ghi 'A.' hay 'A. ' ở đầu)</option>
    <option>Phương án B...</option>
    <option>Phương án C...</option>
    <option>Phương án D...</option>
  </question>
  ... (tiếp tục cho các câu hỏi 2, 3, 4)
</group_question>"""
    else:  # QuestionType.GROUP_SHORT_ANSWER
        num_sub = "2 đến 3 câu trả lời ngắn"
        format_example = """<group_question>
  <context>Đoạn thông tin/Ngữ cảnh dùng chung cho các câu hỏi...</context>
  <question>
    <stem>Câu hỏi 1...</stem>
  </question>
  <question>
    <stem>Câu hỏi 2...</stem>
  </question>
  ... (tiếp tục cho các câu hỏi 3)
</group_question>"""

    system = "Bạn là một AI chuyên môn cao về biên soạn câu hỏi đề thi và tài liệu học tập tại Việt Nam."
    
    prompt = f"""Hãy tạo ngẫu nhiên một nhóm câu hỏi đặc biệt (từ một thông tin dùng chung phát sinh ra nhiều câu hỏi) cho:
- Môn học: {SUBJECT_DISPLAY[subject]}
- Lớp: {grade}
- Mức độ nhận thức: {DIFFICULTY_DISPLAY[difficulty]}
- Định dạng nhóm câu hỏi: Từ một thông tin ngữ cảnh chung, hãy tạo ra {num_sub}.

Yêu cầu định dạng:
Chỉ xuất ra cấu trúc XML sau đây, không kèm theo bất kỳ lời thoại, văn bản giải thích hay bọc định dạng nào ngoài XML.
{format_example}

Quy tắc quan trọng:
1. Công thức toán học, ký hiệu vật lý/hóa học bắt buộc phải sử dụng LaTeX và được bọc trong cặp dấu $...$ (ví dụ: $f(x) = x^2$ hoặc $\\vec{{u}} = (a;b;c)$).
2. Hãy sáng tạo ra câu hỏi độc lạ, ngẫu nhiên nhất có thể, tránh trùng lặp.
3. KHÔNG cần cung cấp lời giải hay đáp án.
4. Câu hỏi không cần đúng thực tế hay hoàn hảo tuyệt đối (chỉ cần sinh ra định dạng đề thi chuẩn, các biểu thức toán học hoặc thông tin có thể giả định tự do vì đây là dữ liệu giả lập phục vụ cho huấn luyện mô hình sequence labeling, không dùng để kiểm tra học sinh).
5. Hãy trả lời NGAY LẬP TỨC dưới dạng XML, bỏ qua các bước phân tích sâu hay lập luận chi tiết. Nghĩ càng ít càng tốt, đi thẳng vào cấu trúc XML.
6. KHÔNG được ghi các ký tự đề mục (như "A.", "B.", "C.", "D." hoặc "a)", "b)", "c)", "d)") vào đầu nội dung thẻ <option>. Thẻ <option> chỉ chứa trực tiếp phần chữ/công thức của phương án.
"""
    return system, prompt



def generate_single_question() -> Optional[Dict[str, Any]]:
    """Generates a single question/group based on randomized criteria, tries to parse it, and returns the dict representation."""
    # 1. Randomize parameters
    subject = random.choice(list(Subject))
    grade = random.choice(GRADES)
    
    # Randomize difficulty with weighted probability:
    # Nhận biết 30%, Thông hiểu 30%, Vận dụng thấp 10%, Vận dụng thường 20%, Vận dụng cao 10%
    difficulty = random.choices(
        [
            Difficulty.RECOGNIZE,
            Difficulty.COMPREHEND,
            Difficulty.LOW_APPLICATION,
            Difficulty.APPLICATION,
            Difficulty.HIGH_APPLICATION
        ],
        weights=[30, 30, 10, 20, 10],
        k=1
    )[0]
    
    # 5% probability of special group question
    is_group = random.random() < 0.05
    
    if is_group:
        # Group questions support multiple choice or short answer
        actual_type = random.choice([QuestionType.GROUP_MULTIPLE_CHOICE, QuestionType.GROUP_SHORT_ANSWER])
        system_prompt, user_prompt = make_group_prompt(subject, grade, actual_type, difficulty)
    else:
        actual_type = random.choice([
            QuestionType.MULTIPLE_CHOICE,
            QuestionType.TRUE_FALSE,
            QuestionType.SHORT_ANSWER,
            QuestionType.ORDERING
        ])
        system_prompt, user_prompt = make_standard_prompt(subject, grade, actual_type, difficulty)

    # 2. Call deepseek with retry logic
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = chat(prompt=user_prompt, system=system_prompt)
            
            parsed_data = parse_question_xml(response)
            if parsed_data:
                # Add metadata to the parsed result
                parsed_data["subject"] = subject.value
                parsed_data["grade"] = grade
                parsed_data["question_type"] = actual_type.value
                parsed_data["difficulty"] = difficulty.value
                
                # Reconstruct raw text and track character spans
                parsed_data = reconstruct_question(parsed_data)
                
                return parsed_data
            else:
                tqdm.write(f"Warning: Failed to parse XML response on attempt {attempt}.")
        except Exception as e:

            tqdm.write(f"Error calling API on attempt {attempt}: {e}")
            
    return None

def run_generator(num_questions: int, output_dir: str = "output", max_workers: int = 4):
    """Generates specified number of questions in parallel and saves each to its own file in the output directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    def generate_and_save(index: int) -> bool:
        q_data = generate_single_question()
        if q_data:
            subject_slug = q_data["subject"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:6]
            
            file_name = f"question_{subject_slug}_g{q_data['grade']}_{timestamp}_{unique_id}.json"
            file_path = out_path / file_name
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(q_data, f, ensure_ascii=False, indent=2)
                
            return True
        else:
            tqdm.write(f"Failed to generate Question {index+1} after all retries.")
            return False

    print(f"Generating {num_questions} question(s) with concurrency={max_workers}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_and_save, i): i for i in range(num_questions)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_questions, desc="Generating"):
            idx = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                tqdm.write(f"Exception raised during generation of Question {idx+1}: {e}")
            
    print(f"Completed: {success_count}/{num_questions} successfully generated. Saved to '{output_dir}/'")




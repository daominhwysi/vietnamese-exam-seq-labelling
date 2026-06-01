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

from src.generation.deepseek_client import chat
from src.generation.parser import parse_question_xml
from src.generation.reconstructor import reconstruct_question
from src.generation.curriculum import (
    load_curriculum,
    select_curriculum_path,
    map_cognitive_level_to_difficulty,
    SUBJECT_DISPLAY
)

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


def make_standard_prompt(
    subject: Subject, 
    grade: int, 
    q_type: QuestionType, 
    difficulty: Difficulty,
    problem_type_info: Optional[Dict[str, Any]] = None,
    include_figure: bool = False,
    include_table: bool = False
) -> tuple[str, str]:
    if q_type == QuestionType.MULTIPLE_CHOICE:
        length_desc = "trung bình phần dẫn (stem) là 30 từ"
        format_example = """<question>
<stem>Câu hỏi của bạn...</stem>
<option>Phương án A (chỉ ghi nội dung phương án, không ghi 'A.' hay 'A. ' ở đầu)</option>
<option>Phương án B...</option>
<option>Phương án C...</option>
<option>Phương án D...</option>
<answer>Ký tự phương án đúng (A hoặc B hoặc C hoặc D)</answer>
<explanation>Lời giải thích ngắn gọn bằng tiếng Việt (cho phép ghi "không có đáp án đúng" nếu cần)</explanation>
</question>"""
    elif q_type == QuestionType.TRUE_FALSE:
        length_desc = "trung bình phần dẫn (stem) là 100 từ"
        format_example = """<question>
<stem>Câu hỏi/Ngữ cảnh của bạn...</stem>
<option>Khẳng định a (chỉ ghi nội dung, không ghi 'a)' hay 'a) ' ở đầu)</option>
<option>Khẳng định b...</option>
<option>Khẳng định c...</option>
<option>Khẳng định d...</option>
<answer>Kết quả Đúng/Sai tương ứng dưới dạng danh sách ngăn cách bởi dấu phẩy (ví dụ: Đúng, Sai, Đúng, Sai)</answer>
<explanation>Lời giải thích ngắn gọn bằng tiếng Việt (cho phép ghi "không có đáp án đúng" nếu cần)</explanation>
</question>"""
    elif q_type == QuestionType.ORDERING:
        length_desc = "trung bình phần dẫn (stem) là 50 từ, yêu cầu sắp xếp thứ tự các sự kiện/bước"
        format_example = """<question>
<stem>Câu hỏi yêu cầu sắp xếp thứ tự các sự kiện hoặc các bước (ví dụ: Sắp xếp các sự kiện lịch sử sau theo thứ tự thời gian từ trước đến sau:)</stem>
<option>Sự kiện/Bước thứ nhất (chỉ ghi nội dung, không ghi số thứ tự ở đầu)</option>
<option>Sự kiện/Bước thứ hai...</option>
<option>Sự kiện/Bước thứ ba...</option>
<option>Sự kiện/Bước thứ tư...</option>
<answer>Thứ tự đúng của các nhãn bước tương ứng, viết thường và ngăn cách bởi dấu gạch ngang (ví dụ: a - b - c - d hoặc a - c - b - d)</answer>
<explanation>Lời giải thích ngắn gọn bằng tiếng Việt (cho phép ghi "không có đáp án đúng" nếu cần)</explanation>
</question>"""
    else:  # QuestionType.SHORT_ANSWER
        length_desc = "trung bình phần dẫn (stem) là 150 từ"
        format_example = """<question>
<stem>Câu hỏi của bạn...</stem>
<answer>Đáp án ngắn (ví dụ: 10 hoặc 2/3 hoặc tên chất)</answer>
<explanation>Lời giải thích ngắn gọn bằng tiếng Việt (cho phép ghi "không có đáp án đúng" nếu cần)</explanation>
</question>"""

    system = "Bạn là một AI chuyên môn cao về biên soạn câu hỏi đề thi và tài liệu học tập tại Việt Nam."
    
    curriculum_context = ""
    if problem_type_info:
        examples_str = ""
        if problem_type_info.get("examples"):
            examples_str = "\n- Ví dụ câu hỏi tham khảo:\n" + "\n".join([f"  + {ex}" for ex in problem_type_info["examples"]])
            
        curriculum_context = f"""
Thông tin chương trình học cụ thể để biên soạn câu hỏi:
- Chương (Chapter): {problem_type_info.get('chapter')}
- Bài học (Unit): {problem_type_info.get('unit')}
- Dạng bài tập (Problem Type): {problem_type_info.get('name')}
- Chi tiết phương pháp/công thức cần áp dụng: {problem_type_info.get('details')}{examples_str}
"""

    multimodal_instr = ""
    if include_figure:
        multimodal_instr += """
- HÌNH VẼ: Bạn được khuyến khích chèn một hoặc nhiều thẻ hình vẽ có định dạng: <figure description="mô tả cực kỳ chi tiết hình vẽ bao gồm tất cả các thông tin khai thác được từ hình vẽ này" label="Figure 1" /> (hoặc label="Figure X" với X là số thứ tự tương ứng).
  Bạn có thể tự do đặt thẻ <figure ... /> này ở bất kỳ vị trí nào bên trong phần dẫn <stem> hoặc bên trong một hoặc nhiều thẻ <option> (ví dụ như hình vẽ biển báo hay hình học của các phương án lựa chọn). Đảm bảo phần mô tả (description) cực kỳ chi tiết, đầy đủ thông tin để người đọc hình dung được chính xác hình vẽ đó.
"""
    if include_table:
        multimodal_instr += """
- BẢNG SỐ LIỆU: Bạn được khuyến khích chèn một hoặc nhiều bảng số liệu sử dụng các thẻ HTML table chuẩn (như <table>, <tr>, <td>) bên trong phần dẫn <stem>. Bảng phải chứa dữ liệu mô phỏng đề thi thực tế.
"""

    prompt = f"""Hãy tạo ngẫu nhiên một câu hỏi cho:
- Môn học: {SUBJECT_DISPLAY.get(subject.value, subject.value)}
- Lớp: {grade}
- Dạng câu hỏi: {QUESTION_TYPE_DISPLAY[q_type]}
- Mức độ nhận thức: {DIFFICULTY_DISPLAY[difficulty]}
- Độ dài phần dẫn (stem): {length_desc}
{curriculum_context}
{multimodal_instr}
Yêu cầu định dạng:
Chỉ xuất ra cấu trúc XML sau đây, không kèm theo bất kỳ lời thoại, văn bản giải thích hay bọc định dạng nào ngoài XML.
{format_example}

Quy tắc quan trọng:
1. Công thức toán học, ký hiệu vật lý/hóa học bắt buộc phải sử dụng LaTeX và được bọc trong cặp dấu $...$ (ví dụ: $f(x) = x^2$ hoặc $\\vec{{u}} = (a;b;c)$).
2. Hãy sáng tạo ra câu hỏi độc lạ, ngẫu nhiên nhất có thể, tránh trùng lặp với các dạng câu hỏi cơ bản.
3. BẮT BUỘC cung cấp lời giải (thẻ <explanation>) và đáp án đúng (thẻ <answer>) theo định dạng ví dụ trên.
4. Lời giải thích trong thẻ <explanation> phải bằng tiếng Việt, súc tích và ngắn gọn như lời giải thích của giáo viên, cho phép có lỗi biên tập hoặc chính tả nhẹ, và cho phép ghi "không có đáp án đúng" nếu cần thiết.
5. Câu hỏi không cần đúng thực tế hay hoàn hảo tuyệt đối (chỉ cần sinh ra định dạng đề thi chuẩn, các biểu thức toán học hoặc thông tin có thể giả định tự do vì đây là dữ liệu giả lập phục vụ cho huấn luyện mô hình sequence labeling, không dùng để kiểm tra học sinh).
6. Hãy trả lời NGAY LẬP TỨC dưới dạng XML, bỏ qua các bước phân tích sâu hay lập luận chi tiết. Nghĩ càng ít càng tốt, đi thẳng vào cấu trúc XML.
7. KHÔNG được ghi các ký tự đề mục (như "A.", "B.", "C.", "D." hoặc "a)", "b)", "c)", "d)") vào đầu nội dung thẻ <option>. Thẻ <option> chỉ chứa trực tiếp phần chữ/công thức của phương án.
"""
    return system, prompt

def make_group_prompt(
    subject: Subject, 
    grade: int, 
    q_type: QuestionType, 
    difficulty: Difficulty,
    problem_type_info: Optional[Dict[str, Any]] = None,
    include_figure: bool = False,
    include_table: bool = False
) -> tuple[str, str]:
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
    <answer>Ký tự phương án đúng (A hoặc B hoặc C hoặc D)</answer>
    <explanation>Lời giải thích ngắn gọn bằng tiếng Việt</explanation>
  </question>
  ... (tiếp tục cho các câu hỏi 2, 3, 4)
</group_question>"""
    else:  # QuestionType.GROUP_SHORT_ANSWER
        num_sub = "2 đến 3 câu trả lời ngắn"
        format_example = """<group_question>
  <context>Đoạn thông tin/Ngữ cảnh dùng chung cho các câu hỏi...</context>
  <question>
    <stem>Câu hỏi 1...</stem>
    <answer>Đáp án ngắn</answer>
    <explanation>Lời giải thích ngắn gọn bằng tiếng Việt</explanation>
  </question>
  <question>
    <stem>Câu hỏi 2...</stem>
    <answer>Đáp án ngắn</answer>
    <explanation>Lời giải thích ngắn gọn bằng tiếng Việt</explanation>
  </question>
  ... (tiếp tục cho các câu hỏi 3)
</group_question>"""

    system = "Bạn là một AI chuyên môn cao về biên soạn câu hỏi đề thi và tài liệu học tập tại Việt Nam."
    
    curriculum_context = ""
    if problem_type_info:
        examples_str = ""
        if problem_type_info.get("examples"):
            examples_str = "\n- Ví dụ câu hỏi tham khảo:\n" + "\n".join([f"  + {ex}" for ex in problem_type_info["examples"]])
            
        curriculum_context = f"""
Thông tin chương trình học cụ thể để biên soạn câu hỏi:
- Chương (Chapter): {problem_type_info.get('chapter')}
- Bài học (Unit): {problem_type_info.get('unit')}
- Dạng bài tập (Problem Type): {problem_type_info.get('name')}
- Chi tiết phương pháp/công thức cần áp dụng: {problem_type_info.get('details')}{examples_str}
"""

    multimodal_instr = ""
    if include_figure:
        multimodal_instr += """
- HÌNH VẼ: Bạn được khuyến khích chèn một hoặc nhiều thẻ hình vẽ có định dạng: <figure description="mô tả cực kỳ chi tiết hình vẽ bao gồm tất cả các thông tin khai thác được từ hình vẽ này" label="Figure 1" /> (hoặc label="Figure X" với X là số thứ tự tương ứng).
  Bạn có thể đặt thẻ <figure ... /> này ở bất kỳ vị trí nào bên trong <context> dùng chung hoặc bên trong <stem> hay <option> của từng câu hỏi nhỏ. Đảm bảo phần mô tả (description) cực kỳ chi tiết, đầy đủ thông tin.
"""
    if include_table:
        multimodal_instr += """
- BẢNG SỐ LIỆU: Bạn được khuyến khích chèn một hoặc nhiều bảng số liệu sử dụng các thẻ HTML table chuẩn (như <table>, <tr>, <td>) bên trong phần ngữ cảnh <context> dùng chung hoặc phần dẫn <stem> của câu hỏi nhỏ.
"""

    prompt = f"""Hãy tạo ngẫu nhiên một nhóm câu hỏi đặc biệt (từ một thông tin dùng chung phát sinh ra nhiều câu hỏi) cho:
- Môn học: {SUBJECT_DISPLAY.get(subject.value, subject.value)}
- Lớp: {grade}
- Mức độ nhận thức: {DIFFICULTY_DISPLAY[difficulty]}
- Định dạng nhóm câu hỏi: Từ một thông tin ngữ cảnh chung, hãy tạo ra {num_sub}.
{curriculum_context}
{multimodal_instr}
Yêu cầu định dạng:
Chỉ xuất ra cấu trúc XML sau đây, không kèm theo bất kỳ lời thoại, văn bản giải thích hay bọc định dạng nào ngoài XML.
{format_example}

Quy tắc quan trọng:
1. Công thức toán học, ký hiệu vật lý/hóa học bắt buộc phải sử dụng LaTeX và được bọc trong cặp dấu $...$ (ví dụ: $f(x) = x^2$ hoặc $\\vec{{u}} = (a;b;c)$).
2. Hãy sáng tạo ra câu hỏi độc lạ, ngẫu nhiên nhất có thể, tránh trùng lặp.
3. BẮT BUỘC cung cấp lời giải (thẻ <explanation>) và đáp án đúng (thẻ <answer>) cho mỗi câu hỏi thành phần.
4. Lời giải thích trong thẻ <explanation> phải bằng tiếng Việt, súc tích và ngắn gọn như lời giải thích của giáo viên, cho phép có lỗi biên tập hoặc chính tả nhẹ, và cho phép ghi "không có đáp án đúng" nếu cần thiết.
5. Câu hỏi không cần đúng thực tế hay hoàn hảo tuyệt đối (chỉ cần sinh ra định dạng đề thi chuẩn, các biểu thức toán học hoặc thông tin có thể giả định tự do vì đây là dữ liệu giả lập phục vụ cho huấn luyện mô hình sequence labeling, không dùng để kiểm tra học sinh).
6. Hãy trả lời NGAY LẬP TỨC dưới dạng XML, bỏ qua các bước phân tích sâu hay lập luận chi tiết. Nghĩ càng ít càng tốt, đi thẳng vào cấu trúc XML.
7. KHÔNG được ghi các ký tự đề mục (như "A.", "B.", "C.", "D." hoặc "a)", "b)", "c)", "d)") vào đầu nội dung thẻ <option>. Thẻ <option> chỉ chứa trực tiếp phần chữ/công thức của phương án.
"""
    return system, prompt


def generate_single_question(
    subject: Optional[Subject] = None,
    grade: Optional[int] = None,
    chapter_filter: Optional[str] = None,
    unit_filter: Optional[str] = None,
    problem_type_filter: Optional[str] = None,
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
    question_type: Optional[QuestionType] = None,
    difficulty: Optional[Difficulty] = None
) -> Optional[Dict[str, Any]]:
    """Generates a single question/group based on curriculum path or randomized criteria, tries to parse it, and returns the dict representation."""
    
    # 1. Resolve Subject and Grade
    if subject is None or grade is None:
        # If filters are present, try to find an existing curriculum JSON that matches
        matched_curricula = []
        curriculum_dir = Path("output") / "curriculum"
        if curriculum_dir.exists():
            for file in curriculum_dir.glob("*.json"):
                match = re.match(r"^([a-zA-Z0-9_]+)_(\d+)\.json$", file.name)
                if match:
                    matched_curricula.append((match.group(1), int(match.group(2))))
                    
        # Filter matching subject/grade if one of them is provided
        if subject is not None:
            matched_curricula = [m for m in matched_curricula if m[0] == subject.value]
        if grade is not None:
            matched_curricula = [m for m in matched_curricula if m[1] == grade]
            
        if matched_curricula and (chapter_filter or unit_filter or problem_type_filter):
            # Try to find a combination that matches our filters
            random.shuffle(matched_curricula)
            found_path = None
            for subj_str, grd_val in matched_curricula:
                curr = load_curriculum(subj_str, grd_val, autogenerate=False, model=model, thinking=thinking)
                if curr:
                    res = select_curriculum_path(curr, chapter_filter, unit_filter, problem_type_filter)
                    if res:
                        subject = Subject(subj_str)
                        grade = grd_val
                        found_path = res
                        break
            if not found_path:
                # If filters are requested but no matching combination exists in loaded files, return None (cannot generate)
                tqdm.write("Warning: No matching curriculum path found with the specified filters.")
                return None
        else:
            # Fallback to random pick
            if subject is None:
                subject = random.choice(list(Subject))
            if grade is None:
                grade = random.choice(GRADES)

    # 2. Load Curriculum and Choose Problem Type (Dạng)
    problem_type_info = None
    
    # Try to load/generate curriculum
    curriculum = load_curriculum(subject.value, grade, autogenerate=True, model=model, thinking=thinking)
    if curriculum:
        selected_path = select_curriculum_path(curriculum, chapter_filter, unit_filter, problem_type_filter)
        if selected_path:
            chapter_dict, unit_dict, pt_dict = selected_path
            problem_type_info = {
                "chapter": chapter_dict.get("name"),
                "unit": unit_dict.get("name"),
                "id": pt_dict.get("id"),
                "name": pt_dict.get("name"),
                "details": pt_dict.get("details", ""),
                "examples": pt_dict.get("examples", []),
                "cognitive_level": pt_dict.get("cognitive_level")
            }
            if difficulty is None:
                difficulty_str = map_cognitive_level_to_difficulty(problem_type_info["cognitive_level"])
                difficulty = Difficulty(difficulty_str)
        else:
            tqdm.write(f"Warning: Curriculum loaded for {subject.value} Grade {grade}, but no items matched filters.")
            return None
            
    # Fallback to random parameters if no curriculum/path found
    if difficulty is None:
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
    
    # Resolve the question type and prompts
    include_figure = random.random() < 0.13
    include_table = random.random() < 0.03
    
    if question_type is not None:
        actual_type = question_type
        is_group = actual_type in [QuestionType.GROUP_MULTIPLE_CHOICE, QuestionType.GROUP_SHORT_ANSWER]
        if is_group:
            system_prompt, user_prompt = make_group_prompt(
                subject, grade, actual_type, difficulty, problem_type_info,
                include_figure=include_figure, include_table=include_table
            )
        else:
            system_prompt, user_prompt = make_standard_prompt(
                subject, grade, actual_type, difficulty, problem_type_info,
                include_figure=include_figure, include_table=include_table
            )
    else:
        # 5% probability of special group question
        is_group = random.random() < 0.05
        if is_group:
            actual_type = random.choice([QuestionType.GROUP_MULTIPLE_CHOICE, QuestionType.GROUP_SHORT_ANSWER])
            system_prompt, user_prompt = make_group_prompt(
                subject, grade, actual_type, difficulty, problem_type_info,
                include_figure=include_figure, include_table=include_table
            )
        else:
            actual_type = random.choices(
                [
                    QuestionType.MULTIPLE_CHOICE,
                    QuestionType.TRUE_FALSE,
                    QuestionType.SHORT_ANSWER,
                    QuestionType.ORDERING
                ],
                weights=[0.45, 0.25, 0.25, 0.05],
                k=1
            )[0]
            system_prompt, user_prompt = make_standard_prompt(
                subject, grade, actual_type, difficulty, problem_type_info,
                include_figure=include_figure, include_table=include_table
            )

    # 3. Call deepseek with retry logic
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            chat_kwargs = {}
            if model is not None:
                chat_kwargs["model"] = model
            if thinking is not None:
                chat_kwargs["thinking"] = thinking
            response = chat(prompt=user_prompt, system=system_prompt, **chat_kwargs)
            
            parsed_data = parse_question_xml(response)
            if parsed_data:
                # Add metadata to the parsed result
                parsed_data["subject"] = subject.value
                parsed_data["grade"] = grade
                parsed_data["question_type"] = actual_type.value
                parsed_data["difficulty"] = difficulty.value
                
                # Add curriculum-based metadata
                if problem_type_info:
                    parsed_data["chapter"] = problem_type_info["chapter"]
                    parsed_data["unit"] = problem_type_info["unit"]
                    parsed_data["problem_type_id"] = problem_type_info["id"]
                    parsed_data["problem_type_name"] = problem_type_info["name"]
                    parsed_data["problem_type_level"] = problem_type_info["cognitive_level"]
                else:
                    parsed_data["chapter"] = None
                    parsed_data["unit"] = None
                    parsed_data["problem_type_id"] = None
                    parsed_data["problem_type_name"] = None
                    parsed_data["problem_type_level"] = None
                
                # Reconstruct raw text and track character spans
                parsed_data = reconstruct_question(parsed_data)
                
                return parsed_data
            else:
                tqdm.write(f"Warning: Failed to parse XML response on attempt {attempt}.")
        except Exception as e:
            tqdm.write(f"Error calling API on attempt {attempt}: {e}")
            
    return None

def run_generator(
    num_questions: int, 
    output_dir: str = "output", 
    max_workers: int = 4,
    subject: Optional[str] = None,
    grade: Optional[int] = None,
    chapter: Optional[str] = None,
    unit: Optional[str] = None,
    problem_type: Optional[str] = None,
    model: Optional[str] = None,
    thinking: Optional[bool] = None
):
    """Generates specified number of questions in parallel and saves each to its own file in the output directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    # Validate Subject if provided
    subj_enum = None
    if subject:
        try:
            subj_enum = Subject(subject)
        except ValueError:
            print(f"Error: Invalid subject '{subject}'. Must be one of: {[s.value for s in Subject]}")
            return
            
    def generate_and_save(index: int) -> bool:
        q_data = generate_single_question(
            subject=subj_enum,
            grade=grade,
            chapter_filter=chapter,
            unit_filter=unit,
            problem_type_filter=problem_type,
            model=model,
            thinking=thinking
        )
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

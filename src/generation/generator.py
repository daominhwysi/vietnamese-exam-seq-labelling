import json
import os
import re
import random
from enum import Enum
from typing import Dict, Any, List, Optional

from src.generation.deepseek_client import chat
from src.generation.curriculum import (
    load_curriculum,
    select_curriculum_path,
    map_cognitive_level_to_difficulty,
)
from src.generation.parser import parse_question_xml


class Subject(Enum):
    ECONOMICS_LAW = "economics_law"
    GEOGRAPHY = "geography"
    HISTORY = "history"
    MATH_ALGEBRA = "math_algebra"
    MATH_GEOMETRY = "math_geometry"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ENGLISH = "english"
    LITERATURE = "literature"


class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ORDERING = "ordering"
    GROUP_MULTIPLE_CHOICE = "group_multiple_choice"
    GROUP_SHORT_ANSWER = "group_short_answer"


class Difficulty(Enum):
    RECOGNIZE = "recognize"
    COMPREHEND = "comprehend"
    LOW_APPLICATION = "low_application"
    APPLICATION = "application"
    HIGH_APPLICATION = "high_application"


SUBJECT_DISPLAY: Dict[str, str] = {
    "economics_law": "Kinh tế & Pháp luật",
    "geography": "Địa lý",
    "history": "Lịch sử",
    "math_algebra": "Toán Đại số",
    "math_geometry": "Toán Hình học",
    "physics": "Vật lý",
    "chemistry": "Hóa học",
    "biology": "Sinh học",
    "english": "Tiếng Anh",
    "literature": "Ngữ văn",
}

QUESTION_TYPE_DISPLAY: Dict[str, str] = {
    "multiple_choice": "Trắc nghiệm",
    "true_false": "Đúng / Sai",
    "short_answer": "Trả lời ngắn",
    "ordering": "Sắp xếp",
    "group_multiple_choice": "Nhóm trắc nghiệm",
    "group_short_answer": "Nhóm trả lời ngắn",
}

DIFFICULTY_DISPLAY: Dict[str, str] = {
    "recognize": "Nhận biết",
    "comprehend": "Thông hiểu",
    "low_application": "Vận dụng thấp",
    "application": "Vận dụng",
    "high_application": "Vận dụng cao",
}


def generate_single_question(
    subject: Subject,
    grade: int,
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
    question_type: Optional[QuestionType] = None,
    difficulty: Optional[Difficulty] = None,
    chapter_filter: Optional[str] = None,
    unit_filter: Optional[str] = None,
    problem_type_filter: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Tạo một câu hỏi đơn lẻ hoặc nhóm câu hỏi dựa trên cấu trúc chương trình học (Curriculum).
    Bảo đảm hướng dẫn làm bài ở mức đề thi không bao giờ lọt vào nội dung chi tiết của stem.
    """
    curr = None
    if subject not in [Subject.ENGLISH, Subject.LITERATURE]:
        curr = load_curriculum(
            subject.value, grade, autogenerate=True, model=model, thinking=thinking, provider=provider
        )

    chapter_name = "Tổng hợp"
    unit_name = "Tổng hợp"
    pt_id = problem_type_filter or "general"
    pt_name = problem_type_filter or "general"
    pt_level = "NB_TH"
    pt_details = ""
    pt_examples = []

    if curr:
        path = select_curriculum_path(
            curr, chapter_filter, unit_filter, problem_type_filter
        )
        if path:
            chapter, unit, pt = path
            chapter_name = chapter.get("name", "Tổng hợp")
            unit_name = unit.get("name", "Tổng hợp")
            pt_id = pt.get("id", pt_id)
            pt_name = pt.get("name", pt_name)
            pt_level = pt.get("cognitive_level", "NB_TH")
            pt_details = pt.get("details", "")
            pt_examples = pt.get("examples", [])
            if difficulty is None:
                diff_str = map_cognitive_level_to_difficulty(pt_level)
                difficulty = Difficulty(diff_str)

    if difficulty is None:
        difficulty = Difficulty.COMPREHEND

    if question_type is None:
        question_type = QuestionType.MULTIPLE_CHOICE

    is_group = question_type in [
        QuestionType.GROUP_MULTIPLE_CHOICE,
        QuestionType.GROUP_SHORT_ANSWER,
    ]

    # Chỉ thị ngăn chặn lặp lại hướng dẫn chung
    strict_no_instructions_guideline = """
YÊU CẦU QUAN TRỌNG - KHÔNG ĐƯỢC CHÈN HƯỚNG DẪN CHUNG CỦA ĐỀ THI VÀO STEM HOẶC STIMULUS:
Tuyệt đối không viết các câu chỉ dẫn đề thi như:
- "Mark the letter A, B, C, or D on your answer sheet to indicate..."
- "Read the following passage and mark..."
- "Mark the letter A, B, C, or D to indicate the word that differs..."
- "Read the following piece of news..."
Các câu chỉ dẫn này đã được xử lý ở cấp độ phân mục (section level) của đề thi.
Thẻ <stem> của bạn CHỈ ĐƯỢC chứa nội dung câu hỏi/câu văn cụ thể. Thẻ <stimulus> chỉ chứa văn bản đọc hiểu/ngữ cảnh gốc.
"""

    english_guidelines = ""
    if subject == Subject.ENGLISH:
        english_guidelines = f"""
Yêu cầu định dạng riêng cho bài thi tiếng Anh dạng '{pt_id}':
- Nếu là dạng phát âm (pronunciation): Thẻ <stem> nên để trống hoàn toàn hoặc chứa khoảng trắng " ". Các thẻ <option> chứa 4 từ tiếng Anh với phần cần gạch chân được bao bởi thẻ <u>...</u> (Ví dụ: f<u>oo</u>d).
- Nếu là dạng trọng âm (stress): Thẻ <stem> nên để trống hoàn toàn hoặc chứa khoảng trắng " ". Các thẻ <option> chứa 4 từ tiếng Anh để tìm từ có trọng âm khác biệt.
- Nếu là dạng tìm từ đồng nghĩa/trái nghĩa (closest_meaning, opposite_meaning): Thẻ <stem> chứa câu văn hoàn chỉnh có phần cần tìm từ đồng nghĩa/trái nghĩa được bao bởi thẻ <u>...</u>.
- Nếu là dạng sửa lỗi sai (error_correction): Thẻ <stem> chứa một câu văn tiếng Anh có lỗi sai. Các phương án lựa chọn là các phần trong câu đó.
- Nếu là dạng điền từ/điền câu (cloze_word, cloze_sentence, cloze_word_news, cloze_word_leaflet): Ngữ cảnh văn bản (stimulus) phải chứa các chỗ trống được đánh số cụ thể dạng '(26) <blank />', '(27) <blank />'.
- Nếu là dạng ngữ pháp/từ vựng (grammar_vocabulary): Thẻ <stem> chứa câu văn tiếng Anh có một chỗ trống cần điền từ/cụm từ. Chỗ trống PHẢI được biểu diễn bằng thẻ XML <blank /> (TUYỆT ĐỐI KHÔNG dùng dấu gạch dưới ______ hay dấu chấm ...). Ví dụ: "She wishes she <blank /> harder for the final exam."
- Nếu là dạng hội thoại/giao tiếp (exchange): Thẻ <stem> chứa lượt nói của người A. Lượt trả lời của người B có chỗ trống PHẢI dùng thẻ <blank /> (TUYỆT ĐỐI KHÔNG dùng ______). Ví dụ: 'John: "I am thinking of taking a gap year." - Mary: "<blank />"'
- Nếu là dạng sắp xếp hội thoại (reordering_dialogue): Thẻ <stem> chứa lượt nói của cuộc đối thoại, mỗi dòng bắt đầu bằng ký tự thường kèm dấu chấm và tên người nói (ví dụ: "a. Tom: ...\nb. Mary: ..."). Các phương án <option> là các tổ hợp thứ tự các câu thoại (ví dụ: "a-b-c", "b-c-a").
- Nếu là dạng sắp xếp thư (reordering_letter): Thẻ <stem> chứa một bức thư hoàn chỉnh có các phần nội dung chính được chia thành các câu bắt đầu bằng ký tự thường kèm dấu chấm (ví dụ: "a. ...\nb. ..."), đồng thời PHẢI chứa câu chào mở đầu (ví dụ: "Dear Ms Smith,") và chữ ký kết thúc bức thư (ví dụ: "Yours sincerely,\nABC Bank"). Các phương án <option> là các tổ hợp thứ tự các câu thư (ví dụ: "d-a-c-b-e").
- Nếu là dạng sắp xếp đoạn văn (reordering_text): Thẻ <stem> chứa một đoạn văn học thuật/nghị luận ngắn có các câu được chia thành các phần bắt đầu bằng ký tự thường kèm dấu chấm (ví dụ: "a. ...\nb. ..."). Mỗi câu có độ dài hợp lý, mang văn phong học thuật (IELTS/THPT). Các phương án <option> là các tổ hợp thứ tự các câu văn (ví dụ: "e-c-d-b-a").
"""

    literature_guidelines = ""
    if subject == Subject.LITERATURE:
        literature_guidelines = f"""
QUY CHUẨN ĐẶC BIỆT CHO ĐỀ THI MÔN NGỮ VĂN FORMAT 2025 - 2026 (CHƯƠNG TRÌNH GDPT 2018):
1. QUY ĐỊNH BẮT BUỘC VỀ NGỮ LIỆU:
   - TUYỆT ĐỐI KHÔNG SỬ DỤNG các tác phẩm quen thuộc trong SGK cũ (như Truyện Kiều, Vợ nhặt, Vợ chồng A Phủ, Chiếc thuyền ngoài xa, Tây Tiến, Sóng, Người lái đò Sông Đà, Rừng xà nu, Việt Bắc, Đất Nước, Đồng chí...).
   - BẮT BUỘC 100% sử dụng ngữ liệu MỞ HOÀN TOÀN NẰM NGOÀI SÁCH GIÁO KHOA.
   - ĐA DẠNG HÓA THỂ LOẠI (Luân phiên phong phú, KHÔNG CHỈ DÙNG THƠ MÀ PHẢI DÙNG CẢ VĂN XUÔI, TRUYỆN NGẮN, TẢN VĂN, KÝ, NGHỊ LUẬN):
     + Thể loại Truyện ngắn / Văn xuôi tự sự hiện đại / Ký / Tản văn (ví dụ: đoạn trích từ truyện ngắn/tiểu thuyết/tản văn ngoài SGK của Nguyễn Ngọc Tư, Nguyễn Huy Thiệp, Ma Văn Kháng, Bảo Ninh, Trang Thế Hy, Nguyễn Một, Đỗ Bích Thúy, Lê Lựu, Nguyễn Bình Phương, Dạ Ngân, Trần Thùy Mai, Phong Điệp, Chu Văn Sơn, Đặng Hoàng Giang, Hoàng Phủ Ngọc Tường...).
     + Thể loại Thơ hiện đại / Thơ đương đại ngoài SGK (ví dụ: Thanh Thảo, Nguyễn Duy, Hoàng Nhuận Cầm, Trần Đăng Khoa, Nguyễn Quang Thiều, Y Phương, Thu Bồn, Lâm Thị Mỹ Dạ, Vi Thùy Linh, Mai Văn Phấn...).
     + Thể loại Văn bản nghị luận / Triết luận văn hóa - nhân văn (Nguyên Ngọc, Phan Cẩm Thượng, Vương Trí Nhàn, Nguyễn Trần Bạt...).

2. NẾU LÀ DẠNG ĐỌC HIỂU VĂN BẢN (reading_comprehension_literature):
   - Thẻ <stimulus> chứa đoạn trích văn bản ngoài SGK hoàn chỉnh (khoảng 200 - 450 từ đối với văn xuôi/truyện hoặc 12 - 24 câu đối với thơ) kèm thông tin tên tác phẩm và tác giả.
   - Các thẻ <question> bên trong gồm ĐÚNG 5 CÂU HỎI phân hóa theo ma trận chuẩn 2025-2026:
     * Câu 1 (Nhận biết - 0.5đ): Xác định thể loại, ngôi kể, nhân vật, thể thơ, hoặc phương thức biểu đạt chính.
     * Câu 2 (Nhận biết/Thông hiểu - 0.75đ): Tìm các chi tiết, hình ảnh, từ ngữ trong văn bản hoặc gọi tên biện pháp tu từ được sử dụng.
     * Câu 3 (Thông hiểu - 1.0đ): Giải thích ý nghĩa của một hình ảnh, câu văn/câu thơ hoặc phân tích tác dụng của biện pháp nghệ thuật trong ngữ cảnh cụ thể.
     * Câu 4 (Thông hiểu/Vận dụng - 1.0đ): Phân tích tình cảm, thái độ của tác giả hoặc thông điệp/tư tưởng cốt lõi mà văn bản gửi gắm.
     * Câu 5 (Vận dụng - 0.75đ): Rút ra bài học cuộc sống, liên hệ bản thân hoặc bày tỏ quan điểm đồng tình/không đồng tình và lý giải thuyết phục.
   - Mỗi câu hỏi: <stem> chứa câu hỏi rõ ràng, <answer> chứa đáp án gợi ý cốt lõi, <explanation> chứa tiêu chí chấm điểm.

3. NẾU LÀ DẠNG ĐOẠN VĂN NGHỊ LUẬN VĂN HỌC (literary_analysis_paragraph):
   - Thẻ <stem> yêu cầu viết một đoạn văn (khoảng 200 chữ) phân tích một nét đặc sắc về nội dung hoặc nghệ thuật (hình tượng nhân vật, chi tiết tiêu biểu, cảm xúc, không gian thời gian...) của chính văn bản ở phần Đọc hiểu. Ví dụ: "Từ văn bản ở phần Đọc hiểu, anh/chị hãy viết một đoạn văn (khoảng 200 chữ) phân tích..."
   - Thẻ <answer> chứa dàn ý đoạn văn (Mở đoạn, Thân đoạn: luận điểm & dẫn chứng phân tích, Đánh giá, Kết đoạn).
   - Thẻ <explanation> chứa biểu điểm chi tiết (0.25đ hình thức đoạn, 1.25đ nội dung phân tích, 0.25đ sáng tạo, 0.25đ chính tả ngữ pháp).

4. NẾU LÀ DẠNG BÀI VĂN NGHỊ LUẬN XÃ HỘI (social_argumentation_essay):
   - Thẻ <stem> yêu cầu viết một bài văn (khoảng 600 chữ) bàn luận về một tư tưởng đạo lý, lối sống, kỹ năng hoặc vấn đề xã hội của thế hệ trẻ được gợi mở từ ngữ liệu Đọc hiểu. Ví dụ: "Từ vấn đề được gợi ra trong văn bản Đọc hiểu, anh/chị hãy viết một bài văn (khoảng 600 chữ) trình bày suy nghĩ về..."
   - Thẻ <answer> chứa dàn ý bài văn hoàn chỉnh (Mở bài, Thân bài: Giải thích -> Phân tích chứng minh -> Bàn luận mở rộng/Phản biện -> Bài học nhận thức & hành động, Kết bài).
   - Thẻ <explanation> chứa biểu điểm chuẩn 4.0 điểm (Mở bài 0.5đ, Thân bài 2.5đ, Đánh giá mở rộng 0.5đ, Sáng tạo & chính tả 0.5đ).
"""

    system_prompt = (
        "Bạn là chuyên gia xây dựng ngân hàng câu hỏi khảo thí chuyên nghiệp tại Việt Nam. "
        "Bạn luôn cung cấp câu hỏi sạch, chuẩn hóa dưới dạng cấu trúc XML và không bao giờ chèn thêm chỉ dẫn chung của đề thi vào nội dung câu hỏi."
    )

    xml_format = ""
    if is_group:
        xml_format = """
<group_question>
  <stimulus>Văn bản đọc hiểu hoặc đoạn thông tin chung...</stimulus>
  <question>
    <stem>Nội dung câu hỏi phụ thứ nhất...</stem>
    <option>Phương án A</option>
    <option>Phương án B</option>
    <option>Phương án C</option>
    <option>Phương án D</option>
    <answer>Đáp án (ví dụ: A hoặc B... hoặc ý trả lời chính nếu tự luận)</answer>
    <explanation>Giải thích lý do lựa chọn đáp án bằng tiếng Việt</explanation>
  </question>
  ... (tiếp tục với các câu hỏi tiếp theo trong nhóm) ...
</group_question>
"""
    else:
        xml_format = """
<question>
  <stem>Nội dung câu hỏi cụ thể...</stem>
  <option>Phương án A</option>
  <option>Phương án B</option>
  <option>Phương án C</option>
  <option>Phương án D</option>
  <answer>Đáp án đúng (ví dụ: A, B, C, hoặc D đối với trắc nghiệm, hoặc nội dung đáp án/dàn ý nếu tự luận)</answer>
  <explanation>Giải thích chi tiết bằng tiếng Việt</explanation>
</question>
"""

    prompt = f"""Hãy thiết kế một câu hỏi chất lượng cao cho môn học '{subject.value}', lớp {grade}.
- Loại câu hỏi: {question_type.value}
- Độ khó: {difficulty.value}
- Chương: {chapter_name}
- Bài học: {unit_name}
- Dạng lý thuyết/bài tập: {pt_name} (Mã dạng: {pt_id})
{f"- Công thức/Yếu tố lý thuyết áp dụng: {pt_details}" if pt_details else ""}
{f"- Ví dụ mẫu: {pt_examples}" if pt_examples else ""}

{strict_no_instructions_guideline}
{english_guidelines}
{literature_guidelines}

Cấu trúc định dạng XML yêu cầu:
{xml_format}

Yêu cầu kỹ thuật:
1. Chỉ xuất ra cấu trúc XML hợp lệ nằm trong một khối mã duy nhất (như ```xml ... ```). Không viết thêm lời giới thiệu hoặc kết luận.
2. Không chèn các câu chỉ thị làm bài chung ở mức đề thi vào trường dữ liệu <stem> hoặc <stimulus>.
3. Đáp án trong thẻ <answer> của câu trắc nghiệm nhiều phương án lựa chọn (multiple_choice) chỉ chứa duy nhất một ký tự chữ cái viết hoa (A, B, C, hoặc D).
"""

    try:
        raw_response = chat(
            prompt=prompt,
            system=system_prompt,
            model=model or "deepseek-v4-flash",
            thinking=thinking,
            provider=provider,
        )
        parsed = parse_question_xml(raw_response)

        # Thử lại một lần nữa nếu phân tích cú pháp XML thất bại
        if parsed is None:
            raw_response = chat(
                prompt=prompt
                + "\nLƯU Ý: Phản hồi trước của bạn gặp lỗi phân tích cú pháp XML. Hãy đảm bảo thẻ mở và đóng hợp lệ.",
                system=system_prompt,
                model=model or "deepseek-v4-flash",
                thinking=thinking,
                provider=provider,
            )
            parsed = parse_question_xml(raw_response)

        if parsed:
            parsed["subject"] = subject.value
            parsed["grade"] = grade
            parsed["question_type"] = question_type.value
            parsed["difficulty"] = difficulty.value
            parsed["chapter"] = chapter_name
            parsed["unit"] = unit_name
            parsed["problem_type_id"] = pt_id
            parsed["problem_type_name"] = pt_name
            parsed["problem_type_level"] = pt_level
            return parsed
    except Exception as e:
        print(f"Error during generator execution: {e}")

    return None

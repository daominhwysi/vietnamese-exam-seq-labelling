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
    ENGLISH = "english"
    LITERATURE = "literature"

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
    level_line = ""
    if subject == Subject.ENGLISH:
        pt_id = problem_type_info.get("id", "").lower() if problem_type_info else ""
        
        # Resolve English problem types and CEFR levels
        if "pronunciation" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            length_desc = "4 từ ngắn có chứa phần gạch chân để kiểm tra cách phát âm của nguyên âm/phụ âm"
            format_example = """<question>
<stem>Mark the letter A, B, C, or D on your answer sheet to indicate the word whose underlined part differs from the other three in pronunciation in each of the following questions.</stem>
<option>coast</option>
<option>board</option>
<option>boat</option>
<option>road</option>
<answer>B</answer>
<explanation>Phần gạch chân của từ 'board' phát âm là /ɔː/, các từ còn lại phát âm là /əʊ/.</explanation>
</question>"""

        elif "stress" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            length_desc = "4 từ có 2 hoặc 3 âm tiết để kiểm tra vị trí trọng âm"
            format_example = """<question>
<stem>Mark the letter A, B, C, or D on your answer sheet to indicate the word that differs from the other three in the position of stress in each of the following questions.</stem>
<option>different</option>
<option>creative</option>
<option>possible</option>
<option>national</option>
<answer>B</answer>
<explanation>Từ 'creative' có trọng âm rơi vào âm tiết thứ 2, các từ còn lại rơi vào âm tiết thứ 1.</explanation>
</question>"""

        elif "exchange" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            length_desc = "đoạn hội thoại ngắn giữa 2 người có 1 chỗ trống cần điền để giao tiếp xã hội trôi chảy"
            format_example = """<question>
<stem>Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that best completes each of the following exchanges.
Sophia and Jenny are talking about solar energy.
Sophia: “I think we should use solar energy.”
Jenny: “<blank/> . It's clean and renewable.”</stem>
<option>I don't think so</option>
<option>I agree with you</option>
<option>Of course not</option>
<option>You're wrong</option>
<answer>B</answer>
<explanation>Jenny đồng ý với Sophia vì cô đưa ra lí do 'It's clean and renewable'. Chọn B (Tôi đồng ý với bạn).</explanation>
</question>"""

        elif "grammar_vocabulary" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            length_desc = "câu đơn lẻ bằng tiếng Anh có 1 chỗ trống để kiểm tra ngữ pháp hoặc từ vựng"
            format_example = """<question>
<stem>Lucy was <blank/> runner in the competition.</stem>
<option>faster than</option>
<option>fastest</option>
<option>faster</option>
<option>the fastest</option>
<answer>D</answer>
<explanation>Sử dụng so sánh nhất: 'the fastest' phù hợp với ngữ cảnh Lucy là người chạy nhanh nhất trong cuộc thi.</explanation>
</question>"""

        elif "closest_meaning" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            length_desc = "câu tiếng Anh chứa một từ/cụm từ gạch chân (bọc trong dấu gạch dưới, ví dụ: _means_)"
            format_example = """<question>
<stem>Mark the letter A, B, C, or D on your answer sheet to indicate the word CLOSEST in meaning to the underlined word in each of the following questions.
The mass media serves as a powerful _means_ of distributing information to the public.</stem>
<option>cheap</option>
<option>great</option>
<option>bad</option>
<option>give (hoặc a way/method)</option>
<answer>D</answer>
<explanation>Từ 'means' trong ngữ cảnh này có nghĩa là phương tiện/phương thức gần nghĩa nhất với phương án D.</explanation>
</question>"""

        elif "opposite_meaning" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            length_desc = "câu tiếng Anh chứa một từ/cụm từ gạch chân (bọc trong dấu gạch dưới, ví dụ: _differences_)"
            format_example = """<question>
<stem>Mark the letter A, B, C, or D on your answer sheet to indicate the word(s) OPPOSITE in meaning to the underlined word(s) in each of the following questions.
Despite their striking _differences_, they have developed a good relationship.</stem>
<option>small</option>
<option>major</option>
<option>clear</option>
<option>similarities (hoặc agreements)</option>
<answer>D</answer>
<explanation>Từ 'differences' (sự khác biệt) có nghĩa trái ngược với 'similarities' (sự tương đồng).</explanation>
</question>"""

        elif "sentence_combination" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            length_desc = "cặp câu đơn cần được kết hợp lại một cách logic bằng liên từ hoặc đảo ngữ"
            format_example = """<question>
<stem>Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that best combines each pair of sentences in the following questions.
You should go over your test paper. You shouldn't hand it in until then.</stem>
<option>Only after you have gone over your test paper should you hand it in.</option>
<option>Were you to go over your test paper, you would hand it in.</option>
<option>Not until you have handed in your test paper should you go over it.</option>
<option>Hardly had you handed in your test paper when you went over it.</option>
<answer>A</answer>
<explanation>Cấu trúc đảo ngữ với 'Only after': Chỉ sau khi rà soát bài kiểm tra bạn mới nên nộp bài.</explanation>
</question>"""

        elif "error_correction" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            length_desc = "câu tiếng Anh chứa lỗi sai ngữ pháp/từ vựng, với 4 cụm từ ứng với A, B, C, D trích từ chính câu đó"
            format_example = """<question>
<stem>Mark the letter A, B, C, or D on your answer sheet to indicate the underlined part that needs correction in each of the following questions.
All of the students should submit his writing assignments by Friday.</stem>
<option>All of the</option>
<option>should submit</option>
<option>his</option>
<option>writing</option>
<answer>C</answer>
<explanation>Chủ ngữ 'All of the students' là số nhiều, do đó tính từ sở hữu 'his' phải sửa thành 'their'.</explanation>
</question>"""

        elif "sentence_transformation" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            length_desc = "câu gốc tiếng Anh cần tìm câu viết lại có nghĩa gần nhất"
            format_example = """<question>
<stem>Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that is closest in meaning to each of the following questions.
Mai last went abroad two years ago.</stem>
<option>Mai started going abroad two years ago.</option>
<option>Mai has gone abroad for two years.</option>
<option>Mai hasn't gone abroad for two years.</option>
<option>Mai didn't go abroad two years ago.</option>
<answer>C</answer>
<explanation>Viết lại câu từ Quá khứ đơn sang Hiện tại hoàn thành phủ định.</explanation>
</question>"""

        elif "reordering_dialogue" in pt_id:
            level = "CEFR B2-C1 (Advanced)"
            length_desc = "sắp xếp thứ tự 3 câu đối thoại của Tom và Mary (labeled a, b, c) để tạo thành một đoạn đối thoại mạch lạc"
            format_example = """<question>
<stem>Mark the letter A, B, C or D on your answer sheet to indicate the best arrangement of utterances or sentences to make a cohesive and coherent exchange or text.
a. Tom: Then, text me when you’re home.
b. Tom: It’s getting late. Would you like me to give you a lift home?
c. Mary: Thanks, but I’m going to walk to the supermarket and then take a bus home.</stem>
<option>a-b-c</option>
<option>b-c-a</option>
<option>a-c-b</option>
<option>b-a-c</option>
<answer>B</answer>
<explanation>Thứ tự hội thoại tự nhiên: Tom đề nghị đưa về (b), Mary từ chối lịch sự (c), Tom dặn nhắn tin khi về nhà (a). Thứ tự là b-c-a.</explanation>
</question>"""

        elif "reordering_letter" in pt_id:
            level = "CEFR B2-C1 (Advanced)"
            length_desc = "sắp xếp thứ tự 5 câu/phần của một bức thư (labeled a, b, c, d, e) để tạo thành bức thư hoàn chỉnh mạch lạc"
            format_example = """<question>
<stem>Mark the letter A, B, C or D on your answer sheet to indicate the best arrangement of utterances or sentences to make a cohesive and coherent exchange or text.
Dear Ms Smith,
a. This has been pre-approved, but you need to have this letter and your identification card produced at the nearest branch to apply.
b. The offer is exclusive and expires on December 31st.
c. Your application will be processed, and your card will be issued within 48 hours for immediate use.
d. It is our honour to offer you credit facilities of $6000, affordable with the monthly instalment of $99.
e. Should you require further details, please call 012388888, or visit any of our branches.
Yours sincerely,
ABC Bank</stem>
<option>d-a-c-b-e</option>
<option>a-c-d-b-e</option>
<option>b-d-a-c-e</option>
<option>c-a-d-b-e</option>
<answer>A</answer>
<explanation>Bắt đầu bằng chào mời (d), hướng dẫn thủ tục (a), cam kết thời gian (c), hạn ưu đãi (b), liên hệ hỗ trợ (e). Thứ tự là d-a-c-b-e.</explanation>
</question>"""

        elif "reordering_text" in pt_id:
            level = "CEFR B2-C1 (Advanced)"
            length_desc = "sắp xếp thứ tự 5 câu của một đoạn văn (labeled a, b, c, d, e) để tạo thành đoạn văn mạch lạc"
            format_example = """<question>
<stem>Mark the letter A, B, C or D on your answer sheet to indicate the best arrangement of utterances or sentences to make a cohesive and coherent exchange or text.
a. The developments demonstrate a clear modernisation of the city of Paragon...
b. This shift was further evidenced by the industrialisation...
c. Residential areas were noticeably transformed...
d. Simultaneously, a significant expansion of commercial infrastructure took place...
e. Between 2000 and 2015, the outskirts of Paragon city underwent a dramatic reshaping...</stem>
<option>e-c-d-b-a</option>
<option>e-c-a-d-b</option>
<option>e-d-b-a-c</option>
<option>e-b-a-c-d</option>
<answer>A</answer>
<explanation>Mở đầu giới thiệu chung sự thay đổi (e), tiếp theo là nhà ở (c), khu thương mại (d), khu công nghiệp (b), và kết luận đánh giá sự hiện đại hóa (a). Thứ tự là e-c-d-b-a.</explanation>
</question>"""

        else:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            length_desc = "trung bình phần dẫn (stem) là 15-30 từ bằng tiếng Anh"
            format_example = """<question>
<stem>Câu hỏi dẫn bằng tiếng Anh...</stem>
<option>Nội dung phương án...</option>
<option>Nội dung phương án...</option>
<option>Nội dung phương án...</option>
<option>Nội dung phương án...</option>
<answer>Ký tự phương án đúng</answer>
<explanation>Lời giải thích bằng tiếng Việt</explanation>
</question>"""
            
        level_line = f"- Trình độ tiếng Anh (CEFR Level): {level}\n"
    elif subject == Subject.LITERATURE:
        # Literature Subject prompt style - always free-response/essay style (short answer)
        length_desc = "phần dẫn (stem) là đề bài nghị luận khoảng 20-50 từ bằng tiếng Việt"
        format_example = """<question>
<stem>Đề bài nghị luận bằng tiếng Việt (Ví dụ: "Anh/Chị hãy viết đoạn văn nghị luận (khoảng 200 chữ) phân tích tình cảm của Lê dành cho Sơn trong văn bản ở phần Đọc hiểu.")</stem>
<answer>Dàn ý hoặc hướng dẫn chấm chi tiết bằng tiếng Việt, bao gồm các tiêu chí chấm điểm như cấu trúc bài viết, xác định đúng vấn đề, triển khai luận điểm, chính tả/ngữ pháp và sự sáng tạo.</answer>
<explanation>Lời giải thích/phân tích thêm nếu cần bằng tiếng Việt</explanation>
</question>"""
    else:
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
    english_rules = ""
    if subject == Subject.ENGLISH:
        english_rules = "8. Để gạch chân các từ/cụm từ cần kiểm tra trong câu hỏi Tiếng Anh (như từ đồng nghĩa, trái nghĩa, phần cần sửa lỗi sai, hoặc nguyên âm gạch chân để phát âm), BẮT BUỘC sử dụng thẻ HTML <u>...</u> (ví dụ: <u>means</u> hoặc <u>c</u>oast).\n"
    
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
{level_line}
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
{english_rules}"""
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
    english_rules = ""
    level_line = ""
    if subject == Subject.ENGLISH:
        english_rules = "8. Để gạch chân các từ/cụm từ/câu cần kiểm tra trong câu hỏi Tiếng Anh (như từ đồng nghĩa, trái nghĩa, phần cần sửa lỗi sai, hoặc câu được hỏi trong bài đọc), BẮT BUỘC sử dụng thẻ HTML <u>...</u> (ví dụ: <u>underlined sentence</u>).\n"
        pt_id = problem_type_info.get("id", "").lower() if problem_type_info else ""
        
        # Decide format, level, and count
        if "cloze_sentence" in pt_id:
            level = "CEFR B2-C1 (Advanced)"
            num_sub = "đúng 5 câu hỏi trắc nghiệm điền câu/mệnh đề liên kết"
            format_example = """<group_question>
  <context>Đoạn văn đọc điền bằng tiếng Anh trình độ B2-C1 dài khoảng 150-200 từ, có chứa đúng 5 chỗ trống được ký hiệu lần lượt là (1) <blank />, (2) <blank />, (3) <blank />, (4) <blank />, (5) <blank />. Chú ý phải viết đúng định dạng "<blank />" này (ví dụ: "... (1) <blank /> ... (2) <blank /> ...")</context>
  <question>
    <stem>Question 1:</stem>
    <option>Mệnh đề hoặc câu hoàn chỉnh phù hợp (không ghi 'A.' hay 'A. ' ở đầu)</option>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <answer>C</answer>
    <explanation>Giải thích lý do lựa chọn câu/mệnh đề này dựa trên sự hòa hợp ngữ pháp và tính liên kết ý nghĩa của chỗ trống (1).</explanation>
  </question>
  <question>
    <stem>Question 2:</stem>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <answer>A</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
  <question>
    <stem>Question 3:</stem>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <answer>D</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
  <question>
    <stem>Question 4:</stem>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <answer>B</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
  <question>
    <stem>Question 5:</stem>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <option>Mệnh đề hoặc câu...</option>
    <answer>A</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
</group_question>"""

        elif "cloze_word_news" in pt_id:
            level = "CEFR B2-C1 (Advanced)"
            num_sub = "đúng 6 câu hỏi trắc nghiệm điền từ/cụm từ (Question 14 đến 19)"
            format_example = """<group_question>
  <context>Một bài báo ngắn (news piece) bằng tiếng Anh trình độ B2-C1 có tiêu đề (ví dụ: Da Nang International Fireworks Festival (DIFF) 2025...) và nội dung chia thành 3-4 đoạn ngắn, chứa đúng 6 chỗ trống ký hiệu lần lượt là (14) <blank />, (15) <blank />, (16) <blank />, (17) <blank />, (18) <blank />, (19) <blank />. Chú ý phải viết đúng định dạng "<blank />" này (ví dụ: "... (14) <blank /> ... (15) <blank /> ...")</context>
  <question>
    <stem>Question 14.</stem>
    <option>Từ/cụm từ lựa chọn A (Ví dụ: number)</option>
    <option>Từ/cụm từ lựa chọn B (Ví dụ: volume)</option>
    <option>Từ/cụm từ lựa chọn C (Ví dụ: amount)</option>
    <option>Từ/cụm từ lựa chọn D (Ví dụ: level)</option>
    <answer>A</answer>
    <explanation>DIFF 2025 có số lượng đội tuyển tham gia lớn nhất - dùng cụm 'number of'.</explanation>
  </question>
  <question>
    <stem>Question 15.</stem>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <answer>D</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
  <question>
    <stem>Question 16.</stem>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <answer>D</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
  <question>
    <stem>Question 17.</stem>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <answer>B</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
  <question>
    <stem>Question 18.</stem>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <answer>C</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
  <question>
    <stem>Question 19.</stem>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <answer>B</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
</group_question>"""

        elif "cloze_word_leaflet" in pt_id:
            level = "CEFR B2-C1 (Advanced)"
            num_sub = "đúng 6 câu hỏi trắc nghiệm điền từ/cụm từ (Question 20 đến 25)"
            format_example = """<group_question>
  <context>Một tờ rơi (leaflet) hướng dẫn hoặc lời khuyên bằng tiếng Anh trình độ B2-C1 có tiêu đề, các phần heading và bullet points (ví dụ: "How to Manage Your Money Wisely?"). Chứa đúng 6 chỗ trống ký hiệu lần lượt là (20) <blank />, (21) <blank />, (22) <blank />, (23) <blank />, (24) <blank />, (25) <blank />. Chú ý phải viết đúng định dạng "<blank />" này (ví dụ: "... (20) <blank /> ... (21) <blank /> ...")</context>
  <question>
    <stem>Question 20.</stem>
    <option>Từ/cụm từ lựa chọn A (Ví dụ: However)</option>
    <option>Từ/cụm từ lựa chọn B</option>
    <option>Từ/cụm từ lựa chọn C</option>
    <option>Từ/cụm từ lựa chọn D</option>
    <answer>A</answer>
    <explanation>Giải thích ngữ nghĩa và liên từ phù hợp ngữ cảnh tờ rơi.</explanation>
  </question>
  <question>
    <stem>Question 21.</stem>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <answer>B</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
  <question>
    <stem>Question 22.</stem>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <answer>D</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
  <question>
    <stem>Question 23.</stem>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <answer>B</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
  <question>
    <stem>Question 24.</stem>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <answer>C</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
  <question>
    <stem>Question 25.</stem>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <option>Từ/cụm từ...</option>
    <answer>A</answer>
    <explanation>Giải thích lý do...</explanation>
  </question>
</group_question>"""

        elif "reading_comprehension_8" in pt_id:
            level = "CEFR B2-C1 (Advanced)"
            num_sub = "đúng 8 câu hỏi trắc nghiệm đọc hiểu (Question 6 đến 13)"
            format_example = """<group_question>
  <context>Đoạn văn đọc hiểu tiếng Anh trình độ B2-C1 dài khoảng 250-300 từ, chia làm 4 đoạn rõ ràng.</context>
  <question>
    <stem>Question 6: The word settle in paragraph 1 mostly means <blank/>.</stem>
    <option>decide</option>
    <option>exchange</option>
    <option>expect</option>
    <option>announce</option>
    <answer>A</answer>
    <explanation>Trong đoạn 1, 'settle' nghĩa là quyết định/chọn lựa.</explanation>
  </question>
  ... (tiếp tục cho các câu hỏi từ 6 đến 13. Đảm bảo có câu hỏi tóm tắt/paraphrase câu gạch chân, câu hỏi đại từ, từ trái nghĩa ngữ cảnh, câu hỏi đúng/sai, và 2 câu hỏi paragraph matching: 'Which paragraph mentions...')
</group_question>"""

        elif "reading_comprehension_10" in pt_id:
            level = "CEFR B2-C1 (Advanced)"
            num_sub = "đúng 10 câu hỏi trắc nghiệm đọc hiểu (Question 26 đến 35)"
            format_example = """<group_question>
  <context>Đoạn văn đọc hiểu tiếng Anh trình độ B2-C1 dài khoảng 350-400 từ, chia làm các đoạn rõ ràng, có chứa các ký hiệu vị trí xen câu [I], [II], [III], [IV].</context>
  <question>
    <stem>Question 26: According to paragraph 1, ...</stem>
    <option>...</option>
    <option>...</option>
    <option>...</option>
    <option>...</option>
    <answer>A</answer>
    <explanation>Giải thích...</explanation>
  </question>
  ... (tiếp tục cho các câu hỏi từ 26 đến 35. Đảm bảo có câu hỏi tóm tắt: 'Which of the following best summarises the passage?', câu hỏi chèn câu 'Where in the passage does the following sentence best fit?', và câu hỏi paraphrasing: 'Which of the following best paraphrases...')
</group_question>"""

        elif "cloze_word_old" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            num_sub = "đúng 5 câu hỏi trắc nghiệm điền từ (Question 26 đến 30)"
            format_example = """<group_question>
  <context>Đoạn văn đọc điền bằng tiếng Anh trình độ B1-B2 dài khoảng 150-200 từ, chứa đúng 5 chỗ trống ký hiệu lần lượt là (26) <blank />, (27) <blank />, (28) <blank />, (29) <blank />, (30) <blank />. Chú ý phải viết đúng định dạng "<blank />" này (ví dụ: "... (26) <blank /> ... (27) <blank /> ...")</context>
  <question>
    <stem>Question 26:</stem>
    <option>Từ/cụm từ lựa chọn A (không ghi 'A.' ở đầu)</option>
    <option>Từ/cụm từ lựa chọn B...</option>
    <option>Từ/cụm từ lựa chọn C...</option>
    <option>Từ/cụm từ lựa chọn D...</option>
    <answer>C</answer>
    <explanation>Giải thích ngữ pháp/ngữ nghĩa của từ điền vào chỗ trống (26).</explanation>
  </question>
  ... (tiếp tục cho các câu hỏi từ 26 đến 30)
</group_question>"""

        elif "reading_comprehension_5" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            num_sub = "đúng 5 câu hỏi trắc nghiệm đọc hiểu (Question 31 đến 35)"
            format_example = """<group_question>
  <context>Đoạn văn đọc hiểu tiếng Anh trình độ B1-B2 dài khoảng 150-200 từ.</context>
  <question>
    <stem>Question 31: What is the passage mainly about?</stem>
    <option>...</option>
    <option>...</option>
    <option>...</option>
    <option>...</option>
    <answer>C</answer>
    <explanation>Giải thích ý chính của bài đọc.</explanation>
  </question>
  ... (tiếp tục cho các câu hỏi từ 31 đến 35)
</group_question>"""

        elif "reading_comprehension_7" in pt_id:
            level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
            num_sub = "đúng 7 câu hỏi trắc nghiệm đọc hiểu (Question 36 đến 42)"
            format_example = """<group_question>
  <context>Đoạn văn đọc hiểu tiếng Anh trình độ B1-B2 dài khoảng 250-300 từ.</context>
  <question>
    <stem>Question 36: What is the passage mainly about?</stem>
    <option>...</option>
    <option>...</option>
    <option>...</option>
    <option>...</option>
    <answer>B</answer>
    <explanation>Giải thích...</explanation>
  </question>
  ... (tiếp tục cho các câu hỏi từ 36 đến 42)
</group_question>"""

        else:
            # Fallback if no specific English problem type matched
            is_cloze = random.random() < 0.5
            is_cloze_sentence = random.random() < 0.5
            if is_cloze:
                if is_cloze_sentence:
                    level = "CEFR B2-C1 (Advanced)"
                    num_sub = "đúng 5 câu hỏi trắc nghiệm điền câu/mệnh đề"
                else:
                    level = "CEFR B1-B2 (Intermediate to Upper-Intermediate)"
                    num_sub = "đúng 5 câu hỏi trắc nghiệm điền từ"
            else:
                level = "CEFR B2 (Upper-Intermediate)"
                num_sub = "5 đến 8 câu hỏi trắc nghiệm đọc hiểu"
            
            format_example = """<group_question>
  <context>Đoạn văn bằng tiếng Anh...</context>
  <question>
    <stem>Question X: ...</stem>
    <option>...</option>
    <option>...</option>
    <option>...</option>
    <option>...</option>
    <answer>A</answer>
    <explanation>Giải thích bằng tiếng Việt.</explanation>
  </question>
  ...
</group_question>"""

        level_line = f"- Trình độ tiếng Anh (CEFR Level): {level}\n"
    elif subject == Subject.LITERATURE:
        # Reading comprehension for Literature (Group Short Answer)
        num_sub = "4 đến 5 câu hỏi đọc hiểu tự luận ngắn"
        format_example = """<group_question>
  <context>Văn bản đọc hiểu bằng tiếng Việt (thơ, truyện ngắn, trích đoạn bài báo/nghị luận...) kèm theo tên tác giả, tác phẩm nguồn.</context>
  <question>
    <stem>Câu hỏi đọc hiểu tự luận (ví dụ: "Câu 1. Xác định ngôi kể/phương thức biểu đạt chính được sử dụng trong văn bản trên.")</stem>
    <answer>Đáp án hướng dẫn chấm chi tiết bằng tiếng Việt cho câu hỏi tự luận này, bao gồm cả mức điểm tối đa và hướng dẫn cho điểm thành phần.</answer>
    <explanation>Lời giải thích/phân tích chi tiết bằng tiếng Việt</explanation>
  </question>
  ... (tiếp tục cho các câu hỏi tiếp theo)
</group_question>"""
    else:
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
    {level_line}
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
    {english_rules}"""
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
    
    # Skip curriculum files for English and Literature
    if subject not in [Subject.ENGLISH, Subject.LITERATURE]:
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
    else:
        # Simulate problem_type_info using filters for English and Literature
        if problem_type_filter:
            problem_type_info = {
                "chapter": "Tổng hợp",
                "unit": "Tổng hợp",
                "id": problem_type_filter,
                "name": problem_type_filter,
                "details": f"Dạng bài tập: {problem_type_filter}",
                "examples": [],
                "cognitive_level": "VD"
            }
            
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
        # Custom logic for English and Literature if question_type is not explicitly specified
        if subject == Subject.ENGLISH:
            # 30% probability of group cloze/reading questions
            is_group = random.random() < 0.30
            if is_group:
                actual_type = QuestionType.GROUP_MULTIPLE_CHOICE
                system_prompt, user_prompt = make_group_prompt(
                    subject, grade, actual_type, difficulty, problem_type_info,
                    include_figure=include_figure, include_table=include_table
                )
            else:
                actual_type = QuestionType.MULTIPLE_CHOICE
                system_prompt, user_prompt = make_standard_prompt(
                    subject, grade, actual_type, difficulty, problem_type_info,
                    include_figure=include_figure, include_table=include_table
                )
        elif subject == Subject.LITERATURE:
            # 50% probability of group reading comprehension, 50% standalone essays
            is_group = random.random() < 0.50
            if is_group:
                actual_type = QuestionType.GROUP_SHORT_ANSWER
                system_prompt, user_prompt = make_group_prompt(
                    subject, grade, actual_type, difficulty, problem_type_info,
                    include_figure=include_figure, include_table=include_table
                )
            else:
                actual_type = QuestionType.SHORT_ANSWER
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

import json
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any

def parse_xml_annotations(tagged_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    allowed_tags = {"question_label", "stem", "option_label", "option_text", "context", "section"}
    raw_chars = []
    spans = []

    tag_pattern = re.compile(r"<(/)?([a-zA-Z_0-9]+)>")

    pos = 0
    current_open_tag = None
    tag_start_idx = -1

    for match in tag_pattern.finditer(tagged_text):
        start, end = match.span()
        # Add the text before the tag to raw_chars
        text_before = tagged_text[pos:start]
        raw_chars.append(text_before)

        is_closing = bool(match.group(1))
        tag_name = match.group(2)

        if tag_name in allowed_tags:
            if not is_closing:
                # Open tag
                current_open_tag = tag_name
                tag_start_idx = len("".join(raw_chars))
            else:
                # Close tag
                if current_open_tag == tag_name and tag_start_idx != -1:
                    tag_end_idx = len("".join(raw_chars))
                    span_text = "".join(raw_chars)[tag_start_idx:tag_end_idx]
                    spans.append(
                        {
                            "start": tag_start_idx,
                            "end": tag_end_idx,
                            "label": tag_name,
                            "text": span_text,
                        }
                    )
                current_open_tag = None
                tag_start_idx = -1
        else:
            # If it's an unallowed tag, treat it as literal text
            raw_chars.append(match.group(0))

        pos = end

    raw_chars.append(tagged_text[pos:])
    raw_text = "".join(raw_chars)

    return raw_text, spans

# Original text file
file_path = Path("d:/project/doc-layout-analysis/sequence-labelling-data-generator/real_data_annotator/out/toan/de-chon-hsg-toan-thpt-nam-2025-2026-truong-chuyen-luong-van-chanh-dak-lak.md")
with open(file_path, "r", encoding="utf-8") as f:
    original_text = f.read()

# Tagged XML text candidate
tagged_xml = """<|page|>Page 1

SỞ GD-ĐT ĐẮK LẮK                                    KỲ THI LẬP ĐỘI TUYỂN DỰ THI
TRƯỜNG THPT CHUYÊN     CHỌN HỌC SINH GIỎI CẤP TỈNH THPT - GDTX
LƯƠNG VĂN CHÁNH                          NĂM HỌC 2025-2026
<u>ĐỀ CHÍNH THỨC</u>                                   Môn thi: Toán
(Đề thi có 1 trang, 5 câu)                       Ngày thi: 03/02/2026
                            Thời gian làm bài: 180 phút (không kể thời gian giao đề)

<question_label>Câu 1 (4,00 điểm).</question_label> <question_label>1.</question_label> <stem>Tìm tất cả các giá trị thực của tham số $m$ để hàm số $y = f(x) = x^6 - 5x^3 + mx + 2$ đồng biến trên khoảng $(0; +\infty)$.</stem>

<question_label>2.</question_label> <stem>Cho các hàm số $y = f(x), y = f[f(x)], y = f(\sqrt{x^2+24})$ có đồ thị lần lượt là $(C_1), (C_2), (C_3)$. Đường thẳng $x = 1$ cắt $(C_1), (C_2), (C_3)$ lần lượt tại các điểm $M, N, P$. Biết phương trình tiếp tuyến của $(C_1)$ tại $M$ và của $(C_2)$ tại $N$ lần lượt là $y = 2x + 3$ và $y = 202(10x+1)$. Viết phương trình tiếp tuyến của $(C_3)$ tại $P$.</stem>

<question_label>Câu 2 (4,00 điểm).</question_label> <stem>Giải hệ phương trình $\\begin{cases} y^4 - 16y^2 + 15 = 2x(3y^2 - 4x - 17) \\\\ (y^2 + 2x - 15)\\left(\\sqrt{5x+1} - \\sqrt{y^2 + x + 3}\\right) = 14 \\end{cases}$ với $x, y \\in \\mathbb{R}$.</stem>

<question_label>Câu 3 (4,00 điểm).</question_label> <question_label>1.</question_label> <stem>Cho một đa giác đều 8 cạnh. Chọn ngẫu nhiên 3 đỉnh trong 8 đỉnh của đa giác. Tìm xác suất để 3 đỉnh được chọn là 3 đỉnh của tam giác vuông.</stem>

<question_label>2.</question_label> <stem>Cho lăng trụ đứng $ABC.A'B'C'$ có đáy là tam giác cân tại $C$, $AB = 2a$, $AA' = a$ và $BC'$ tạo với mặt phẳng $(ABB'A')$ một góc bằng $60°$. Gọi $N$ là trung điểm của $AA'$ và $M$ là trung điểm của $BB'$. Tính theo $a$ thể tích khối lăng trụ $ABC.A'B'C'$ và khoảng cách từ $M$ đến mặt phẳng $(BC'N)$.</stem>

<question_label>Câu 4 (4,00 điểm).</question_label> <question_label>1)</question_label> <stem>Cho tam giác $ABC$ nhọn nội tiếp đường tròn $(O)$ có $BC$ cố định, $A$ di chuyển trên cung lớn $BC$, $M$ là trung điểm $BC$. Đường thẳng qua $M$ song song với $AB, AC$ lần lượt cắt tiếp tuyến tại $A$ của $(O)$ tại $E, F$. Đoạn thẳng $BF, CE$ lần lượt cắt đường tròn $(O)$ tại điểm thứ hai là $K, L$.</stem>

<option_label>a)</option_label> <option_text>Chứng minh rằng $FB \\parallel EC$ và $FK \cdot EC = FB \cdot EL$.</option_text>

<option_label>b)</option_label> <option_text>Gọi $X$ là giao điểm của $BL$ và $CK$. Chứng minh $AX$ luôn đi qua một điểm cố định khi $A$ di chuyển.</option_text>

<question_label>2)</option_label> <stem>Cho tập hợp $A = \\{0; 1; 2; 3; 4; 5; 6\\}$. Có bao nhiêu số tự nhiên chẵn có 5 chữ số đôi một khác nhau được lập thành từ các chữ số của tập $A$, đồng thời có đúng 2 chữ số lẻ và 2 chữ số lẻ đó đứng cạnh nhau.</stem>

<question_label>Câu 5 (4,00 điểm).</question_label> <question_label>1.</question_label> <stem>Cho dãy số xác định bởi

$$u_1 = 1, u_{n+1} = \\frac{1}{3}\\left(2u_n + \\frac{n-1}{n^2+3n+2}\\right); n \\in \\mathbb{N}^*.$$

Tính $u_{2026}$.</stem>

<question_label>2.</question_label> <stem>Một người bắt đầu đi làm được nhận được số tiền lương là 6 000 000 đ một tháng. Sau 24 tháng người đó được tăng lương 10%. Hằng tháng người đó tiết kiệm 30% lương để gửi vào ngân hàng với lãi suất 0,3%/ tháng theo hình thức lãi kép (tiền lãi của kỳ trước được cộng vào tiền gốc để tính lãi cho kỳ tiếp theo). Biết rằng người đó nhận lương vào đầu tháng và số tiền tiết kiệm được chuyển ngay vào ngân hàng. Sau 36 tháng tính từ lúc bắt đầu đi làm, tổng số tiền người đó tiết kiệm được là bao nhiêu?</stem>

—Hết—

Họ và tên thí sinh:........................................; Số báo danh.......................
Chữ ký của giám thị 1:............................; Chữ kí của giám thị 2:...............

Trang 1/1

"""

raw_text, spans = parse_xml_annotations(tagged_xml)

if raw_text == original_text:
    print("SUCCESS: The stripped text matches the original text exactly!")
    
    # Save files
    output_dir = Path("d:/project/doc-layout-analysis/sequence-labelling-data-generator/output/real_exams")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path_hash = "bc1607cf"
    
    json_path = output_dir / f"real_exam_{path_hash}.json"
    xml_path = output_dir / f"real_exam_{path_hash}.xml"
    
    result_data = {
        "exam_id": f"real_{path_hash}",
        "created_at": datetime.now().isoformat(),
        "is_real": True,
        "raw_text": raw_text,
        "spans": spans,
        "raw_xml": tagged_xml,
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON to {json_path}")
    
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(tagged_xml)
    print(f"Saved XML to {xml_path}")
    
else:
    print("ERROR: Mismatch found!")
    # Find first mismatch
    min_len = min(len(original_text), len(raw_text))
    for i in range(min_len):
        if original_text[i] != raw_text[i]:
            print(f"First mismatch at index {i}:")
            print(f"Original: {repr(original_text[i-20:i+20])}")
            print(f"Stripped: {repr(raw_text[i-20:i+20])}")
            break
    if len(original_text) != len(raw_text):
        print(f"Lengths differ: Original={len(original_text)}, Stripped={len(raw_text)}")

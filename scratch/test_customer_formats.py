import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.inference.predict import load_inference_model, predict_text

MODEL_ID = "daominhwysi/mmbert-small-vi-exam-seq-labeling"

CUSTOMER_TEST_CASES = [
    {
        "category": "IELTS Reading - Headings & TFNG Format",
        "text": """READING PASSAGE 1
You should spend about 20 minutes on Questions 1-8.

The Secrets of the Maya Civilization

Paragraph A
The Maya civilization was a Mesoamerican civilization developed by the Maya peoples, noted for its logosyllabic script as well as for its art, architecture, mathematics, calendar, and astronomical system.

Paragraph B
Archaeological excavations in the tropical lowlands of northern Guatemala have revealed extensive urban settlements. Sophisticated agricultural techniques, including raised-field cultivation and terracing, supported large populations.

Questions 1-2
Choose the correct heading for paragraphs A and B from the list of headings below.
List of Headings:
i. Agricultural innovation and urban centers
ii. Overview of cultural and scientific achievements
iii. Theories regarding the decline of Maya cities

1. Paragraph A
2. Paragraph B

Questions 3-4
Do the following statements agree with the information given in Reading Passage 1?
TRUE if the statement agrees with the information
FALSE if the statement contradicts the information
NOT GIVEN if there is no information on this

3. The Maya developed a sophisticated calendar system.
4. The Maya population declined primarily due to warfare.
"""
    },
    {
        "category": "Digital SAT Reading & Writing Format",
        "text": """Module 1: Reading and Writing

Question 1:
Many animals use camouflage to blend into their surroundings and avoid predators. In contrast, the brightly colored poison dart frog utilizes aposematism—a warning coloration that signals its toxicity to potential predators. Thus, conspicuous coloration in this species serves as a deterrent rather than an invitation.

Which choice best describes the main function of the underlined sentence?
(A) It provides an exception to a general biological principle.
(B) It summarizes the adaptive advantage of the frog's coloration.
(C) It questions the effectiveness of camouflage strategies.
(D) It introduces a newly discovered defensive mechanism.
"""
    },
    {
        "category": "ĐGNL / ĐGTD (ĐH Bách Khoa / ĐHQG) - True/False Matrix & Short Answer",
        "text": """PHẦN II. Câu trắc nghiệm đúng sai. Thí sinh trả lời từ câu 1 đến câu 2. Trong mỗi ý a), b), c), d) ở mỗi câu, thí sinh chọn đúng hoặc sai.

Câu 1: Cho hàm số bậc ba $y = f(x) = ax^3 + bx^2 + cx + d$ có đồ thị như hình vẽ.
a) Hàm số đồng biến trên khoảng $(-\infty; 1)$.
b) Điểm cực đại của đồ thị hàm số là $M(1; 3)$.
c) Giá trị nhỏ nhất của hàm số trên đoạn $[0; 3]$ bằng $-1$.
d) Phương trình $f(x) - 2 = 0$ có đúng 3 nghiệm thực phân biệt.

PHẦN III. Câu trắc nghiệm trả lời ngắn. Thí sinh trả lời từ câu 3 đến câu 4.

Câu 3: Một vật chuyển động theo quy luật $s(t) = -t^3 + 9t^2 + t$ với $t$ (giây) là khoảng thời gian từ lúc bắt đầu chuyển động. Tìm vận tốc lớn nhất của vật (đơn vị: $m/s$).

Câu 4: Có bao nhiêu giá trị nguyên của tham số $m \in [-10; 10]$ để hàm số nghịch biến trên $\mathbb{R}$?
"""
    },
    {
        "category": "Multilingual Exams (JLPT / HSK / Circular Numbers ① ② ③ ④)",
        "text": """問題1: 次の文の言葉はどう読みますか。最もよいものを 1・2・3・4 から一つ選びなさい。

1. 毎朝、公園を散歩します。
① さんぼ
② さんぽ
③ ざんぼ
④ ざんぽ

2. 彼女は親切な人です。
(1) しんせつ
(2) しんぜつ
(3) じんせつ
(4) じんぜつ
"""
    },
    {
        "category": "Exam Metadata, Watermarks, Headers & Page Numbers",
        "text": """SỞ GD&ĐT HÀ NỘI
TRƯỜNG THPT CHUYÊN HÀ NỘI - AMSTERDAM
ĐỀ THI THỬ TỐT NGHIỆP THPT NĂM 2026
MÃ ĐỀ THI: 102 - Số trang: 4 trang

Mark the letter A, B, C, or D on your answer sheet to indicate the correct answer.

Câu 1: If she _____ harder, she would pass the entrance examination with flying colors.
A. studied
B. studies
C. study
D. had studied

--- Trang 1/4 - Mã đề thi 102 ---

Câu 2: The company decided to _____ all outdated equipment before moving to the new facility.
A. dispose of
B. take off
C. look after
D. turn down
"""
    }
]

def main():
    print("=" * 80)
    print(f"EVALUATING CUSTOMER EXAM FORMATS ON: {MODEL_ID}")
    print("=" * 80)
    model, tokenizer, id_to_tag, device = load_inference_model(MODEL_ID)

    for i, test in enumerate(CUSTOMER_TEST_CASES, 1):
        print(f"\n[{i}/{len(CUSTOMER_TEST_CASES)}] FORMAT: {test['category']}")
        print("-" * 80)
        res = predict_text(test["text"], model, tokenizer, id_to_tag, device)
        print("XML RESULT:")
        print(res["xml_content"])
        print("-" * 80)

if __name__ == "__main__":
    main()

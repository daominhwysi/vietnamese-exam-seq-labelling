import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.inference.predict import load_inference_model, predict_text

MODEL_ID = "daominhwysi/mmbert-small-vi-exam-seq-labeling"

ADDITIONAL_TESTS = [
    {
        "name": "Bilingual Vietnamese-English TOEIC Prep Format",
        "text": """PART 5: Chọn đáp án đúng nhất để hoàn thành câu.

Câu 101: The marketing team will _____ the new campaign strategy tomorrow.
(A) discuss
(B) discussion
(C) discussible
(D) discussing

Câu 102: According to the survey, most employees prefer working from home _____ Fridays.
A. in
B. on
C. at
D. to
"""
    },
    {
        "name": "TOEIC Part 7 - Table & Invoice Format",
        "text": """READING TEST
Questions 153-154 refer to the following order form.

Apex Office Supplies
Order Date: September 4
Customer: Bright Future Academy

Item | Quantity | Unit Price | Total
Laser Printer | 2 | $250.00 | $500.00
Printer Cartridges | 6 | $45.00 | $270.00
Copy Paper (Box) | 10 | $30.00 | $300.00

Subtotal: $1,070.00
Shipping: FREE (Orders over $500)
Total Amount Due: $1,070.00

Question 153: What item was ordered in the greatest quantity?
(A) Laser Printer
(B) Copy Paper
(C) Printer Cartridges
(D) Office Desk

Question 154: Why did the customer receive free shipping?
(A) They used a promotional coupon.
(B) The order total exceeded $500.
(C) They are a first-time customer.
(D) The delivery location is nearby.
"""
    },
    {
        "name": "TOEIC Part 5 - Mixed Blank Types (_____, ....., [BLANK])",
        "text": """105. All visitors are required to sign in at the front desk before entering the building ..... security purposes.
(A) for
(B) with
(C) by
(D) from

106. Please review the attached contract [BLANK] sign it if you agree with the terms.
(A) and
(B) or
(C) but
(D) so
"""
    }
]

def main():
    print("=" * 80)
    print(f"ADDITIONAL TOEIC FORMAT EVALUATION: {MODEL_ID}")
    print("=" * 80)
    model, tokenizer, id_to_tag, device = load_inference_model(MODEL_ID)

    for i, t in enumerate(ADDITIONAL_TESTS, 1):
        print(f"\n[{i}/{len(ADDITIONAL_TESTS)}] {t['name']}")
        print("-" * 80)
        res = predict_text(t["text"], model, tokenizer, id_to_tag, device)
        print("XML RESULT:")
        print(res["xml_content"])
        print("-" * 80)

if __name__ == "__main__":
    main()

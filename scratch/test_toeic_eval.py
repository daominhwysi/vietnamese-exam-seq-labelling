import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.inference.predict import load_inference_model, predict_text

MODEL_ID = "daominhwysi/mmbert-small-vi-exam-seq-labeling"

TOEIC_TEST_CASES = [
    {
        "part": "TOEIC Part 5 - Incomplete Sentences (Standard Format with 'Question 101:' and '(A)...')",
        "text": """PART 5 - INCOMPLETE SENTENCES
Directions: A word or phrase is missing in each of the sentences below. Four answer choices are given below each sentence. Select the best answer to complete the sentence.

Question 101: Ms. Clara requested that the financial report be delivered to her office _____ Friday morning.
(A) by
(B) on
(C) at
(D) with

Question 102: The newly designed website makes it easier for customers to navigate _____ products efficiently.
(A) between
(B) through
(C) among
(D) across
"""
    },
    {
        "part": "TOEIC Part 5 - Numbered format ('103.', '104.', inline options)",
        "text": """103. All employees must submit their travel expense reports _____ two weeks of returning from a business trip.
A. within
B. inside
C. along
D. upon

104. Mr. Tanaka is _____ looking for a qualified assistant to manage international logistics.
(A) currently  (B) current  (C) currency  (D) cure
"""
    },
    {
        "part": "TOEIC Part 6 - Text Completion (Passage with Blanks & Sub-questions)",
        "text": """PART 6 - TEXT COMPLETION
Questions 131-134 refer to the following email.

To: All Staff <staff@apexsolutions.com>
From: Human Resources <hr@apexsolutions.com>
Date: October 15
Subject: Annual Health & Wellness Fair

We are pleased to announce our upcoming Annual Health & Wellness Fair, scheduled for Friday, November 10. The event will take place in the main auditorium from 9:00 A.M. to 4:00 P.M. 

Various healthcare providers will offer free health screenings, including blood pressure and cholesterol tests. _____ [131] _____, there will be workshops on stress management and nutrition throughout the day. 

Employees who wish to attend the workshops are requested to register in advance _____ [132] _____ seating is limited. Please visit the company portal to reserve your spot. 

We look forward to your active participation in making this event a success.

131. (A) In addition
(B) Otherwise
(C) In contrast
(D) However

132. (A) because
(B) although
(C) despite
(D) unless

133. (A) They have confirmed their attendance already.
(B) Healthy refreshments will be provided free of charge.
(C) The parking lot will be closed for renovation.
(D) Please submit your overtime requests promptly.

134. (A) participate
(B) participation
(C) participated
(D) participant
"""
    },
    {
        "part": "TOEIC Part 7 - Reading Comprehension (Single Passage: Notice/Memo with Questions)",
        "text": """PART 7 - READING COMPREHENSION
Questions 147-148 refer to the following notice.

ATTENTION METRO PASSENGERS
Track Maintenance Notice - Green Line

Please be advised that track maintenance work will be carried out on the Green Line between Central Station and North Park Station from Saturday, July 12 through Sunday, July 13.

During this period, train services will be suspended along this section. Complimentary shuttle buses will operate every 10 minutes to transport passengers between the affected stations. Normal train operations will resume at 5:00 A.M. on Monday, July 14.

We apologize for any inconvenience caused.

147. What is the notice mainly about?
(A) A fare increase for public transit
(B) Temporary disruption in train service
(C) Safety guidelines for passengers
(D) Opening of a new station

148. What will be provided for passengers during the maintenance?
(A) Discount vouchers
(B) Shuttle bus service
(C) Free subway passes
(D) Refund forms
"""
    },
    {
        "part": "TOEIC Part 7 - Double Passage (Email + Schedule)",
        "text": """Questions 181-185 refer to the following advertisement and email.

[Passage 1: Advertisement]
GreenTech Conference 2026
Join over 500 industry leaders on March 24-25 at the Grand Pacific Hotel in Seattle. Early bird registration is $350 until February 15. Standard registration after February 15 is $450. Group discounts of 15% are available for parties of 5 or more.

[Passage 2: Email]
To: Sarah Jenkins <s.jenkins@ecobuild.com>
From: David Miller <d.miller@ecobuild.com>
Date: February 10
Subject: GreenTech Conference Attendance

Hi Sarah,
I reviewed the conference schedule and believe our sustainability team of six engineers should attend. Could you please proceed with the group registration before the early bird deadline?

Question 181: What is indicated about the GreenTech Conference?
A. It is held annually in Seattle.
B. It offers a discount for early registration.
C. It lasts for three full days.
D. It is free for local residents.

Question 182: How many people from EcoBuild will attend the conference?
A. Two
B. Four
C. Six
D. Eight
"""
    },
    {
        "part": "TOEIC Part 2 & 3 - Listening Comprehension Transcript Format",
        "text": """PART 2 - QUESTION-RESPONSE
Directions: You will hear a question or statement and three responses. Select the best response to the question or statement.

7. Where did you leave the keys to the conference room?
(A) On the reception desk.
(B) Yes, the meeting went well.
(C) About two hours ago.

8. Who is responsible for reviewing the marketing proposal?
(A) In the conference room.
(B) Ms. Patterson in legal.
(C) Tomorrow afternoon.
"""
    }
]

def main():
    print("=" * 80)
    print(f"EVALUATING MODEL: {MODEL_ID}")
    print("TASK: TOEIC EXAM FORMAT SEQUENCE LABELING")
    print("=" * 80)

    start_time = time.time()
    model, tokenizer, id_to_tag, device = load_inference_model(MODEL_ID)
    print(f"Model loaded in {time.time() - start_time:.2f}s on {device}")
    print(f"Target Tags: {id_to_tag}")
    print("=" * 80)

    total_cases = len(TOEIC_TEST_CASES)

    for idx, test_case in enumerate(TOEIC_TEST_CASES, 1):
        print(f"\n[{idx}/{total_cases}] TEST CASE: {test_case['part']}")
        print("-" * 80)
        t0 = time.time()
        res = predict_text(test_case["text"], model, tokenizer, id_to_tag, device)
        elapsed = time.time() - t0

        print(f"Inference completed in {elapsed:.2f}s ({res['token_count']} tokens)")
        print("\n--- EXTRACTED SEGMENTS ---")
        for seg in res["segments"]:
            badge = f"[{seg['label'].upper()}]"
            print(f"  {badge:<18} : {seg['text']}")

        print("\n--- INLINE XML RESULT ---")
        print(res["xml_content"])
        print("-" * 80)

if __name__ == "__main__":
    main()

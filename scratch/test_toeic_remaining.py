import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.inference.predict import load_inference_model, predict_text

MODEL_ID = "daominhwysi/mmbert-small-vi-exam-seq-labeling"

REMAINING_TEST_CASES = [
    {
        "name": "1. Part 7 - Online Chat / Instant Messaging Chain",
        "text": """Questions 158-160 refer to the following text message chain.

[8:45 AM] Marcus Bennett: Good morning team. Does anyone know if the conference room on the 4th floor is available this afternoon?
[8:47 AM] Clara Rios: I checked earlier, and the marketing department has booked it from 1:00 PM to 3:00 PM.
[8:48 AM] Marcus Bennett: Thanks Clara. We only need it from 3:30 PM for about an hour.
[8:50 AM] Clara Rios: That should be fine. I will reserve it on the internal calendar for you right now.

158. What is Mr. Bennett asking about?
(A) The location of a marketing presentation
(B) The availability of a meeting room
(C) The schedule for the company holiday party
(D) The contact information for IT support

159. At 8:48 AM, what does Mr. Bennett mean when he writes, "Thanks Clara"?
(A) He appreciates her quick response.
(B) He will attend the marketing presentation.
(C) He already made the reservation himself.
(D) He wants Clara to cancel the meeting.

160. What will Ms. Rios do next?
(A) Speak with the marketing director
(B) Book the room on the company calendar
(C) Send an email to Mr. Bennett's supervisor
(D) Set up chairs in the conference room
"""
    },
    {
        "name": "2. Part 7 - Triple Passage (Article + Email + Schedule)",
        "text": """Questions 186-190 refer to the following announcement, email, and schedule.

[Passage 1: Announcement]
Riverside Community Center Renovation Project
The Riverside Community Center will undergo major interior renovations starting June 1st. Facilities including the gym and indoor swimming pool will be closed until August 15th. All other rooms will remain open with adjusted operating hours.

[Passage 2: Email]
To: Members <members@riversidecc.org>
From: Kevin Vance, Director <k.vance@riversidecc.org>
Date: May 20
Subject: Class Relocations During Renovation

Dear Members,
Due to the gym renovation, our morning fitness classes will be moved to the multi-purpose room on the second floor. Please check the revised class timetable below.

[Passage 3: Revised Class Timetable]
Monday / Wednesday: 8:00 AM - Yoga Basics (Room 201)
Tuesday / Thursday: 9:00 AM - Pilates (Room 202)
Saturday: 10:00 AM - Aerobics (Room 201)

Question 186: What is the main purpose of the announcement?
(A) To introduce new fitness instructors
(B) To inform members about facility closures
(C) To advertise an upcoming athletic tournament
(D) To request donations for community programs

Question 187: According to the email, where will morning fitness classes take place?
(A) In the main gymnasium
(B) In the second-floor multi-purpose room
(C) At an outdoor sports field
(D) In the basement exercise area
"""
    },
    {
        "name": "3. Part 7 - Sentence Placement Question & Synonym Question",
        "text": """Questions 161-163 refer to the following article.

TechNova Solutions announced today that it will expand its software engineering division in Boston. [1] The company plans to hire over 150 developers within the next two quarters. [2] Company executives attribute this rapid growth to the soaring demand for their cloud-based analytics platform. [3] The expansion is anticipated to be complete by December. [4]

161. In which of the positions marked [1], [2], [3], and [4] does the following sentence best belong?
"To accommodate the new personnel, TechNova has leased two additional floors in the Prudential Center."
(A) [1]
(B) [2]
(C) [3]
(D) [4]

162. In the article, the word "soaring" in sentence 3 is closest in meaning to
(A) flying
(B) rising
(C) declining
(D) changing
"""
    },
    {
        "name": "4. Part 3 & 4 - Conversation Transcript & Graphic/Visual Question",
        "text": """PART 3: CONVERSATIONS
Questions 41 through 43 refer to the following conversation and flight schedule.

(Man): Excuse me, I'm waiting for flight 402 to Chicago, but the gate information on the board hasn't updated yet.
(Woman): Flight 402 has been delayed by 45 minutes due to inclement weather. Passengers are now boarding at Gate B12 instead of Gate B4.
(Man): I see. Will there still be a connecting shuttle once we land?
(Woman): Yes, airport staff in Chicago will guide passengers to the connecting shuttles upon arrival.

41. Why is the man concerned?
(A) He lost his boarding pass.
(B) The gate display information has not updated.
(C) His luggage was misplaced.
(D) His flight was cancelled completely.

42. Look at the graphic. Which gate will flight 402 depart from?
(A) Gate B4
(B) Gate B8
(C) Gate B12
(D) Gate C2

43. What does the woman say will happen upon arrival in Chicago?
(A) Baggage claim will be expedited.
(B) Meal vouchers will be distributed.
(C) Airport staff will assist with connecting shuttles.
(D) Free hotel accommodations will be arranged.
"""
    },
    {
        "name": "5. TOEIC Exam Prep with Vietnamese Explanations & Answer Keys Embedded",
        "text": """Question 108: The customer service department has _____ improved its response time over the past quarter.
(A) significant
(B) significantly
(C) signify
(D) significance
* Đáp án đúng: B
* Giải thích: Cần một phó từ (adverb) bổ nghĩa cho động từ "improved". "Significantly" có nghĩa là một cách đáng kể.

Question 109: All attendees must wear their name badges at all times _____ the conference premises.
A. during
B. throughout
C. between
D. among
* Đáp án: B. throughout the conference premises (khắp khuôn viên hội nghị).
"""
    },
    {
        "name": "6. OCR Noise & Layout Variations (Hyphenation, Header/Footer, Lowercase Options)",
        "text": """TOEIC PRACTICE TEST 2026 - ETS FORMAT - PAGE 12

110. The manager praised the team for their outstand-
ing contribution to the annual pro-
ject.
a) remark
b) remarkably
c) remarkable
d) remarking

111. Neither the director nor the assistant manag-
ers were able to attend the meet-
ing yesterday.
a. was
b. were
c. is
d. are
"""
    }
]

def main():
    print("=" * 80)
    print(f"EVALUATING REMAINING TOEIC FORMATS ON: {MODEL_ID}")
    print("=" * 80)
    model, tokenizer, id_to_tag, device = load_inference_model(MODEL_ID)

    for i, test in enumerate(REMAINING_TEST_CASES, 1):
        print(f"\n[{i}/{len(REMAINING_TEST_CASES)}] {test['name']}")
        print("-" * 80)
        res = predict_text(test["text"], model, tokenizer, id_to_tag, device)
        print("XML RESULT:")
        print(res["xml_content"])
        print("-" * 80)

if __name__ == "__main__":
    main()

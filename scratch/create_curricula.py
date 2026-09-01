import json
from pathlib import Path

curriculum_toeic = {
  "subject": "toeic",
  "grade": 12,
  "chapters": [
    {
      "name": "Part 1 - Photographs (Photographic Descriptions)",
      "units": [
        {
          "name": "Workplace, Office & Human Action Scenes",
          "problem_types": [
            {
              "id": "toeic_part1_human_action",
              "name": "Photographs of People in Workplace Action",
              "cognitive_level": "NB_TH",
              "details": "Description of people performing office, laboratory, or meeting tasks. 4 choices (A), (B), (C), (D) representing present continuous active/passive verbs. Correct statement matches subject-verb-object dynamics in workplace contexts.",
              "examples": [
                "(A) They are reviewing blueprints on a table.\n(B) They are packing boxes into a van.\n(C) One of the men is adjusting his necktie.\n(D) A document is being fed into a photocopier."
              ]
            },
            {
              "id": "toeic_part1_object_spatial",
              "name": "Photographs of Spatial Relations & Office Equipment",
              "cognitive_level": "NB_TH",
              "details": "Description of static indoor rooms, arrangement of furniture, stacked files, or computer monitors. Focuses on prepositions of place and passive states (e.g. 'has been set up', 'are hanging on the wall').",
              "examples": [
                "(A) Chairs are arranged around a conference table.\n(B) The curtains have been drawn shut.\n(C) A potted plant is resting on the windowsill.\n(D) All the computer monitors are turned off."
              ]
            },
            {
              "id": "toeic_part1_outdoor_transit",
              "name": "Photographs of Transit, Construction & Outdoor Environments",
              "cognitive_level": "NB_TH",
              "details": "Description of streets, airports, harbors, construction sites, pedestrian walkways, or cargo loading areas.",
              "examples": [
                "(A) Pedestrians are crossing at an intersection.\n(B) Cargo is being unloaded from a vessel.\n(C) Scaffolding has been erected alongside the building.\n(D) Vehicles are parked parallel to the curb."
              ]
            }
          ]
        }
      ]
    },
    {
      "name": "Part 2 - Question-Response (Spoken Interactions)",
      "units": [
        {
          "name": "Information Questions & Polite Inquiries",
          "problem_types": [
            {
              "id": "toeic_part2_wh_questions",
              "name": "Wh-Questions (Who, Where, When, What, Why, How)",
              "cognitive_level": "NB_TH",
              "details": "Prompt is a single question asking for factual business information (locations, personnel, deadlines, reasons, or methods). 3 answer choices (A), (B), (C).",
              "examples": [
                "Where is the annual shareholder meeting taking place this year?\n(A) In the grand ballroom on the third floor.\n(B) Yes, I received the invitation yesterday.\n(C) At approximately two o'clock."
              ]
            },
            {
              "id": "toeic_part2_indirect_responses",
              "name": "Indirect, Conversational & Unexpected Responses",
              "cognitive_level": "VD",
              "details": "Prompt is a suggestion, statement, or inquiry where the correct answer provides context-aware indirect information rather than a simple literal answer.",
              "examples": [
                "Shouldn't we order additional printer toner cartridges?\n(A) Check the supply cabinet in room 204 first.\n(B) The printout was double-sided.\n(C) He was promoted last Tuesday."
              ]
            }
          ]
        }
      ]
    },
    {
      "name": "Part 3 - Short Conversations (Dialogues & Schedules)",
      "units": [
        {
          "name": "Workplace Operations & Customer Services",
          "problem_types": [
            {
              "id": "toeic_part3_dialogue_operations",
              "name": "Workplace Operations & Project Deadlines",
              "cognitive_level": "VD",
              "details": "A transcript of a 2-3 speaker conversation regarding budget allocations, vendor negotiations, software migration, or event logistics, followed by 3 multiple choice questions.",
              "examples": [
                "Questions 32-34 refer to the following conversation."
              ]
            },
            {
              "id": "toeic_part3_dialogue_graphic",
              "name": "Conversation with Graphic / Timetable Reference",
              "cognitive_level": "VDC",
              "details": "Conversation where speakers explicitly refer to an included text table, flight timetable, or price list, followed by 3 questions.",
              "examples": [
                "Questions 41-43 refer to the following conversation and flight schedule."
              ]
            }
          ]
        }
      ]
    },
    {
      "name": "Part 4 - Short Talks & Monologues",
      "units": [
        {
          "name": "Announcements, Voicemails & Presentations",
          "problem_types": [
            {
              "id": "toeic_part4_public_announcements",
              "name": "Transit, Facility & Corporate Announcements",
              "cognitive_level": "VD",
              "details": "A monologue transcript delivering public instructions (gate change, facility maintenance, store promotions, tour guide overview), followed by 3 questions.",
              "examples": [
                "Questions 71-73 refer to the following announcement."
              ]
            },
            {
              "id": "toeic_part4_voicemail_messages",
              "name": "Telephone Voicemail Messages & Briefings",
              "cognitive_level": "VD",
              "details": "Recorded voicemail detailing an inquiry, order delay, or request for callback, followed by 3 questions.",
              "examples": [
                "Questions 84-86 refer to the following telephone message."
              ]
            }
          ]
        }
      ]
    },
    {
      "name": "Part 5 - Incomplete Sentences (Grammar & Collocations)",
      "units": [
        {
          "name": "Grammar, Parts of Speech & Collocations",
          "problem_types": [
            {
              "id": "toeic_part5_word_form",
              "name": "Word Form & Morphological Derivation",
              "cognitive_level": "NB_TH",
              "details": "Tests grammatical suffix recognition (e.g. decisive, decision, decisively, decide) in typical enterprise contexts.",
              "examples": [
                "The board of directors commended Ms. Jensen for handling the client negotiations _____.\n(A) decisive\n(B) decisively\n(C) decision\n(D) decisiveness"
              ]
            },
            {
              "id": "toeic_part5_verb_tenses_voice",
              "name": "Verb Tenses, Passive Voice & Conditionals",
              "cognitive_level": "VD",
              "details": "Tests subject-verb agreement, present perfect vs simple past, passive constructions, and subjunctive verb forms (recommend that he be informed).",
              "examples": [
                "By the time the new facility opens next quarter, the engineering team _____ the safety trial.\n(A) will complete\n(B) will have completed\n(C) completed\n(D) has completed"
              ]
            },
            {
              "id": "toeic_part5_conjunctions_prepositions",
              "name": "Conjunctions, Prepositions & Transition Words",
              "cognitive_level": "VD",
              "details": "Tests contrast (despite, although, in spite of), condition (unless, provided that), addition (in addition to, furthermore), and cause/effect (due to, since).",
              "examples": [
                "_____ severe weather conditions disrupted local transportation, all retail stores remained open.\n(A) Despite\n(B) Although\n(C) Because\n(D) In addition"
              ]
            },
            {
              "id": "toeic_part5_business_collocations",
              "name": "Advanced Business Collocations & Phrasal Verbs",
              "cognitive_level": "VDC",
              "details": "Tests natural enterprise collocations (e.g. conduct a feasibility study, meet stringent standards, adhere to regulations, express reservations).",
              "examples": [
                "The pharmaceutical company must adhere strictly to international safety _____ when conducting clinical trials.\n(A) standards\n(B) permissions\n(C) occasions\n(D) preferences"
              ]
            }
          ]
        }
      ]
    },
    {
      "name": "Part 6 - Text Completion (Cloze Passages)",
      "units": [
        {
          "name": "Enterprise Memos, Notices & Correspondence",
          "problem_types": [
            {
              "id": "toeic_part6_internal_memos",
              "name": "Internal Company Memos with Blanks [131]-[134]",
              "cognitive_level": "VD",
              "details": "A continuous business document (150-250 words) with 4 numbered blanks [131], [132], [133], [134] testing vocabulary, connectors, verb forms, and 1 whole-sentence insertion.",
              "examples": [
                "Questions 131-134 refer to the following internal memorandum."
              ]
            },
            {
              "id": "toeic_part6_customer_advisories",
              "name": "Customer Advisories & Product Launch Letters",
              "cognitive_level": "VD",
              "details": "Formal letters and customer notices regarding service upgrades, policy modifications, or warranty terms.",
              "examples": [
                "Questions 135-138 refer to the following email advisory."
              ]
            }
          ]
        }
      ]
    },
    {
      "name": "Part 7 - Reading Comprehension (Single, Double & Triple Passages)",
      "units": [
        {
          "name": "Single Passages, Invoices & Chat Chains",
          "problem_types": [
            {
              "id": "toeic_part7_chat_chains",
              "name": "Instant Messaging & SMS Chat Chains",
              "cognitive_level": "VD",
              "details": "Multi-speaker instant messaging transcript with timestamps ([9:02 AM] Person: ...), followed by 2-3 questions.",
              "examples": [
                "Questions 158-160 refer to the following text message chain."
              ]
            },
            {
              "id": "toeic_part7_invoices_tables",
              "name": "Invoices, Receipts, Price Schedules & Financial Tables",
              "cognitive_level": "VD",
              "details": "Structured invoice, billing summary, or shipping manifest with column headers, itemized charges, discounts, and customer notes, followed by 2-3 questions.",
              "examples": [
                "Questions 150-152 refer to the following invoice."
              ]
            },
            {
              "id": "toeic_part7_sentence_placement",
              "name": "Article / Press Release with Sentence Placement [1]-[4]",
              "cognitive_level": "VDC",
              "details": "A formal article containing markers [1], [2], [3], [4], concluding with a sentence-insertion question.",
              "examples": [
                "Questions 161-164 refer to the following press release.\n164. In which of the positions marked [1], [2], [3], and [4] does the following sentence best belong?"
              ]
            }
          ]
        },
        {
          "name": "Multi-Passages (Double & Triple)",
          "problem_types": [
            {
              "id": "toeic_part7_double_passages",
              "name": "Double Passages (Job Ad + Application / Invoice + Email)",
              "cognitive_level": "VDC",
              "details": "Two interconnected documents (e.g. [Document 1: Job Posting] and [Document 2: Email Cover Letter]) followed by 5 cross-referencing questions.",
              "examples": [
                "Questions 176-180 refer to the following job advertisement and email."
              ]
            },
            {
              "id": "toeic_part7_triple_passages",
              "name": "Triple Passages (Agenda + Email + Feedback / Brochure + Booking + Invoice)",
              "cognitive_level": "VDC",
              "details": "Three interconnected documents (e.g. [Passage 1: Conference Program], [Passage 2: Email Inquiry], [Passage 3: Registration Confirmation]) followed by 5 synthesized comprehension questions.",
              "examples": [
                "Questions 186-190 refer to the following schedule, email, and confirmation notice."
              ]
            }
          ]
        }
      ]
    }
  ]
}

curriculum_literature_12 = {
  "subject": "literature",
  "grade": 12,
  "chapters": [
    {
      "name": "Phần I. Đọc hiểu văn bản (Chương trình 2018 & HSGQG)",
      "units": [
        {
          "name": "Đọc hiểu Thơ, Kịch và Văn bản Đa phương thức",
          "problem_types": [
            {
              "id": "reading_comprehension_literature_poetry",
              "name": "Đọc hiểu thơ hiện đại nhiều khổ (Multi-Stanza Poetry)",
              "cognitive_level": "NB_TH",
              "details": "Đoạn trích thơ hiện đại ngoài SGK (3-5 khổ thơ) có ngắt dòng, xuất xứ và chú thích tác giả/hoàn cảnh sáng tác nằm trọn vẹn trong <stimulus>. Gồm 5 câu hỏi tự luận.",
              "examples": [
                "Xác định thể thơ và mạch cảm xúc của bài thơ.",
                "Chỉ ra và phân tích tác dụng của biện pháp tu từ trong khổ thơ thứ hai."
              ]
            },
            {
              "id": "reading_comprehension_literature_drama",
              "name": "Đọc hiểu kịch bản sân khấu và đối thoại kịch (Drama & Script)",
              "cognitive_level": "VD",
              "details": "Trích đoạn kịch bản có tên nhân vật viết hoa kèm chỉ dẫn sân khấu in nghiêng trong ngoặc đơn, phân tích xung đột kịch và tính cách nhân vật.",
              "examples": [
                "Phân tích xung đột nội tâm của nhân vật qua các lượt thoại trong đoạn trích."
              ]
            },
            {
              "id": "reading_comprehension_literature_multimodal",
              "name": "Đọc hiểu văn bản thông tin & báo chí đa phương thức",
              "cognitive_level": "VD",
              "details": "Bài báo khoa học/xã hội có số liệu %, tiểu mục đánh số, trích dẫn chuyên gia. Kiểm tra phân biệt thông tin khách quan (facts) và ý kiến chủ quan (opinions).",
              "examples": [
                "Phân biệt thông tin khách quan và ý kiến đánh giá của tác giả trong đoạn trích."
              ]
            },
            {
              "id": "reading_comprehension_literature_dual",
              "name": "Đọc hiểu văn bản kép / song hành (Dual Paired Texts)",
              "cognitive_level": "VDC",
              "details": "Hai văn bản ngắn (Văn bản 1: Thơ/Văn xuôi; Văn bản 2: Tiểu luận phê bình) về cùng một đề tài, yêu cầu so sánh điểm tương đồng và khác biệt.",
              "examples": [
                "Chỉ ra điểm tương đồng và khác biệt về quan niệm nhân sinh giữa hai văn bản."
              ]
            },
            {
              "id": "reading_comprehension_literature",
              "name": "Đọc hiểu truyện ngắn, tản văn, tùy bút ngoài SGK",
              "cognitive_level": "NB_TH",
              "details": "Đoạn trích truyện ngắn/tản văn hiện đại ngoài SGK kèm chú thích nguồn và 5 câu hỏi tự luận nhận biết, thông hiểu, vận dụng.",
              "examples": [
                "Xác định ngôi kể và điểm nhìn trần thuật trong đoạn trích."
              ]
            }
          ]
        }
      ]
    },
    {
      "name": "Phần II. Viết / Làm văn (Chương trình 2018 & HSGQG)",
      "units": [
        {
          "name": "Nghị luận Xã hội (200 chữ & 8.0 điểm HSG)",
          "problem_types": [
            {
              "id": "social_argumentation_paragraph",
              "name": "Đoạn văn nghị luận xã hội 200 chữ",
              "cognitive_level": "VD",
              "details": "Viết đoạn văn khoảng 200 chữ bàn về một khía cạnh tư tưởng, đạo lý, lối sống được gợi mở từ ngữ liệu Đọc hiểu.",
              "examples": [
                "Từ ngữ liệu Đọc hiểu, anh/chị hãy viết đoạn văn (khoảng 200 chữ) trình bày suy nghĩ về..."
              ]
            },
            {
              "id": "social_applied_writing",
              "name": "Viết ứng dụng: Thư ngỏ, Kiến nghị, Book Review, Bài phát biểu",
              "cognitive_level": "VD",
              "details": "Yêu cầu viết một bức thư ngỏ, bản đề xuất giải pháp, bài giới thiệu sách hoặc bài phát biểu (khoảng 200 chữ).",
              "examples": [
                "Hãy viết một bức thư ngỏ (khoảng 200 chữ) gửi các bạn trẻ về..."
              ]
            },
            {
              "id": "social_philosophical_dialogue_hsg",
              "name": "Nghị luận xã hội đối thoại triết học & nghịch lý (HSGQG 8.0 điểm)",
              "cognitive_level": "VDC",
              "details": "Đoạn đối thoại tư tưởng (Triết gia & Chàng thanh niên) hoặc mệnh đề nghịch lý kèm cước chú học thuật. Yêu cầu bàn về cặp phạm trù triết học.",
              "examples": [
                "Từ đoạn đối thoại trên, anh/chị hãy viết bài văn nghị luận bàn về mối quan hệ giữa tự lập, tình yêu và cảm thức cộng đồng."
              ]
            }
          ]
        },
        {
          "name": "Nghị luận Văn học (600 chữ & 12.0 điểm HSG)",
          "problem_types": [
            {
              "id": "literary_analysis_essay_600",
              "name": "Bài văn nghị luận văn học 600 chữ phân tích tác phẩm",
              "cognitive_level": "VD",
              "details": "Viết bài văn khoảng 600 chữ phân tích một nét đặc sắc về nội dung hoặc nghệ thuật của tác phẩm trong phần Đọc hiểu.",
              "examples": [
                "Viết bài văn nghị luận (khoảng 600 chữ) phân tích hình tượng..."
              ]
            },
            {
              "id": "literary_comparative_essay_600",
              "name": "Bài văn nghị luận so sánh 2 tác phẩm / 2 đoạn thơ (600 chữ)",
              "cognitive_level": "VDC",
              "details": "Cho 2 đoạn trích thơ/văn xuôi của 2 tác giả kèm chú thích tác giả. Yêu cầu so sánh điểm tương đồng và nét độc đáo riêng.",
              "examples": [
                "Cho hai đoạn thơ sau:... Anh/chị hãy viết bài văn (khoảng 600 chữ) so sánh vẻ đẹp..."
              ]
            },
            {
              "id": "literary_reception_theory_hsg",
              "name": "Nghị luận văn học lý luận tiếp nhận & bản thể luận (HSGQG 12.0 điểm)",
              "cognitive_level": "VDC",
              "details": "Nhận định lý luận văn học chuyên sâu của học giả quốc tế/Việt Nam. Yêu cầu bàn luận bằng trải nghiệm đọc văn học phong phú.",
              "examples": [
                "Nhận định của C.S. Lewis: 'Chỉ khi tôi không đủ cho tôi...' Bằng trải nghiệm văn học, anh/chị hãy bàn luận vấn đề đó."
              ]
            }
          ]
        }
      ]
    }
  ]
}

# Write TOEIC curriculum
out_toeic = Path('output/curriculum/toeic_12.json')
out_toeic.parent.mkdir(parents=True, exist_ok=True)
with open(out_toeic, 'w', encoding='utf-8') as f:
    json.dump(curriculum_toeic, f, ensure_ascii=False, indent=2)
print(f"Wrote {out_toeic} ({len(curriculum_toeic['chapters'])} chapters)")

# Write Literature 12 curriculum
out_lit = Path('output/curriculum/literature_12.json')
out_lit.parent.mkdir(parents=True, exist_ok=True)
with open(out_lit, 'w', encoding='utf-8') as f:
    json.dump(curriculum_literature_12, f, ensure_ascii=False, indent=2)
print(f"Wrote {out_lit} ({len(curriculum_literature_12['chapters'])} chapters)")

# Also write Literature 11 and Literature 10 copies for complete grade coverage
for g in [10, 11]:
    lit_copy = dict(curriculum_literature_12)
    lit_copy['grade'] = g
    p = Path(f'output/curriculum/literature_{g}.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(lit_copy, f, ensure_ascii=False, indent=2)
    print(f"Wrote {p}")

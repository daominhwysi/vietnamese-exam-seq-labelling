import unittest
from src.generation.reconstructor import (
    reconstruct_question,
    reconstruct_exam,
    ReconstructorConfig,
    generate_ordering_choices
)

class TestReconstructor(unittest.TestCase):
    def test_standard_multiple_choice(self):
        q_data = {
            "is_group": False,
            "stem": "Họ nguyên hàm của hàm số $f(x) = x^2$ là",
            "options": [
                "\\frac{1}{3}x^3 + C.",
                "2x^3 + C.",
                "3x^3 + C.",
                "\\frac{1}{2}x^3 + C."
            ],
            "question_type": "multiple_choice",
            "subject": "chemistry",
            "grade": 10,
            "difficulty": "comprehend"
        }
        
        # Test with a specific configuration
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            option_prefix_style="capital_dot",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        
        enriched = reconstruct_question(q_data, config, start_q_num=5)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        expected_text = (
            "Câu 5: Họ nguyên hàm của hàm số $f(x) = x^2$ là\n"
            "A. \\frac{1}{3}x^3 + C.\n"
            "B. 2x^3 + C.\n"
            "C. 3x^3 + C.\n"
            "D. \\frac{1}{2}x^3 + C."
        )
        
        self.assertEqual(raw_text, expected_text)
        
        # Verify spans
        for span in spans:
            start = span["start"]
            end = span["end"]
            label = span["label"]
            text = span["text"]
            
            # Check slice matches
            self.assertEqual(raw_text[start:end], text)
            
            if label == "question_label":
                self.assertEqual(text, "Câu 5: ")
            elif label == "stem":
                self.assertEqual(text, "Họ nguyên hàm của hàm số $f(x) = x^2$ là")
            elif label == "option_label":
                self.assertIn(text, ["A. ", "B. ", "C. ", "D. "])
            elif label == "option_text":
                self.assertIn(text, q_data["options"])

    def test_standard_true_false(self):
        q_data = {
            "is_group": False,
            "stem": "Một vật chuyển động thẳng...",
            "options": [
                "Phương trình đường thẳng AB là...",
                "Góc giữa hai vectơ..."
            ],
            "question_type": "true_false",
            "subject": "physics",
            "grade": 11,
            "difficulty": "comprehend"
        }
        
        config = ReconstructorConfig(
            question_prefix_template="**Câu {num}:** ",
            option_prefix_style="bold_lowercase_paren",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        
        enriched = reconstruct_question(q_data, config, start_q_num=2)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        expected_text = (
            "**Câu 2:** Một vật chuyển động thẳng...\n"
            "**a)** Phương trình đường thẳng AB là...\n"
            "**b)** Góc giữa hai vectơ..."
        )
        
        self.assertEqual(raw_text, expected_text)
        
        # Check slice matches
        for span in spans:
            self.assertEqual(raw_text[span["start"]:span["end"]], span["text"])

    def test_short_answer(self):
        q_data = {
            "is_group": False,
            "stem": "Tính nguyên hàm...",
            "options": [],
            "question_type": "short_answer",
            "subject": "math_algebra",
            "grade": 12,
            "difficulty": "comprehend"
        }
        
        config = ReconstructorConfig(
            question_prefix_template="Q{num}. ",
            randomize_q_num=False
        )
        
        enriched = reconstruct_question(q_data, config, start_q_num=10)
        self.assertEqual(enriched["raw_text"], "Q10. Tính nguyên hàm...")

    def test_ordering_question(self):
        q_data = {
            "is_group": False,
            "stem": "Sắp xếp thứ tự các bước:",
            "options": ["Bước 1", "Bước 2", "Bước 3"],
            "question_type": "ordering",
            "subject": "chemistry",
            "grade": 10,
            "difficulty": "comprehend"
        }
        
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            option_prefix_style="capital_dot",
            ordering_item_label_style="char",
            ordering_item_prefix_template="* {label}. ",
            ordering_choice_separator=" – ",
            seed="my_stable_seed",
            randomize_q_num=False
        )
        
        enriched = reconstruct_question(q_data, config, start_q_num=16)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        # Verify ordering item text and prefixes are labeled as stem
        stem_segments = [s["text"] for s in spans if s["label"] == "stem"]
        
        self.assertIn("Sắp xếp thứ tự các bước:", stem_segments)
        self.assertIn("Bước 1", stem_segments)
        self.assertIn("Bước 2", stem_segments)
        self.assertIn("Bước 3", stem_segments)
        self.assertIn("* a. ", stem_segments)
        self.assertIn("* b. ", stem_segments)
        self.assertIn("* c. ", stem_segments)
        
        # Verify multiple choice options representing permutations are added
        option_texts = [s["text"] for s in spans if s["label"] == "option_text"]
        option_labels = [s["text"] for s in spans if s["label"] == "option_label"]
        
        self.assertEqual(len(option_texts), 4)
        self.assertEqual(len(option_labels), 4)
        
        # The correct option should be present: "a – b – c"
        self.assertIn("a – b – c", option_texts)
        
        # Each permutation option should contain a mix of a, b, c separated by ' – '
        for opt in option_texts:
            self.assertEqual(len(opt.split(" – ")), 3)
            
        # Check that answer is resolved to A, B, C, or D
        self.assertIn("answer", enriched)
        self.assertIn(enriched["answer"], ["A", "B", "C", "D"])
        
        # Verify the choice text corresponding to answer equals correct sequence "a – b – c"
        correct_index = ["A", "B", "C", "D"].index(enriched["answer"])
        self.assertEqual(option_texts[correct_index], "a – b – c")
        
        # Ensure spans are valid slices of raw_text
        for span in spans:
            self.assertEqual(raw_text[span["start"]:span["end"]], span["text"])

    def test_group_question(self):
        q_data = {
            "is_group": True,
            "context": "Đây là ngữ cảnh chung.",
            "questions": [
                {
                    "stem": "Câu hỏi 1",
                    "options": ["A1", "B1"]
                },
                {
                    "stem": "Câu hỏi 2",
                    "options": []
                }
            ],
            "question_type": "group_multiple_choice",
            "subject": "history",
            "grade": 12,
            "difficulty": "comprehend"
        }
        
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            option_prefix_style="capital_dot",
            separator_context_questions="\n",
            separator_questions="\n",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        expected_text = (
            "Đây là ngữ cảnh chung.\n"
            "Câu 1: Câu hỏi 1\n"
            "A. A1\n"
            "B. B1\n"
            "Câu 2: Câu hỏi 2"
        )
        
        self.assertEqual(raw_text, expected_text)
        
        # Check slice matches
        for span in spans:
            self.assertEqual(raw_text[span["start"]:span["end"]], span["text"])

    def test_deterministic_seeding(self):
        q_data = {
            "is_group": False,
            "stem": "Hỏi han gì đó...",
            "options": ["O1", "O2", "O3", "O4"],
            "question_type": "multiple_choice",
            "subject": "geography",
            "grade": 8,
            "difficulty": "recognize"
        }
        
        # With default randomized config, two calls on the same question dict (no config passed)
        # should produce identical output because seed defaults to the question content.
        enriched1 = reconstruct_question(q_data)
        enriched2 = reconstruct_question(q_data)
        
        self.assertEqual(enriched1["raw_text"], enriched2["raw_text"])
        self.assertEqual(enriched1["spans"], enriched2["spans"])

    def test_reconstruct_literature_no_options(self):
        # Literature question has no options list
        q_data = {
            "is_group": False,
            "stem": "Đề bài tự luận Ngữ văn...",
            "question_type": "short_answer",
            "subject": "literature",
            "grade": 12,
            "difficulty": "high_application"
        }
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            randomize_q_num=False
        )
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        self.assertEqual(enriched["raw_text"], "Câu 1: Đề bài tự luận Ngữ văn...")
        
        spans = enriched["spans"]
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["label"], "question_label")
        self.assertEqual(spans[1]["label"], "stem")

    def test_reconstruct_english_cloze_blank_normalization(self):
        # English cloze passages should normalize spaces/underscores/dots into [BLANK]
        q_data = {
            "is_group": True,
            "context": "This is a passage with (1)_____ word and (2).... sentence gap.",
            "questions": [
                {
                    "stem": "Question 1: What does it mean _____?",
                    "options": ["A", "B"]
                },
                {
                    "stem": "Question 2",
                    "options": ["C", "D"]
                }
            ],
            "question_type": "group_multiple_choice",
            "subject": "english",
            "grade": 10,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(
            question_prefix_template="Question {num}: ",
            option_prefix_style="capital_dot",
            separator_context_questions="\n",
            separator_questions="\n",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        raw_text = enriched["raw_text"]
        
        # Check that context has been normalized
        self.assertTrue(raw_text.startswith("This is a passage with (1) <blank /> word and (2) <blank /> sentence gap."))
        # Check that group sub-question stem has been normalized
        self.assertIn("Question 1: What does it mean <blank/>?", raw_text)

    def test_reconstruct_english_standard_blank_normalization(self):
        q_data = {
            "is_group": False,
            "stem": "He ___ a book yesterday.",
            "options": ["read", "reads", "reading", "has read"],
            "question_type": "multiple_choice",
            "subject": "english",
            "grade": 12,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(
            question_prefix_template="Question {num}: ",
            option_prefix_style="capital_dot",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        raw_text = enriched["raw_text"]
        
        self.assertEqual(raw_text, "Question 1: He <blank/> a book yesterday.\nA. read\nB. reads\nC. reading\nD. has read")

    def test_reconstruct_entire_exam(self):
        from src.generation.reconstructor import reconstruct_exam
        exam_data = {
            "exam_id": "test_exam_123",
            "subject": "physics",
            "grade": 11,
            "sections": {
                "PHẦN I. Trắc nghiệm": [
                    {
                        "is_group": False,
                        "stem": "Câu hỏi vật lý.",
                        "options": ["A1", "B1"],
                        "answer": "A",
                        "explanation": "Vì A1 đúng.",
                        "question_type": "multiple_choice",
                        "subject": "physics",
                        "grade": 11
                    }
                ]
            }
        }
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            option_prefix_style="capital_dot",
            separator_questions="\n",
            randomize_q_num=False,
            explanation_layout="separated"
        )
        enriched = reconstruct_exam(exam_data, config)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        self.assertIn("ĐỀ THI MÔN: VẬT LÝ - LỚP 11", raw_text)
        self.assertIn("PHẦN I. Trắc nghiệm", raw_text)
        self.assertIn("Câu 1: Câu hỏi vật lý.", raw_text)
        self.assertIn("A. A1", raw_text)
        
        # Verify answer key table is reconstructed
        self.assertIn("ĐÁP ÁN THAM KHẢO", raw_text)
        self.assertIn("| Câu | 1 |", raw_text)
        self.assertIn("| Đáp án | A |", raw_text)
        
        # Verify explanation text is reconstructed
        self.assertIn("LỜI GIẢI THAM KHẢO", raw_text)
        self.assertIn("Câu 1: A", raw_text)
        self.assertIn("Vì A1 đúng.", raw_text)
        
        # Verify spans align correctly
        for span in spans:
            self.assertEqual(raw_text[span["start"]:span["end"]], span["text"])
            
        # Verify section titles and explanation are tagged
        section_spans = [s for s in spans if s["label"] == "section"]
        explanation_spans = [s for s in spans if s["label"] == "explanation"]
        question_label_spans = [s for s in spans if s["label"] == "question_label"]
        option_label_spans = [s for s in spans if s["label"] == "option_label"]
        
        self.assertTrue(any("ĐÁP ÁN THAM KHẢO" in s["text"] for s in section_spans))
        self.assertTrue(any("LỜI GIẢI THAM KHẢO" in s["text"] for s in section_spans))
        self.assertTrue(any("Vì A1 đúng." in s["text"] for s in explanation_spans))
        
        # Verify the explanation prefix "Câu 1:" and answer "A**" are labeled correctly
        self.assertTrue(any("Câu 1:" in s["text"] for s in question_label_spans))
        self.assertTrue(any("A**" in s["text"] for s in explanation_spans))

    def test_inject_vietnamese_typos(self):
        from src.generation.reconstructor import inject_vietnamese_typos, get_stable_random
        rng = get_stable_random("test_seed")
        text = "hàm số đồng biến trên khoảng"
        typo_text = inject_vietnamese_typos(text, typo_rate=1.0, rng=rng)
        self.assertNotEqual(text, typo_text)

    def test_randomize_blank_tokens(self):
        from src.generation.reconstructor import randomize_blank_tokens, get_stable_random
        rng = get_stable_random("test_seed_blank")
        text = "This is a <blank/> space."
        randomized = randomize_blank_tokens(text, rng)
        self.assertNotIn("<blank/>", randomized)

    def test_process_latex_variations(self):
        from src.generation.reconstructor import process_latex_variations, get_stable_random
        rng = get_stable_random("test_seed_latex")
        text = "Hàm số $y = x^2$ liên tục."
        
        masked = process_latex_variations(text, "[LATEX]", mask_prob=1.0, rng=rng)
        self.assertIn("[LATEX]", masked)
        self.assertNotIn("y = x^2", masked)
        
        raw_var = process_latex_variations(text, "[LATEX]", mask_prob=0.0, rng=rng)
        self.assertTrue(any(x in raw_var for x in ["$y = x^2$", "\\( y = x^2 \\)", "y = x^2"]))

    def test_option_dropping(self):
        q_data = {
            "is_group": False,
            "stem": "Câu hỏi test option drop.",
            "options": ["A1", "B1", "C1", "D1"],
            "question_type": "multiple_choice",
            "subject": "chemistry",
            "grade": 10,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(option_drop_prob=1.0, seed="drop_seed")
        enriched = reconstruct_question(q_data, config)
        options_in_spans = [s["text"] for s in enriched["spans"] if s["label"] == "option_text"]
        self.assertTrue(len(options_in_spans) < 4)

    def test_option_permutation(self):
        q_data = {
            "is_group": False,
            "stem": "Câu hỏi trắc nghiệm.",
            "options": ["Đáp án A (Đúng)", "Đáp án B", "Đáp án C", "Đáp án D"],
            "answer": "A",
            "question_type": "multiple_choice",
            "subject": "chemistry",
            "grade": 10,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(enable_permutations=True, seed="perm_seed")
        
        shuffled = False
        for i in range(10):
            config.seed = f"seed_{i}"
            enriched = reconstruct_question(q_data, config)
            options_in_spans = [s["text"] for s in enriched["spans"] if s["label"] == "option_text"]
            if options_in_spans != q_data["options"]:
                shuffled = True
                new_ans_letter = enriched.get("answer", "")
                opt_letters = ["A", "B", "C", "D"]
                if new_ans_letter in opt_letters:
                    new_idx = opt_letters.index(new_ans_letter)
                    self.assertEqual(options_in_spans[new_idx], "Đáp án A (Đúng)")
                break
                
        self.assertTrue(shuffled)

    def test_exam_explanation_layouts(self):
        exam_data = {
            "exam_id": "test_layout_exam",
            "subject": "physics",
            "grade": 11,
            "sections": {
                "PHẦN I": [
                    {
                        "is_group": False,
                        "stem": "Câu hỏi vật lý.",
                        "options": ["A1", "B1"],
                        "answer": "A",
                        "explanation": "Giải thích chi tiết.",
                        "question_type": "multiple_choice",
                        "subject": "physics",
                        "grade": 11
                    }
                ]
            }
        }
        
        # Test table_only
        config = ReconstructorConfig(explanation_layout="table_only", randomize_q_num=False)
        enriched = reconstruct_exam(exam_data, config)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        self.assertIn("ĐÁP ÁN THAM KHẢO", raw_text)
        self.assertNotIn("LỜI GIẢI THAM KHẢO", raw_text)
        self.assertNotIn("Giải thích chi tiết.", raw_text)
        self.assertFalse(any(s["label"] == "explanation" for s in spans))
        
        # Test interleaved
        config = ReconstructorConfig(explanation_layout="interleaved", randomize_q_num=False)
        enriched = reconstruct_exam(exam_data, config)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        self.assertIn("ĐÁP ÁN THAM KHẢO", raw_text)
        self.assertNotIn("LỜI GIẢI THAM KHẢO", raw_text)
        self.assertIn("Giải thích chi tiết.", raw_text)
        self.assertTrue(any(s["label"] == "explanation" for s in spans))
        # Ensure the explanation is placed near the question, i.e., before the answer table
        ans_table_idx = raw_text.find("ĐÁP ÁN THAM KHẢO")
        expl_idx = raw_text.find("Giải thích chi tiết.")
        self.assertTrue(0 <= expl_idx < ans_table_idx)
        
        # Test separated
        config = ReconstructorConfig(explanation_layout="separated", randomize_q_num=False)
        enriched = reconstruct_exam(exam_data, config)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        self.assertIn("ĐÁP ÁN THAM KHẢO", raw_text)
        self.assertIn("LỜI GIẢI THAM KHẢO", raw_text)
        self.assertIn("Giải thích chi tiết.", raw_text)
        self.assertTrue(any(s["label"] == "explanation" for s in spans))
        # Ensure the explanation is placed after LỜI GIẢI THAM KHẢO
        expl_title_idx = raw_text.find("LỜI GIẢI THAM KHẢO")
        expl_idx = raw_text.find("Giải thích chi tiết.")
        self.assertTrue(expl_title_idx < expl_idx)

    def test_answer_table_formats(self):
        from src.generation.reconstructor import reconstruct_exam
        exam_data = {
            "exam_id": "test_table_exam",
            "subject": "math_algebra",
            "grade": 12,
            "sections": {
                "PHẦN I": [
                    {
                        "is_group": False,
                        "stem": "Câu hỏi 1.",
                        "options": ["A1", "B1"],
                        "answer": "A",
                        "explanation": "",
                        "question_type": "multiple_choice",
                        "subject": "math_algebra",
                        "grade": 12
                    },
                    {
                        "is_group": False,
                        "stem": "Câu hỏi 2.",
                        "options": ["A2", "B2"],
                        "answer": "B",
                        "explanation": "",
                        "question_type": "multiple_choice",
                        "subject": "math_algebra",
                        "grade": 12
                    }
                ]
            }
        }
        
        # 1. Test markdown vertical
        config = ReconstructorConfig(
            answer_table_format="md",
            answer_table_direction="vertical",
            randomize_q_num=False
        )
        enriched = reconstruct_exam(exam_data, config)
        self.assertIn("| Câu | Đáp án |", enriched["raw_text"])
        self.assertIn("| 1 | A |", enriched["raw_text"])
        self.assertIn("| 2 | B |", enriched["raw_text"])

        # 2. Test HTML horizontal
        config = ReconstructorConfig(
            answer_table_format="html",
            answer_table_direction="horizontal",
            answer_table_chunk_size=10,
            randomize_q_num=False
        )
        enriched = reconstruct_exam(exam_data, config)
        self.assertIn("<table>", enriched["raw_text"])
        self.assertIn("<th>1</th>", enriched["raw_text"])
        self.assertIn("<td>A</td>", enriched["raw_text"])

        # 3. Test HTML vertical
        config = ReconstructorConfig(
            answer_table_format="html",
            answer_table_direction="vertical",
            randomize_q_num=False
        )
        enriched = reconstruct_exam(exam_data, config)
        self.assertIn("<th>Câu</th>", enriched["raw_text"])
        self.assertIn("<td>1</td>", enriched["raw_text"])
        self.assertIn("<td>A</td>", enriched["raw_text"])

        # 4. Test CSV horizontal
        config = ReconstructorConfig(
            answer_table_format="csv",
            answer_table_direction="horizontal",
            answer_table_chunk_size=10,
            randomize_q_num=False
        )
        enriched = reconstruct_exam(exam_data, config)
        self.assertIn("Câu,1,2", enriched["raw_text"])
        self.assertIn("Đáp án,A,B", enriched["raw_text"])

        # 5. Test CSV vertical
        config = ReconstructorConfig(
            answer_table_format="csv",
            answer_table_direction="vertical",
            randomize_q_num=False
        )
        enriched = reconstruct_exam(exam_data, config)
        self.assertIn("Câu,Đáp án", enriched["raw_text"])
        self.assertIn("1,A", enriched["raw_text"])
        self.assertIn("2,B", enriched["raw_text"])

        # 6. Test Random format and direction does not crash
        config = ReconstructorConfig(
            answer_table_format="random",
            answer_table_direction="random",
            randomize_q_num=False
        )
        enriched = reconstruct_exam(exam_data, config)
        self.assertIn("ĐÁP ÁN THAM KHẢO", enriched["raw_text"])

    def test_section_title_paraphrasing(self):
        from src.generation.reconstructor import reconstruct_exam
        exam_data = {
            "exam_id": "test_section_exam",
            "subject": "physics",
            "grade": 11,
            "sections": {
                "PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn": [
                    {
                        "is_group": False,
                        "stem": "Câu hỏi.",
                        "options": ["A1", "B1"],
                        "answer": "A",
                        "explanation": "",
                        "question_type": "multiple_choice",
                        "subject": "physics",
                        "grade": 11
                    }
                ]
            }
        }
        
        # Test with paraphrase_section_titles = True
        config = ReconstructorConfig(
            paraphrase_section_titles=True,
            randomize_q_num=False,
            seed="my_stable_seed"
        )
        enriched = reconstruct_exam(exam_data, config)
        raw_text = enriched["raw_text"]
        
        # Check that the section title was changed from the original
        if "PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn\n" in raw_text:
            self.fail(f"Title was not paraphrased: {raw_text}")
        if not ("PHẦN I" in raw_text or "Phần I" in raw_text or "Phần 1" in raw_text or "Phần thứ nhất" in raw_text):
            self.fail(f"Expected paraphrased title, got: {raw_text!r}")

    def test_bold_option_label(self):
        q_data = {
            "is_group": False,
            "stem": "Bold option label test question.",
            "options": ["Option 1", "Option 2"],
            "question_type": "multiple_choice",
            "subject": "chemistry",
            "grade": 10,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(
            question_prefix_template="Câu {num}: ",
            option_prefix_style="bold_capital_dot",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        # Expect raw text to contain **A.** Option 1
        self.assertIn("**A.** Option 1", raw_text)
        
        # Check that span for option_label includes "**A.**"
        opt_spans = [s for s in spans if s["label"] == "option_label"]
        self.assertTrue(len(opt_spans) >= 1)
        self.assertEqual(opt_spans[0]["text"], "**A.**")

    def test_bold_question_label(self):
        q_data = {
            "is_group": False,
            "stem": "Bold question label test.",
            "options": ["Option 1", "Option 2"],
            "question_type": "multiple_choice",
            "subject": "chemistry",
            "grade": 10,
            "difficulty": "comprehend"
        }
        config = ReconstructorConfig(
            question_prefix_template="**Câu {num}:** ",
            option_prefix_style="capital_dot",
            separator_stem_options="\n",
            separator_options="\n",
            randomize_q_num=False
        )
        enriched = reconstruct_question(q_data, config, start_q_num=1)
        raw_text = enriched["raw_text"]
        spans = enriched["spans"]
        
        # Expect raw text to contain **Câu 1:** Bold question label test.
        self.assertIn("**Câu 1:** Bold question label test.", raw_text)
        
        # Check that span for question_label includes "**Câu 1:**"
        q_spans = [s for s in spans if s["label"] == "question_label"]
        self.assertTrue(len(q_spans) >= 1)
        self.assertEqual(q_spans[0]["text"], "**Câu 1:**")

if __name__ == '__main__':
    unittest.main()

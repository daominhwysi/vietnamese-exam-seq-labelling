import unittest
from sequence_labelling_data_generator.reconstructor import (
    reconstruct_question,
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
        
        # Verify ordering item text has spans labeled correctly
        ordering_item_texts = [s["text"] for s in spans if s["label"] == "ordering_item_text"]
        ordering_item_labels = [s["text"] for s in spans if s["label"] == "ordering_item_label"]
        
        self.assertEqual(ordering_item_texts, ["Bước 1", "Bước 2", "Bước 3"])
        self.assertEqual(ordering_item_labels, ["* a. ", "* b. ", "* c. "])
        
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

if __name__ == '__main__':
    unittest.main()

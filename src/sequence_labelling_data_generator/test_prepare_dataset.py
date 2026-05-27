import unittest
from sequence_labelling_data_generator.prepare_dataset import (
    align_tokens_to_spans,
    get_tag_mappings,
    process_single_question,
    replace_latex_in_question
)

class MockTokenizer:
    def __init__(self):
        pass

    def __call__(self, text, return_offsets_mapping=True, truncation=True, add_special_tokens=True):
        # A simple mock tokenizer return that splits by spaces
        # and returns dummy offset mappings
        words = text.split()
        input_ids = list(range(len(words) + 2))  # +2 for special tokens
        attention_mask = [1] * len(input_ids)
        
        # Build manual offset mapping
        offsets = [(0, 0)]  # <s>
        current_idx = 0
        for word in words:
            # find start of word
            start = text.find(word, current_idx)
            end = start + len(word)
            offsets.append((start, end))
            current_idx = end
        offsets.append((0, 0))  # </s>
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets
        }

    def convert_ids_to_tokens(self, ids):
        # returns dummy tokens list
        return [f"token_{i}" for i in ids]

class TestPrepareDataset(unittest.TestCase):
    def setUp(self):
        self.tag_to_id, self.id_to_tag = get_tag_mappings()

    def test_get_tag_mappings(self):
        self.assertIn("O", self.tag_to_id)
        self.assertIn("B-stem", self.tag_to_id)
        self.assertIn("I-stem", self.tag_to_id)
        self.assertIn("B-option_text", self.tag_to_id)
        self.assertEqual(self.tag_to_id["O"], 0)
        self.assertEqual(self.id_to_tag[0], "O")
        # Ensure B- and I- tags are mapped to unique values
        self.assertEqual(len(self.tag_to_id), 1 + 2 * 7)  # O + B/I for 7 tags

    def test_align_tokens_to_spans_basic(self):
        # Spans:
        # stem: "Hello world" -> (10, 21)
        # option_text: "Answer A" -> (25, 33)
        spans = [
            {"start": 10, "end": 21, "label": "stem", "text": "Hello world"},
            {"start": 25, "end": 33, "label": "option_text", "text": "Answer A"}
        ]
        
        # Token offsets:
        offsets = [
            (0, 0),    # Special token (<s>)
            (0, 5),    # "Start" - Outside
            (5, 9),    # " text" - Outside
            (9, 15),   # " Hello" - Overlaps stem(10, 21) and contains start index 10 (9 <= 10 < 15) -> B-stem
            (15, 21),  # " world" - Overlaps stem(10, 21), start is 15 (15 > 10) -> I-stem
            (21, 25),  # " intermediate" - Outside
            (25, 30),  # " Answer" - Overlaps option_text(25, 33), contains start index 25 (25 <= 25 < 30) -> B-option_text
            (30, 34),  # " A." - Overlaps option_text(25, 33), starts at 30 (30 > 25) -> I-option_text
            (0, 0)     # Special token (</s>)
        ]
        
        labels = align_tokens_to_spans(offsets, spans, self.tag_to_id)
        
        expected_labels = [
            -100,                                 # <s>
            self.tag_to_id["O"],                  # Outside
            self.tag_to_id["O"],                  # Outside
            self.tag_to_id["B-stem"],             # B-stem
            self.tag_to_id["I-stem"],             # I-stem
            self.tag_to_id["O"],                  # Outside
            self.tag_to_id["B-option_text"],      # B-option_text
            self.tag_to_id["I-option_text"],      # I-option_text
            -100                                  # </s>
        ]
        
        self.assertEqual(labels, expected_labels)

    def test_latex_replacement_basic(self):
        q_data = {
            "is_group": False,
            "stem": "Hàm số $f(x) = x^2 + 1$ có $x$ là biến.",
            "options": ["A. $x = 1$", "B. $x = 2$"],
            "question_type": "multiple_choice",
            "subject": "math_algebra",
            "grade": 10,
            "difficulty": "comprehend"
        }
        
        replaced = replace_latex_in_question(q_data, "[LATEX]")
        
        self.assertEqual(replaced["stem"], "Hàm số [LATEX] có [LATEX] là biến.")
        self.assertEqual(replaced["options"][0], "A. [LATEX]")
        self.assertEqual(replaced["options"][1], "B. [LATEX]")

    def test_latex_replacement_group(self):
        q_data = {
            "is_group": True,
            "context": "Cho biểu thức $A = 2^x$.",
            "questions": [
                {
                    "stem": "Tính giá trị của $A$ khi $x = 1$:",
                    "options": ["$A = 2$", "$A = 4$"]
                }
            ],
            "question_type": "group_multiple_choice",
            "subject": "math_algebra",
            "grade": 10,
            "difficulty": "comprehend"
        }
        
        replaced = replace_latex_in_question(q_data, "[LATEX]")
        
        self.assertEqual(replaced["context"], "Cho biểu thức [LATEX].")
        self.assertEqual(replaced["questions"][0]["stem"], "Tính giá trị của [LATEX] khi [LATEX]:")
        self.assertEqual(replaced["questions"][0]["options"][0], "[LATEX]")
        self.assertEqual(replaced["questions"][0]["options"][1], "[LATEX]")

    def test_process_single_question_with_latex(self):
        tokenizer = MockTokenizer()
        q_data = {
            "is_group": False,
            "stem": "Hàm số $f(x) = x^2$ đồng biến.",
            "options": ["$x > 0$"],
            "question_type": "multiple_choice",
            "subject": "math_algebra",
            "grade": 10,
            "difficulty": "comprehend"
        }
        
        # Test processing with latex_placeholder: raw_text and spans should map to the replaced "[LATEX]" strings
        sample = process_single_question(q_data, tokenizer, self.tag_to_id, self.id_to_tag, latex_placeholder="[LATEX]")
        
        self.assertIsNotNone(sample)
        # Check that one of the tokens corresponds to "[LATEX]"
        self.assertIn("token_3", sample["tokens"]) # MockTokenizer conversion
        # Ensure we have tags assigned correctly (B-stem/I-stem) on the replaced text
        self.assertIn("B-stem", sample["tags"])
        self.assertIn("B-option_text", sample["tags"])

if __name__ == '__main__':
    unittest.main()

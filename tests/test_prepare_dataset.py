import unittest
from src.data.prepare import (
    align_tokens_to_spans,
    get_tag_mappings,
    process_single_question,
    replace_latex_in_question,
    resolve_stimulus_anchors,
    parse_xml_annotations,
    spans_to_xml
)

class MockTokenizer:
    def __init__(self):
        pass

    def add_special_tokens(self, *args, **kwargs):
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
        self.assertIn("B-section", self.tag_to_id)
        self.assertIn("I-section", self.tag_to_id)
        self.assertIn("B-explanation", self.tag_to_id)
        self.assertIn("I-explanation", self.tag_to_id)
        self.assertEqual(len(self.tag_to_id), 1 + 2 * 7)  # O + B/I for 7 tags = 15 classes

    def test_sanitize_nested_explanation_tags(self):
        from src.data.prepare import sanitize_nested_explanation_tags
        xml_in = "<explanation>Lời giải: <option_label>a.</option_label> <stem>Xét tam giác</stem> ABC.</explanation>"
        sanitized = sanitize_nested_explanation_tags(xml_in)
        self.assertEqual(sanitized, "<explanation>Lời giải: a. Xét tam giác ABC.</explanation>")

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
        
        raw_text = "Starttext Hello world  Answer A "
        labels = align_tokens_to_spans(offsets, spans, self.tag_to_id, raw_text)
        
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
        
    def test_resolve_stimulus_anchors(self):
        xml_input = (
            "<section># Section 1</section>\n"
            "<stimulus id=\"stim_1\" start_anchor=\"Read the following passage\" end_anchor=\"end of passage.\" />\n\n"
            "Read the following passage and answer the questions.\n"
            "This is paragraph one.\n"
            "This is paragraph two and the end of passage.\n\n"
            "<question_label>Question 1:</question_label> <stem>What is this about?</stem>"
        )
        resolved = resolve_stimulus_anchors(xml_input)
        self.assertNotIn("<stimulus id=", resolved)
        self.assertIn("<stimulus>Read the following passage and answer the questions.\nThis is paragraph one.\nThis is paragraph two and the end of passage.</stimulus>", resolved)

    def test_parse_xml_annotations_with_stimulus(self):
        tagged_text = (
            "<section># Section 1</section>\n"
            "<stimulus>This is the reading stimulus.</stimulus>\n"
            "<question_label>Question 1:</question_label> <stem>What is this?</stem>\n"
            "<option_label>A.</option_label> <option_text>Choice A</option_text>"
        )
        raw_text, spans = parse_xml_annotations(tagged_text)
        labels = [s["label"] for s in spans]
        self.assertIn("section", labels)
        self.assertIn("stimulus", labels)
        self.assertIn("question_label", labels)
        self.assertIn("stem", labels)
        self.assertIn("option_label", labels)
        self.assertIn("option_text", labels)

        stim_span = next(s for s in spans if s["label"] == "stimulus")
        self.assertEqual(stim_span["text"], "This is the reading stimulus.")

    def test_spans_to_xml_with_stimulus(self):
        raw_text = "Stimulus text. Question 1: Stem text."
        spans = [
            {"start": 0, "end": 14, "label": "stimulus"},
            {"start": 15, "end": 26, "label": "question_label"},
            {"start": 27, "end": 37, "label": "stem"}
        ]
        xml = spans_to_xml(raw_text, spans)
        self.assertEqual(xml, "<stimulus>Stimulus text.</stimulus> <question_label>Question 1:</question_label> <stem>Stem text.</stem>")

    def test_consolidate_raw_exams(self):
        import tempfile
        import json
        from pathlib import Path
        from src.data.prepare import consolidate_raw_exams

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # 1. Create a synthetic question file
            q_file = tmppath / "question_math_g10_1.json"
            with open(q_file, "w", encoding="utf-8") as f:
                json.dump({
                    "is_group": False,
                    "stem": "Tính giá trị $x$?",
                    "options": ["1", "2", "3", "4"],
                    "question_type": "multiple_choice",
                    "subject": "math_algebra",
                    "grade": 10
                }, f)

            # 2. Create a real annotated folder with audit PASS
            real_folder = tmppath / "real_annotated" / "exam_test"
            real_folder.mkdir(parents=True)
            with open(real_folder / "merged.xml", "w", encoding="utf-8") as f:
                f.write("<question_label>Câu 1:</question_label> <stem>Đề thi thử</stem>")
            with open(real_folder / "audit_report.json", "w", encoding="utf-8") as f:
                json.dump({"decision": "PASS", "is_malfunctioned": False}, f)

            # 3. Create a real annotated folder with audit FAIL
            fail_folder = tmppath / "real_annotated" / "exam_fail"
            fail_folder.mkdir(parents=True)
            with open(fail_folder / "merged.xml", "w", encoding="utf-8") as f:
                f.write("<question_label>Câu 1:</question_label> <stem>Hỏng</stem>")
            with open(fail_folder / "audit_report.json", "w", encoding="utf-8") as f:
                json.dump({"decision": "FAIL", "is_malfunctioned": True}, f)

            out_jsonl = tmppath / "raw_exams.jsonl"
            records = consolidate_raw_exams(
                input_dir=tmppath,
                output_file=out_jsonl,
                filter_passed=True
            )

            self.assertTrue(out_jsonl.exists())
            self.assertEqual(len(records), 2)  # 1 synthetic + 1 passed real (fail filtered out)
            ids = [r["exam_id"] for r in records]
            self.assertIn("real_annotated_exam_test", ids)
            self.assertNotIn("real_annotated_exam_fail", ids)

    def test_fix_xml_stimulus_citations_trailing_untagged(self):
        from src.data.fix_root_data import fix_xml_stimulus_citations
        xml = "<stimulus>Passage text.</stimulus>\n*(Adapted from CNN)*\n<question_label>Question 1:</question_label>"
        normalized = fix_xml_stimulus_citations(xml)
        self.assertIn("<stimulus>Passage text.\n\n*(Adapted from CNN)*</stimulus>", normalized)
        self.assertNotIn("</stimulus>\n*(Adapted from CNN)*", normalized)

    def test_fix_xml_stimulus_citations_in_stem(self):
        from src.data.fix_root_data import fix_xml_stimulus_citations
        xml = "<stimulus>Passage text.</stimulus>\n<stem>(Adapted from The Guardian)\nWhat is the main idea?</stem>"
        normalized = fix_xml_stimulus_citations(xml)
        self.assertIn("<stimulus>Passage text.\n\n(Adapted from The Guardian)</stimulus>", normalized)
        self.assertIn("<stem>What is the main idea?</stem>", normalized)

    def test_parse_xml_annotations_with_explanation(self):
        xml = "<question_label>Câu 1:</question_label> <stem>Tính 1+1?</stem> <explanation>Lời giải: 1+1=2. Chọn A.</explanation>"
        raw_text, spans = parse_xml_annotations(xml)
        expl_spans = [s for s in spans if s["label"] == "explanation"]
        self.assertEqual(len(expl_spans), 1)
        self.assertEqual(expl_spans[0]["text"], "Lời giải: 1+1=2. Chọn A.")
        self.assertEqual(raw_text[expl_spans[0]["start"]:expl_spans[0]["end"]], "Lời giải: 1+1=2. Chọn A.")

    def test_document_level_partitioning_no_leakage(self):
        import tempfile
        import json
        from pathlib import Path
        import argparse
        from unittest.mock import patch
        from src.data.prepare import main

        with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
            in_path = Path(tmp_in)
            out_path = Path(tmp_out)

            # Create 10 synthetic question files
            for i in range(10):
                q_file = in_path / f"question_math_g10_{i}.json"
                with open(q_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "is_group": False,
                        "stem": f"Câu hỏi số {i} với độ dài nội dung dài để tạo sliding window.",
                        "options": ["A", "B", "C", "D"],
                        "question_type": "multiple_choice",
                        "subject": "math_algebra",
                        "grade": 10
                    }, f)

            test_args = [
                "prepare.py",
                "--input-dir", str(in_path),
                "--output-dir", str(out_path),
                "--model", "mock-model",
                "--train-ratio", "0.6",
                "--val-ratio", "0.2",
                "--seed", "42"
            ]

            with patch("sys.argv", test_args), patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer()):
                main()

            train_file = out_path / "train.jsonl"
            val_file = out_path / "val.jsonl"
            test_file = out_path / "test.jsonl"

            self.assertTrue(train_file.exists())
            self.assertTrue(val_file.exists())
            self.assertTrue(test_file.exists())

            train_sources = set()
            with open(train_file) as f:
                for line in f:
                    train_sources.add(json.loads(line)["metadata"]["source_file"])

            val_sources = set()
            with open(val_file) as f:
                for line in f:
                    val_sources.add(json.loads(line)["metadata"]["source_file"])

            test_sources = set()
            with open(test_file) as f:
                for line in f:
                    test_sources.add(json.loads(line)["metadata"]["source_file"])

            # Zero leakage verification
            self.assertEqual(len(train_sources.intersection(val_sources)), 0)
            self.assertEqual(len(train_sources.intersection(test_sources)), 0)
            self.assertEqual(len(val_sources.intersection(test_sources)), 0)

if __name__ == '__main__':
    unittest.main()


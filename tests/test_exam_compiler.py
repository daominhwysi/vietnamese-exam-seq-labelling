import unittest
from unittest.mock import patch
from src.generation.exam_compiler import (
    generate_exam_tasks,
    generate_single_exam,
    SECTION_MC,
    SECTION_TF,
    SECTION_SA,
    SECTION_ESSAY
)
from src.generation.generator import Subject

class TestExamCompiler(unittest.TestCase):
    def test_generate_exam_tasks(self):
        tasks = generate_exam_tasks()
        sections = [t[0] for t in tasks]
        
        mc_count = sections.count(SECTION_MC)
        self.assertTrue(10 <= mc_count <= 20)
        
        tf_count = sections.count(SECTION_TF)
        self.assertTrue(1 <= tf_count <= 6)
        
        sa_count = sections.count(SECTION_SA)
        self.assertTrue(0 <= sa_count <= 6)
        
        essay_count = sections.count(SECTION_ESSAY)
        self.assertTrue(0 <= essay_count <= 4)

    @patch('src.generation.exam_compiler.generate_single_question')
    def test_generate_single_exam(self, mock_gen):
        mock_gen.return_value = {
            "stem": "Dummy question stem",
            "options": ["A", "B", "C", "D"],
            "raw_text": "Dummy raw text",
            "spans": [{"start": 0, "end": 5}],
            "problem_type_level": "VD"
        }
        
        exam_data = generate_single_exam(
            subject=Subject.CHEMISTRY,
            grade=10,
            concurrency=2
        )
        
        self.assertIsNotNone(exam_data)
        self.assertEqual(exam_data["subject"], "chemistry")
        self.assertEqual(exam_data["grade"], 10)
        self.assertIn("exam_id", exam_data)
        
        sections = exam_data["sections"]
        self.assertIn(SECTION_MC, sections)
        self.assertIn(SECTION_TF, sections)
        self.assertIn(SECTION_SA, sections)
        self.assertIn(SECTION_ESSAY, sections)
        
        mc_questions = sections[SECTION_MC]
        self.assertTrue(len(mc_questions) >= 10)
        for q in mc_questions:
            self.assertNotIn("raw_text", q)
            self.assertNotIn("spans", q)
            self.assertNotIn("problem_type_level", q)

if __name__ == '__main__':
    unittest.main()

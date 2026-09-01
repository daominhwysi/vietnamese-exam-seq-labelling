import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.exam_compiler import generate_exam_tasks
from src.generation.generator import Subject
from src.generation.curriculum import load_curriculum, select_curriculum_path

# 1. Test TOEIC curriculum loading
print("=== 1. Checking TOEIC Curriculum Loading ===")
curr_toeic = load_curriculum("toeic", 12)
assert curr_toeic is not None, "Failed to load toeic_12.json"
print(f"Loaded TOEIC curriculum with {len(curr_toeic['chapters'])} chapters.")

# 2. Test selecting paths from TOEIC curriculum
for pt_name in ["toeic_part1_human_action", "toeic_part5_word_form", "toeic_part7_chat_chains", "toeic_part7_triple_passages"]:
    res = select_curriculum_path(curr_toeic, problem_type_filter=pt_name)
    assert res is not None, f"Failed to select path for {pt_name}"
    ch, un, pt = res
    print(f" - Found [{pt['id']}]: {pt['name']} (Chapter: {ch['name']})")

# 3. Test Literature curriculum loading
print("\n=== 2. Checking Literature Curriculum Loading ===")
curr_lit = load_curriculum("literature", 12)
assert curr_lit is not None, "Failed to load literature_12.json"
print(f"Loaded Literature curriculum with {len(curr_lit['chapters'])} chapters.")

# 4. Test selecting paths from Literature curriculum
for pt_name in ["reading_comprehension_literature_poetry", "literary_comparative_essay_600", "social_philosophical_dialogue_hsg"]:
    res = select_curriculum_path(curr_lit, problem_type_filter=pt_name)
    assert res is not None, f"Failed to select path for {pt_name}"
    ch, un, pt = res
    print(f" - Found [{pt['id']}]: {pt['name']} (Chapter: {ch['name']})")

# 5. Test Exam Task Compilation for TOEIC & Literature
print("\n=== 3. Checking Exam Tasks Generation ===")
toeic_tasks = generate_exam_tasks(Subject.TOEIC)
print(f"Generated {len(toeic_tasks)} tasks for a TOEIC exam.")
sections_toeic = set(t[0] for t in toeic_tasks)
print(f"Sections in TOEIC exam: {sections_toeic}")

lit_tasks = generate_exam_tasks(Subject.LITERATURE)
print(f"Generated {len(lit_tasks)} tasks for a Literature exam.")
sections_lit = set(t[0] for t in lit_tasks)
print(f"Sections in Literature exam: {sections_lit}")

print("\n>>> ALL TESTS COMPLETED SUCCESSFULLY! <<<")

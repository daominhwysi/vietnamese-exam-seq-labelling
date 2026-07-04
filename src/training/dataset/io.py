import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

def scan_input_files(input_dir: str) -> Tuple[List[Path], List[Path]]:
    input_path = Path(input_dir)
    json_files = list(input_path.glob("question_*.json"))
    exam_files = list(input_path.glob("**/exam_*.json"))
    real_exam_files = list(input_path.glob("**/real_exam_*.json"))
    
    # Combine real exams into exam_files list
    exam_files.extend(real_exam_files)
    return json_files, exam_files

def save_jsonl_split(samples: List[Dict[str, Any]], output_file: Path):
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

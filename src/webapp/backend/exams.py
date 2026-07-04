import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.generation.generator import SUBJECT_DISPLAY

EXAMS_DIR = Path("output/exams")
REAL_EXAMS_DIR = Path("output/real-exams")

# Global in-memory cache to speed up exam loading
_exams_cache: Dict[str, Dict[str, Any]] = {}

def get_all_exams() -> List[Dict[str, Any]]:
    global _exams_cache
    exam_files = []
    if EXAMS_DIR.exists():
        exam_files.extend(list(EXAMS_DIR.glob("*.json")))
    if REAL_EXAMS_DIR.exists():
        exam_files.extend(list(REAL_EXAMS_DIR.glob("*.json")))
        
    if not exam_files:
        return []
    
    exams = []
    for file in exam_files:
        filepath_str = str(file.resolve())
        try:
            mtime = file.stat().st_mtime
            
            if filepath_str in _exams_cache and _exams_cache[filepath_str]["mtime"] == mtime:
                exams.append(_exams_cache[filepath_str]["metadata"])
                continue
                
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if data.get("is_real", False):
                q_count = 1
            else:
                q_count = 0
                sections = data.get("sections", {})
                for sec_name, q_list in sections.items():
                    for q in q_list:
                        if q.get("is_group"):
                            q_count += len(q.get("questions", []))
                        else:
                            q_count += 1
                        
            created_str = data.get("created_at", "")
            try:
                dt = datetime.fromisoformat(created_str)
                formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                formatted_date = created_str
                
            exam_meta = {
                "filename": file.name,
                "exam_id": data.get("exam_id"),
                "subject": data.get("subject"),
                "subject_display": SUBJECT_DISPLAY.get(data.get("subject"), data.get("subject")),
                "grade": data.get("grade"),
                "created_at": formatted_date,
                "question_count": q_count,
                "raw_created_at": created_str,
                "is_real": data.get("is_real", False),
                "annotated": data.get("annotated", len(data.get("spans", [])) > 0)
            }
            
            _exams_cache[filepath_str] = {
                "mtime": mtime,
                "metadata": exam_meta
            }
            exams.append(exam_meta)
        except Exception as e:
            print(f"Error loading exam {file}: {e}")
            
    exams.sort(key=lambda x: x["raw_created_at"] or "", reverse=True)
    return exams

def find_exam_file(exam_id: str) -> Optional[Path]:
    search_dirs = [EXAMS_DIR, REAL_EXAMS_DIR]
    for directory in search_dirs:
        if not directory.exists():
            continue
        for file in directory.glob("*.json"):
            if exam_id in file.name:
                return file
            try:
                with open(file, "r", encoding="utf-8") as _f:
                    _data = json.load(_f)
                if _data.get("exam_id") == exam_id:
                    return file
            except Exception:
                continue
    return None

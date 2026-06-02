import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from src.generation.reconstructor import (
    reconstruct_question, 
    ReconstructorConfig, 
    get_stable_random, 
    generate_ordering_choices
)
from src.generation.generator import SUBJECT_DISPLAY, QUESTION_TYPE_DISPLAY, DIFFICULTY_DISPLAY

app = FastAPI(title="Vietnamese Exam Sequence Labeling Web Viewer")

# Setup templates
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

EXAMS_DIR = Path("output/exams")

def get_all_exams() -> List[Dict[str, Any]]:
    if not EXAMS_DIR.exists():
        return []
    
    exams = []
    for file in EXAMS_DIR.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Count total questions
            q_count = 0
            sections = data.get("sections", {})
            for sec_name, q_list in sections.items():
                for q in q_list:
                    if q.get("is_group"):
                        q_count += len(q.get("questions", []))
                    else:
                        q_count += 1
                        
            # Format datetime
            created_str = data.get("created_at", "")
            try:
                dt = datetime.fromisoformat(created_str)
                formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                formatted_date = created_str
                
            exams.append({
                "filename": file.name,
                "exam_id": data.get("exam_id"),
                "subject": data.get("subject"),
                "subject_display": SUBJECT_DISPLAY.get(data.get("subject"), data.get("subject")),
                "grade": data.get("grade"),
                "created_at": formatted_date,
                "question_count": q_count,
                "raw_created_at": created_str
            })
        except Exception as e:
            print(f"Error loading exam {file}: {e}")
            
    # Sort by created time descending
    exams.sort(key=lambda x: x["raw_created_at"], reverse=True)
    return exams

@app.get("/")
def index(request: Request):
    exams = get_all_exams()
    
    # Calculate stats
    total_exams = len(exams)
    subject_counts = {}
    grade_counts = {}
    for ex in exams:
        subj = ex["subject_display"]
        grade = f"Lớp {ex['grade']}"
        subject_counts[subj] = subject_counts.get(subj, 0) + 1
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "exams": exams,
            "total_exams": total_exams,
            "subject_counts": subject_counts,
            "grade_counts": grade_counts
        }
    )

@app.get("/random")
def get_random_exam():
    exams = get_all_exams()
    if not exams:
        raise HTTPException(status_code=404, detail="No exams found. Generate some first!")
    random_exam = random.choice(exams)
    return RedirectResponse(url=f"/exam/{random_exam['exam_id']}", status_code=303)

@app.get("/exam/{exam_id}")
def view_exam(request: Request, exam_id: str):
    # Find the file with matching exam_id
    if not EXAMS_DIR.exists():
        raise HTTPException(status_code=404, detail="Exams directory not found.")
        
    target_file = None
    for file in EXAMS_DIR.glob("*.json"):
        if exam_id in file.name:
            target_file = file
            break
            
    if not target_file:
        raise HTTPException(status_code=404, detail=f"Exam with ID {exam_id} not found.")
        
    with open(target_file, "r", encoding="utf-8") as f:
        exam_data = json.load(f)
        
    # Standardize question formats, sequential numbering, and run reconstructor for span highlights
    sections = exam_data.get("sections", {})
    reconstructed_sections = {}
    
    q_num = 1
    for section_title, questions in sections.items():
        reconstructed_questions = []
        for q in questions:
            stable_seed = q.get("context", "") or q.get("stem", "") or str(q)
            config = ReconstructorConfig(randomize_q_num=False, include_span_text=True, seed=stable_seed)
            
            # Reconstruct to get raw_text and spans
            q_reconstructed = reconstruct_question(q, config=config, start_q_num=q_num)
            
            # Normalize `<blank/>` to `<blank></blank>` for safe HTML rendering without layout breakage in standard view
            if q_reconstructed.get("subject") == "english":
                import re
                def normalize_blanks(text: str) -> str:
                    if not text:
                        return text
                    return re.sub(r'<\s*blank\s*/?\s*>', '<blank></blank>', text)
                
                if "stem" in q_reconstructed:
                    q_reconstructed["stem"] = normalize_blanks(q_reconstructed["stem"])
                if "context" in q_reconstructed:
                    q_reconstructed["context"] = normalize_blanks(q_reconstructed["context"])
                if "questions" in q_reconstructed:
                    q_reconstructed["questions"] = [dict(sub) for sub in q_reconstructed["questions"]]
                    for sub_q in q_reconstructed["questions"]:
                        if "stem" in sub_q:
                            sub_q["stem"] = normalize_blanks(sub_q["stem"])
            
            # Record starting and ending question numbers for this item
            q_reconstructed["start_number"] = q_num
            if q.get("is_group"):
                sub_q_count = len(q.get("questions", []))
                q_reconstructed["end_number"] = q_num + sub_q_count - 1
                q_num += sub_q_count
            else:
                q_reconstructed["end_number"] = q_num
                q_num += 1
                
            # If ordering question, generate choices based on stable seed
            if q.get("question_type") == "ordering":
                rng = get_stable_random(stable_seed)
                item_labels = ["a", "b", "c", "d", "e", "f", "g", "h"][:len(q.get("options", []))]
                choices = generate_ordering_choices(item_labels, " – ", rng)
                q_reconstructed["ordering_choices"] = choices
                
            reconstructed_questions.append(q_reconstructed)
            
        reconstructed_sections[section_title] = reconstructed_questions
        
    exam_data["sections"] = reconstructed_sections
    exam_data["total_questions"] = q_num - 1
    exam_data["subject_display"] = SUBJECT_DISPLAY.get(exam_data.get("subject"), exam_data.get("subject"))
    
    return templates.TemplateResponse(
        request=request,
        name="viewer.html",
        context={
            "exam": exam_data,
            "SUBJECT_DISPLAY": SUBJECT_DISPLAY,
            "QUESTION_TYPE_DISPLAY": QUESTION_TYPE_DISPLAY,
            "DIFFICULTY_DISPLAY": DIFFICULTY_DISPLAY
        }
    )

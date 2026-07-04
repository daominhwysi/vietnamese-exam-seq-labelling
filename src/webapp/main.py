import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.generation.reconstructor import (
    reconstruct_question, 
    reconstruct_exam,
    ReconstructorConfig, 
    get_stable_random, 
    generate_ordering_choices
)
from src.generation.generator import SUBJECT_DISPLAY, QUESTION_TYPE_DISPLAY, DIFFICULTY_DISPLAY
from src.webapp.backend import (
    load_or_compute_dataset_stats,
    get_all_exams,
    find_exam_file
)

app = FastAPI(title="Vietnamese Exam Sequence Labeling Web Viewer")

# Setup templates
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount assets directory for React Vite SPA frontend
frontend_assets = BASE_DIR / "frontend" / "dist" / "assets"
frontend_assets.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(frontend_assets)), name="assets")

EXAMS_DIR = Path("output/exams")
REAL_EXAMS_DIR = Path("output/real-exams")
DATASET_DIR = Path("output/dataset")

@app.get("/")
def index(request: Request):
    FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"
    if (FRONTEND_DIST_DIR / "index.html").exists():
        return FileResponse(str(FRONTEND_DIST_DIR / "index.html"))
        
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
    target_file = find_exam_file(exam_id)
    if not target_file:
        raise HTTPException(status_code=404, detail=f"Exam with ID {exam_id} not found.")
        
    with open(target_file, "r", encoding="utf-8") as f:
        exam_data = json.load(f)
        
    if exam_data.get("is_real", False):
        from src.webapp.inference_helper import parse_segments_to_questions
        spans = exam_data.get("spans", [])
        parsed_qs = parse_segments_to_questions(spans)
        
        if parsed_qs:
            formatted_questions = []
            for idx, pq in enumerate(parsed_qs, start=1):
                formatted_questions.append({
                    "is_group": bool(pq.get("context")),
                    "context": pq.get("context", ""),
                    "stem": pq.get("stem", ""),
                    "options": pq.get("options", []),
                    "answer": "",
                    "explanation": pq.get("explanation", "") or "Đây là câu hỏi từ đề thi thực tế.",
                    "subject": exam_data.get("subject"),
                    "grade": exam_data.get("grade"),
                    "question_type": "multiple_choice" if pq.get("options") else "short_answer",
                    "raw_text": exam_data.get("raw_text", ""),
                    "spans": spans,
                    "start_number": idx,
                    "end_number": idx
                })
            exam_data["sections"] = {
                "ĐỀ THI THỰC TẾ (ĐÃ PHÂN TÁCH)": formatted_questions
            }
            exam_data["total_questions"] = len(formatted_questions)
        else:
            exam_data["sections"] = {
                "ĐỀ THI THỰC TẾ (REAL EXAM PAPER)": [
                    {
                        "is_group": False,
                        "stem": exam_data.get("raw_text", ""),
                        "options": [],
                        "answer": "",
                        "explanation": "Đây là đề thi thực tế đã được gán nhãn OCR.",
                        "subject": exam_data.get("subject"),
                        "grade": exam_data.get("grade"),
                        "question_type": "real_exam",
                        "raw_text": exam_data.get("raw_text", ""),
                        "spans": spans,
                        "start_number": 1,
                        "end_number": 1
                    }
                ]
            }
            exam_data["total_questions"] = 1
        exam_data["subject_display"] = SUBJECT_DISPLAY.get(exam_data.get("subject"), exam_data.get("subject"))
    else:
        sections = exam_data.get("sections", {})
        reconstructed_sections = {}
        
        q_num = 1
        for section_title, questions in sections.items():
            reconstructed_questions = []
            for q in questions:
                stable_seed = q.get("context", "") or q.get("stem", "") or str(q)
                is_english = q.get("subject") == "english" or exam_data.get("subject") == "english"
                config = ReconstructorConfig(
                    randomize_q_num=False, 
                    include_span_text=True, 
                    seed=stable_seed,
                    inline_option_prob=0.50 if is_english else 0.10,
                    typo_rate=0.02,
                    space_noise_rate=0.15,
                    latex_mask_prob=0.50,
                    enable_permutations=False,
                    option_drop_prob=0.05,
                    casing_noise_prob=0.10,
                    synonym_swap_prob=0.10,
                    formatting_noise_prob=0.10,
                    min_inline_spaces=4,
                    max_inline_spaces=12,
                    min_inline_tabs=1,
                    max_inline_tabs=2
                )
                
                q_reconstructed = reconstruct_question(q, config=config, start_q_num=q_num)
                
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
                
                q_reconstructed["start_number"] = q_num
                if q.get("is_group"):
                    sub_q_count = len(q.get("questions", []))
                    q_reconstructed["end_number"] = q_num + sub_q_count - 1
                    q_num += sub_q_count
                else:
                    q_reconstructed["end_number"] = q_num
                    q_num += 1
                    
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

@app.get("/dataset")
def dataset_dashboard(request: Request):
    splits = ["train", "val", "test"]
    stats_data = {}
    for split in splits:
        stats_data[split] = load_or_compute_dataset_stats(split)
        
    return templates.TemplateResponse(
        request=request,
        name="dataset_dashboard.html",
        context={
            "stats": stats_data,
            "SUBJECT_DISPLAY": SUBJECT_DISPLAY
        }
    )

@app.get("/dataset/{split}")
def view_dataset_split(
    request: Request, 
    split: str, 
    page: int = 1, 
    page_size: int = 20, 
    subject: Optional[str] = None, 
    grade: Optional[int] = None
):
    if split not in ["train", "val", "test"]:
        raise HTTPException(status_code=404, detail="Dataset split not found. Must be train, val, or test.")
        
    filepath = DATASET_DIR / f"{split}.jsonl"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Dataset split file for '{split}' not found. Please prepare the dataset first.")
        
    samples = []
    total_matching = 0
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    stats = load_or_compute_dataset_stats(split)
    available_subjects = list(stats.get("subjects", {}).keys())
    available_grades = sorted([int(g) for g in stats.get("grades", {}).keys() if g.isdigit()])
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line)
            except Exception:
                continue
            metadata = sample.get("metadata", {})
            
            if subject and metadata.get("subject") != subject:
                continue
            if grade and metadata.get("grade") != grade:
                try:
                    if int(metadata.get("grade")) != int(grade):
                        continue
                except Exception:
                    if metadata.get("grade") != grade:
                        continue
                
            if start_idx <= total_matching < end_idx:
                samples.append(sample)
            total_matching += 1
            
    total_pages = max(1, (total_matching + page_size - 1) // page_size)
    page = min(page, total_pages)
    page = max(1, page)
    
    start_sample_idx = (page - 1) * page_size + 1 if total_matching > 0 else 0
    end_sample_idx = min(page * page_size, total_matching)
    
    return templates.TemplateResponse(
        request=request,
        name="dataset_viewer.html",
        context={
            "split": split,
            "samples": samples,
            "page": page,
            "page_size": page_size,
            "total_matching": total_matching,
            "total_pages": total_pages,
            "start_sample_idx": start_sample_idx,
            "end_sample_idx": end_sample_idx,
            "selected_subject": subject,
            "selected_grade": grade,
            "available_subjects": available_subjects,
            "available_grades": available_grades,
            "SUBJECT_DISPLAY": SUBJECT_DISPLAY,
            "QUESTION_TYPE_DISPLAY": QUESTION_TYPE_DISPLAY,
            "DIFFICULTY_DISPLAY": DIFFICULTY_DISPLAY
        }
    )

@app.get("/exam/{exam_id}/json")
def download_exam_json(exam_id: str):
    file_path = find_exam_file(exam_id)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Exam not found")
    return FileResponse(path=file_path, filename=file_path.name, media_type="application/json")

@app.get("/exam/{exam_id}/xml")
def download_exam_xml(exam_id: str):
    file_path = find_exam_file(exam_id)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Exam not found")
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    from src.training.dataset.alignment import spans_to_xml
    
    if data.get("is_real", False) and "raw_text" in data and "spans" in data:
        raw_text = data["raw_text"]
        spans = data["spans"]
    else:
        stable_seed = data.get("exam_id", "") or str(data)
        config = ReconstructorConfig(seed=stable_seed, randomize_q_num=False)
        reconstructed = reconstruct_exam(data, config)
        raw_text = reconstructed["raw_text"]
        spans = reconstructed["spans"]
        
    xml_content = spans_to_xml(raw_text, spans)
    
    filename = file_path.stem + "_annotated.xml"
    return Response(
        content=xml_content,
        media_type="application/xml",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.post("/api/exam/{exam_id}/spans")
def save_exam_spans(exam_id: str, payload: Dict[str, Any]):
    file_path = find_exam_file(exam_id)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Exam not found")
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        spans = payload.get("spans", [])
        
        raw_text = data.get("raw_text", "")
        if not raw_text:
            raise HTTPException(status_code=400, detail="Exam file is missing raw_text")
            
        normalized_spans = []
        for span in spans:
            start = int(span["start"])
            end = int(span["end"])
            label = span["label"]
            
            start = max(0, min(start, len(raw_text)))
            end = max(start, min(end, len(raw_text)))
            
            if start == end:
                continue
                
            span_text = raw_text[start:end]
            normalized_spans.append({
                "start": start,
                "end": end,
                "label": label,
                "text": span_text
            })
            
        data["spans"] = normalized_spans
        
        from src.training.dataset.alignment import spans_to_xml
        data["raw_xml"] = spans_to_xml(raw_text, normalized_spans)
        data["annotated"] = True

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        xml_file = file_path.with_suffix(".xml")
        if xml_file.exists() or file_path.parent.name in ["real_exams", "real-exams"]:
            xml_content = spans_to_xml(raw_text, normalized_spans)
            with open(xml_file, "w", encoding="utf-8") as f:
                f.write(xml_content)
                
        return {"status": "success", "message": "Spans and XML saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/exam/{exam_id}/annotated")
def save_exam_annotated_status(exam_id: str, payload: Dict[str, Any]):
    file_path = find_exam_file(exam_id)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Exam not found")
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        annotated = bool(payload.get("annotated", False))
        data["annotated"] = annotated
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return {"status": "success", "annotated": annotated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------
# UNIFIED JSON REST API ENDPOINTS FOR VITE REACT SPA FRONTEND
# -------------------------------------------------------------
from pydantic import BaseModel
from src.webapp.inference_helper import model_manager, run_model_inference

class LoadModelRequest(BaseModel):
    model_path: str
    base_model_name: str = "aisingapore/SEA-LION-ModernBERT-300M"
    device_choice: str = "auto"

class InferenceRequest(BaseModel):
    raw_text: str
    max_length: int = 1024
    stride: int = 256

@app.get("/api/exams")
def api_get_exams():
    try:
        return get_all_exams()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/exams/{exam_id}")
def api_get_exam_details(exam_id: str):
    target_file = find_exam_file(exam_id)
    if not target_file:
        raise HTTPException(status_code=404, detail=f"Exam with ID {exam_id} not found.")
        
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            exam_data = json.load(f)
            
        exam_data["annotated"] = exam_data.get("annotated", len(exam_data.get("spans", [])) > 0)
        
        if exam_data.get("is_real", False):
            from src.webapp.inference_helper import parse_segments_to_questions
            spans = exam_data.get("spans", [])
            parsed_qs = parse_segments_to_questions(spans)
            
            if parsed_qs:
                formatted_questions = []
                for idx, pq in enumerate(parsed_qs, start=1):
                    formatted_questions.append({
                        "is_group": bool(pq.get("context")),
                        "context": pq.get("context", ""),
                        "stem": pq.get("stem", ""),
                        "options": pq.get("options", []),
                        "answer": "",
                        "explanation": pq.get("explanation", "") or "Đây là câu hỏi từ đề thi thực tế.",
                        "subject": exam_data.get("subject"),
                        "grade": exam_data.get("grade"),
                        "question_type": "multiple_choice" if pq.get("options") else "short_answer",
                        "raw_text": exam_data.get("raw_text", ""),
                        "spans": spans,
                        "start_number": idx,
                        "end_number": idx
                    })
                exam_data["sections"] = {
                    "ĐỀ THI THỰC TẾ (ĐÃ PHÂN TÁCH)": formatted_questions
                }
                exam_data["total_questions"] = len(formatted_questions)
            else:
                exam_data["sections"] = {
                    "ĐỀ THI THỰC TẾ (REAL EXAM PAPER)": [
                        {
                            "is_group": False,
                            "stem": exam_data.get("raw_text", ""),
                            "options": [],
                            "answer": "",
                            "explanation": "Đây là đề thi thực tế đã được gán nhãn OCR.",
                            "subject": exam_data.get("subject"),
                            "grade": exam_data.get("grade"),
                            "question_type": "real_exam",
                            "raw_text": exam_data.get("raw_text", ""),
                            "spans": spans,
                            "start_number": 1,
                            "end_number": 1
                        }
                    ]
                }
                exam_data["total_questions"] = 1
            exam_data["subject_display"] = SUBJECT_DISPLAY.get(exam_data.get("subject"), exam_data.get("subject"))
        else:
            sections = exam_data.get("sections", {})
            reconstructed_sections = {}
            
            q_num = 1
            for section_title, questions in sections.items():
                reconstructed_questions = []
                for q in questions:
                    stable_seed = q.get("context", "") or q.get("stem", "") or str(q)
                    is_english = q.get("subject") == "english" or exam_data.get("subject") == "english"
                    config = ReconstructorConfig(
                        randomize_q_num=False, 
                        include_span_text=True, 
                        seed=stable_seed,
                        inline_option_prob=0.50 if is_english else 0.10,
                        typo_rate=0.02,
                        space_noise_rate=0.15,
                        latex_mask_prob=0.50,
                        enable_permutations=False,
                        option_drop_prob=0.05,
                        casing_noise_prob=0.10,
                        synonym_swap_prob=0.10,
                        formatting_noise_prob=0.10,
                        min_inline_spaces=4,
                        max_inline_spaces=12,
                        min_inline_tabs=1,
                        max_inline_tabs=2
                    )
                    
                    q_reconstructed = reconstruct_question(q, config=config, start_q_num=q_num)
                    
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
                    
                    q_reconstructed["start_number"] = q_num
                    if q.get("is_group"):
                        sub_q_count = len(q.get("questions", []))
                        q_reconstructed["end_number"] = q_num + sub_q_count - 1
                        q_num += sub_q_count
                    else:
                        q_reconstructed["end_number"] = q_num
                        q_num += 1
                        
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
            
        return exam_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/stats")
def api_dataset_stats():
    splits = ["train", "val", "test"]
    stats_data = {}
    for split in splits:
        stats_data[split] = load_or_compute_dataset_stats(split)
    return stats_data

@app.get("/api/dataset/{split}")
def api_dataset_split(
    split: str, 
    page: int = 1, 
    page_size: int = 20, 
    subject: Optional[str] = None, 
    grade: Optional[int] = None
):
    if split not in ["train", "val", "test"]:
        raise HTTPException(status_code=404, detail="Dataset split not found. Must be train, val, or test.")
        
    filepath = DATASET_DIR / f"{split}.jsonl"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Dataset split file for '{split}' not found.")
        
    samples = []
    total_matching = 0
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    stats = load_or_compute_dataset_stats(split)
    available_subjects = list(stats.get("subjects", {}).keys())
    available_grades = sorted([int(g) for g in stats.get("grades", {}).keys() if g.isdigit()])
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                except Exception:
                    continue
                metadata = sample.get("metadata", {})
                
                if subject and metadata.get("subject") != subject:
                    continue
                if grade and metadata.get("grade") != grade:
                    try:
                        if int(metadata.get("grade")) != int(grade):
                            continue
                    except Exception:
                        if metadata.get("grade") != grade:
                            continue
                    
                if start_idx <= total_matching < end_idx:
                    samples.append(sample)
                total_matching += 1
                
        total_pages = max(1, (total_matching + page_size - 1) // page_size)
        page = min(page, total_pages)
        page = max(1, page)
        
        start_sample_idx = (page - 1) * page_size + 1 if total_matching > 0 else 0
        end_sample_idx = min(page * page_size, total_matching)
        
        return {
            "split": split,
            "samples": samples,
            "page": page,
            "page_size": page_size,
            "total_matching": total_matching,
            "total_pages": total_pages,
            "start_sample_idx": start_sample_idx,
            "end_sample_idx": end_sample_idx,
            "available_subjects": available_subjects,
            "available_grades": available_grades
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load-model")
def api_load_model(req: LoadModelRequest):
    model_manager.load_model_in_background(
        model_path=req.model_path,
        base_model_name=req.base_model_name,
        device_choice=req.device_choice
    )
    return {"message": "Model loading started in background."}

@app.get("/api/model-status")
def api_model_status():
    return model_manager.get_status()

@app.post("/api/run-inference")
def api_run_inference(req: InferenceRequest):
    if model_manager.model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")
    try:
        results = run_model_inference(
            raw_text=req.raw_text,
            max_length=req.max_length,
            stride=req.stride
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/presets")
def api_list_presets():
    presets_dir = Path("sample")
    if not presets_dir.exists():
        return []
    txt_files = list(presets_dir.glob("*.txt"))
    return [f.name for f in txt_files]

@app.get("/api/presets/{filename}")
def api_get_preset_content(filename: str):
    presets_dir = Path("sample")
    filepath = presets_dir / filename
    
    resolved_path = filepath.resolve()
    resolved_presets_dir = presets_dir.resolve()
    if not resolved_path.is_relative_to(resolved_presets_dir):
        raise HTTPException(status_code=403, detail="Access denied.")
        
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="Preset file not found.")
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

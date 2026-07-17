import os
import json
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.webapp.inference_helper import model_manager, run_model_inference

app = FastAPI(title="Vietnamese Exam Sequence Labeling Web Inference")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

class LoadModelRequest(BaseModel):
    model_path: str
    base_model_name: str = "aisingapore/SEA-LION-ModernBERT-300M"
    device_choice: str = "auto"

class InferenceRequest(BaseModel):
    raw_text: str
    max_length: int = 1024
    stride: int = 256

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="inference.html",
        context={}
    )

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
    
    # Prevent directory traversal
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

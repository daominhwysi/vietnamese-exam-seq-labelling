# Backward compatibility wrapper for webapp inference

from src.webapp.inference import (
    ModelManager,
    model_manager,
    is_valid_latex,
    get_latex_spans,
    OffsetMapper,
    resolve_bio_violations,
    build_xml,
    parse_segments_to_questions,
)
from src.webapp.inference.core import run_model_inference as _run_model_inference

def run_model_inference(
    raw_text: str,
    max_length: int = 1024,
    stride: int = 256
):
    """
    Backward-compatible wrapper using global model_manager.
    """
    return _run_model_inference(
        model_manager=model_manager,
        raw_text=raw_text,
        max_length=max_length,
        stride=stride
    )

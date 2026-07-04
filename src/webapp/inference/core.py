import re
from typing import Dict, Any, List, Tuple
from src.webapp.inference.latex import get_latex_spans, OffsetMapper
from src.webapp.inference.bio import resolve_bio_violations
from src.webapp.inference.parser import build_xml, parse_segments_to_questions
from src.webapp.inference.manager import ModelManager

# Import torch conditionally
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

def run_model_inference(
    model_manager: ModelManager,
    raw_text: str,
    max_length: int = 1024,
    stride: int = 256
) -> Dict[str, Any]:
    """
    Main inference worker. Tokenizes raw text with sliding window, aggregates overlapping
    logits, recovers character offsets, maps spans, and outputs structures.
    """
    if model_manager.model is None or model_manager.tokenizer is None:
        raise ValueError("No model loaded. Please load a model first via /api/load-model.")

    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    device = model_manager.device
    model = model_manager.model
    tokenizer = model_manager.tokenizer
    id_to_tag = model_manager.id_to_tag

    # 1. LaTeX replacement
    latex_spans = get_latex_spans(raw_text)
    processed_text = ""
    last_idx = 0
    for start, end in latex_spans:
        processed_text += raw_text[last_idx:start] + "[LATEX]"
        last_idx = end
    processed_text += raw_text[last_idx:]

    mapper = OffsetMapper(latex_spans)
    map_idx = mapper.map_idx

    # 2. Tokenize sliding window
    tokenized = tokenizer(
        processed_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True
    )

    span_logits = {}
    span_tokens = {}

    num_chunks = len(tokenized["input_ids"])
    sigma = 64

    for chunk_idx in range(num_chunks):
        chunk_input_ids = tokenized["input_ids"][chunk_idx]
        chunk_attention_mask = tokenized["attention_mask"][chunk_idx]
        chunk_mod_offsets = tokenized["offset_mapping"][chunk_idx]
        
        chunk_offsets = []
        for start, end in chunk_mod_offsets:
            if start == 0 and end == 0:
                chunk_offsets.append((0, 0))
            else:
                chunk_offsets.append((map_idx(start), map_idx(end)))
        
        if model_manager.is_onnx:
            import numpy as np
            onnx_inputs = {
                "input_ids": np.array([chunk_input_ids], dtype=np.int64),
                "attention_mask": np.array([chunk_attention_mask], dtype=np.int64)
            }
            outputs = model.run(["logits"], onnx_inputs)
            chunk_logits = outputs[0][0]
        else:
            inputs = {
                "input_ids": torch.tensor([chunk_input_ids]).to(device),
                "attention_mask": torch.tensor([chunk_attention_mask]).to(device)
            }
            with torch.no_grad():
                outputs = model(**inputs)
            chunk_logits = outputs.logits[0].cpu()
            
        chunk_tokens = tokenizer.convert_ids_to_tokens(chunk_input_ids)
        chunk_len = len(chunk_input_ids)
        
        for i, (token, offset, mask) in enumerate(zip(chunk_tokens, chunk_offsets, chunk_attention_mask)):
            start, end = offset
            if (start == 0 and end == 0) or mask == 0:
                continue
                
            span = (start, end)
            
            dist_to_boundary = min(i, chunk_len - 1 - i)
            w = min(1.0, dist_to_boundary / sigma)
            w = max(0.0001, w)
            
            if span not in span_logits:
                span_logits[span] = {"logits_sum": chunk_logits[i] * w, "weight_sum": w}
                span_tokens[span] = token
            else:
                span_logits[span]["logits_sum"] += chunk_logits[i] * w
                span_logits[span]["weight_sum"] += w

    # 3. Average overlapping logits and generate sequence predictions
    sorted_spans = sorted(span_logits.keys(), key=lambda x: (x[0], x[1]))
    
    predictions = []
    offsets = []
    tokens = []
    
    for span in sorted_spans:
        start, end = span
        logits_sum = span_logits[span]["logits_sum"]
        weight_sum = span_logits[span]["weight_sum"]
        avg_logits = logits_sum / weight_sum
        
        if model_manager.is_onnx:
            import numpy as np
            pred_id = int(np.argmax(avg_logits))
        else:
            pred_id = torch.argmax(avg_logits).item()
        
        predictions.append(pred_id)
        offsets.append(span)
        tokens.append(span_tokens[span])
        
    tag_to_id = {v: k for k, v in id_to_tag.items()}
    predictions = resolve_bio_violations(predictions, id_to_tag, tag_to_id)

    # 4. Extract contiguous labeled spans from tokens
    segments = []
    current_label = None
    current_start = -1
    current_end = -1
    
    for idx, (pred_id, offset) in enumerate(zip(predictions, offsets)):
        start, end = offset
        label = id_to_tag[pred_id]
        
        if label.startswith("B-"):
            if current_label and current_start < current_end:
                segments.append({
                    "label": current_label,
                    "start": current_start,
                    "end": current_end,
                    "text": raw_text[current_start:current_end]
                })
            current_label = label[2:]
            current_start = start
            current_end = end
        elif label.startswith("I-") and current_label == label[2:]:
            if current_start == -1:
                current_start = start
            current_end = end
        else:
            if current_label and current_start < current_end:
                segments.append({
                    "label": current_label,
                    "start": current_start,
                    "end": current_end,
                    "text": raw_text[current_start:current_end]
                })
                current_label = None
                current_start = -1
                current_end = -1
                
    if current_label and current_start < current_end:
        segments.append({
            "label": current_label,
            "start": current_start,
            "end": current_end,
            "text": raw_text[current_start:current_end]
        })

    filtered_spans = []
    for seg in segments:
        text_content = seg["text"]
        filtered_spans.append({
            "label": seg["label"],
            "start": seg["start"],
            "end": seg["end"],
            "text": text_content
        })

    # 5. Build XML
    xml_content = build_xml(raw_text, filtered_spans)

    # 6. Parse segments to questions
    structured_exam = parse_segments_to_questions(filtered_spans)

    # 7. Token details
    token_details = []
    for token, pred_id, offset in zip(tokens, predictions, offsets):
        start, end = offset
        tag = id_to_tag[pred_id]
        readable_token = token.replace(" ", " ").replace("▁", "")
        token_details.append({
            "token": readable_token,
            "tag": tag,
            "start": start,
            "end": end
        })

    return {
        "raw_text": raw_text,
        "spans": filtered_spans,
        "xml_content": xml_content,
        "structured_exam": structured_exam,
        "token_details": token_details
    }

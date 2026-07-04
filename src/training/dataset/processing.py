import re
import random
import copy
from typing import Dict, Any, List, Tuple, Optional
from src.generation.reconstructor import (
    reconstruct_question, 
    reconstruct_exam, 
    ReconstructorConfig
)
from src.training.dataset.alignment import align_tokens_to_spans

def mask_latex_in_real_data(
    raw_text: str,
    spans: List[Dict[str, Any]],
    placeholder: str,
    mask_prob: float,
    rng: random.Random
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Finds LaTeX formulas in raw_text, masks them with placeholder, and shifts span offsets.
    """
    if mask_prob <= 0.0 or not raw_text:
        return raw_text, spans

    pattern = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)
    matches = list(pattern.finditer(raw_text))
    if not matches:
        return raw_text, spans
        
    current_text = raw_text
    new_spans = [dict(s) for s in spans]
    
    for match in reversed(matches):
        if rng.random() > mask_prob:
            continue
            
        m_start, m_end = match.span()
        diff = len(placeholder) - (m_end - m_start)
        
        current_text = current_text[:m_start] + placeholder + current_text[m_end:]
        
        updated_spans = []
        for span in new_spans:
            s_start = span["start"]
            s_end = span["end"]
            
            if s_start >= m_end:
                span["start"] += diff
                span["end"] += diff
            elif s_start < m_start and s_end > m_end:
                span["end"] += diff
            elif s_start >= m_start and s_end <= m_end:
                span["start"] = m_start
                span["end"] = m_start + len(placeholder)
            elif s_start < m_start and s_end > m_start:
                span["end"] = m_start + len(placeholder)
            
            if "text" in span:
                span["text"] = current_text[span["start"]:span["end"]]
            updated_spans.append(span)
        new_spans = updated_spans
        
    return current_text, new_spans

def process_exam_level(
    exam_data: Dict[str, Any],
    tokenizer: Any,
    tag_to_id: Dict[str, int],
    id_to_tag: Dict[int, str],
    window_configs: List[Tuple[int, int]],
    reconstructor_config: ReconstructorConfig
) -> List[Dict[str, Any]]:
    """
    Reconstructs the full exam document, tokenizes it across multiple sliding window configs,
    aligns labels, and returns a list of prepared samples.
    """
    if exam_data.get("is_real", False) and "raw_text" in exam_data and "spans" in exam_data:
        raw_text = exam_data["raw_text"]
        spans = exam_data["spans"]
        if reconstructor_config.latex_mask_prob > 0.0:
            import hashlib
            h = hashlib.md5((exam_data.get("exam_id", "") or raw_text).encode("utf-8")).hexdigest()
            rng = random.Random(int(h, 16) & 0xFFFFFFFF)
            raw_text, spans = mask_latex_in_real_data(
                raw_text, spans, reconstructor_config.latex_placeholder, reconstructor_config.latex_mask_prob, rng
            )
    else:
        exam_reconstructed = reconstruct_exam(exam_data, reconstructor_config)
        raw_text = exam_reconstructed["raw_text"]
        spans = exam_reconstructed["spans"]
    
    samples = []
    for max_len, stride in window_configs:
        tokenized = tokenizer(
            raw_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            add_special_tokens=True
        )
        
        num_chunks = len(tokenized["input_ids"])
        for chunk_idx in range(num_chunks):
            input_ids = tokenized["input_ids"][chunk_idx]
            attention_mask = tokenized["attention_mask"][chunk_idx]
            offset_mapping = tokenized["offset_mapping"][chunk_idx]
            
            labels = align_tokens_to_spans(offset_mapping, spans, tag_to_id, raw_text)
            tags = [id_to_tag.get(label_id, "O") if label_id != -100 else "IGNORE" for label_id in labels]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            samples.append({
                "tokens": tokens,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "tags": tags,
                "metadata": {
                    "subject": exam_data.get("subject"),
                    "grade": exam_data.get("grade"),
                    "exam_id": exam_data.get("exam_id"),
                    "is_real": exam_data.get("is_real", False),
                    "max_len": max_len,
                    "stride": stride,
                    "chunk_idx": chunk_idx,
                    "total_chunks": num_chunks
                }
            })
            
    return samples

def process_question_as_exam_level(
    q_data: Dict[str, Any],
    tokenizer: Any,
    tag_to_id: Dict[str, int],
    id_to_tag: Dict[int, str],
    window_configs: List[Tuple[int, int]],
    reconstructor_config: ReconstructorConfig
) -> List[Dict[str, Any]]:
    """
    Treats an individual question as a mini-exam and tokenizes it using sliding window configs.
    """
    if q_data.get("is_real", False) and "raw_text" in q_data and "spans" in q_data:
        raw_text = q_data["raw_text"]
        spans = q_data["spans"]
        if reconstructor_config.latex_mask_prob > 0.0:
            import hashlib
            h = hashlib.md5((q_data.get("exam_id", "") or raw_text).encode("utf-8")).hexdigest()
            rng = random.Random(int(h, 16) & 0xFFFFFFFF)
            raw_text, spans = mask_latex_in_real_data(
                raw_text, spans, reconstructor_config.latex_placeholder, reconstructor_config.latex_mask_prob, rng
            )
    else:
        q_reconstructed = reconstruct_question(q_data, reconstructor_config)
        raw_text = q_reconstructed["raw_text"]
        spans = q_reconstructed["spans"]
    
    samples = []
    for max_len, stride in window_configs:
        tokenized = tokenizer(
            raw_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            add_special_tokens=True
        )
        
        num_chunks = len(tokenized["input_ids"])
        for chunk_idx in range(num_chunks):
            input_ids = tokenized["input_ids"][chunk_idx]
            attention_mask = tokenized["attention_mask"][chunk_idx]
            offset_mapping = tokenized["offset_mapping"][chunk_idx]
            
            labels = align_tokens_to_spans(offset_mapping, spans, tag_to_id, raw_text)
            tags = [id_to_tag.get(label_id, "O") if label_id != -100 else "IGNORE" for label_id in labels]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            samples.append({
                "tokens": tokens,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "tags": tags,
                "metadata": {
                    "subject": q_data.get("subject"),
                    "grade": q_data.get("grade"),
                    "question_type": q_data.get("question_type"),
                    "difficulty": q_data.get("difficulty"),
                    "is_real": q_data.get("is_real", False),
                    "max_len": max_len,
                    "stride": stride,
                    "chunk_idx": chunk_idx,
                    "total_chunks": num_chunks
                }
            })
            
    return samples

def process_single_question_legacy(
    q_data: Dict[str, Any], 
    tokenizer: Any, 
    tag_to_id: Dict[str, int], 
    id_to_tag: Dict[int, str],
    reconstructor_config: ReconstructorConfig
) -> Optional[Dict[str, Any]]:
    """
    Legacy method for single question parsing.
    """
    if q_data.get("is_real", False) and "raw_text" in q_data and "spans" in q_data:
        raw_text = q_data["raw_text"]
        spans = q_data["spans"]
        if reconstructor_config.latex_mask_prob > 0.0:
            import hashlib
            h = hashlib.md5((q_data.get("exam_id", "") or raw_text).encode("utf-8")).hexdigest()
            rng = random.Random(int(h, 16) & 0xFFFFFFFF)
            raw_text, spans = mask_latex_in_real_data(
                raw_text, spans, reconstructor_config.latex_placeholder, reconstructor_config.latex_mask_prob, rng
            )
    else:
        q_reconstructed = reconstruct_question(q_data, reconstructor_config)
        raw_text = q_reconstructed["raw_text"]
        spans = q_reconstructed["spans"]
    
    tokenized = tokenizer(
        raw_text,
        return_offsets_mapping=True,
        truncation=True,
        add_special_tokens=True
    )
    
    offset_mapping = tokenized["offset_mapping"]
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    labels = align_tokens_to_spans(offset_mapping, spans, tag_to_id, raw_text)
    tags = [id_to_tag.get(label_id, "O") if label_id != -100 else "IGNORE" for label_id in labels]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    return {
        "tokens": tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "tags": tags,
        "metadata": {
            "subject": q_data.get("subject"),
            "grade": q_data.get("grade"),
            "question_type": q_data.get("question_type"),
            "difficulty": q_data.get("difficulty"),
            "is_group": q_data.get("is_group", False),
            "is_real": q_data.get("is_real", False),
            "chapter": q_data.get("chapter"),
            "unit": q_data.get("unit"),
            "problem_type_id": q_data.get("problem_type_id"),
            "problem_type_name": q_data.get("problem_type_name"),
            "problem_type_level": q_data.get("problem_type_level")
        }
    }

def replace_latex_in_question(q_data: Dict[str, Any], placeholder: str) -> Dict[str, Any]:
    q_copy = copy.deepcopy(q_data)
    
    def process_field(val):
        if isinstance(val, str):
            return re.sub(r"\$\$.*?\$\$|\$.*?\$", placeholder, val)
        elif isinstance(val, list):
            return [process_field(x) for x in val]
        return val

    if q_copy.get("is_group", False):
        if "context" in q_copy:
            q_copy["context"] = process_field(q_copy["context"])
        if "questions" in q_copy and isinstance(q_copy["questions"], list):
            for sub_q in q_copy["questions"]:
                if "stem" in sub_q:
                    sub_q["stem"] = process_field(sub_q["stem"])
                if "options" in sub_q:
                    sub_q["options"] = process_field(sub_q["options"])
    else:
        if "stem" in q_copy:
            q_copy["stem"] = process_field(q_copy["stem"])
        if "options" in q_copy:
            q_copy["options"] = process_field(q_copy["options"])
            
    return q_copy

def process_single_question(
    q_data: Dict[str, Any],
    tokenizer: Any,
    tag_to_id: Dict[str, int],
    id_to_tag: Dict[int, str],
    latex_placeholder: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    config = ReconstructorConfig()
    if latex_placeholder is not None:
        config.latex_placeholder = latex_placeholder
        config.latex_mask_prob = 1.0
    return process_single_question_legacy(q_data, tokenizer, tag_to_id, id_to_tag, config)

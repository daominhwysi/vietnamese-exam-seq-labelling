import os
import sys
import json
import re
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

# Set up local import path if needed
sys.path.append(str(Path(__file__).parent.parent))

from src.generation.reconstructor import (
    reconstruct_question, 
    reconstruct_exam, 
    ReconstructorConfig
)
from src.webapp.inference_helper import get_latex_spans

# Define base tags and generate tag mapping
BASE_TAGS = [
    "question_label",
    "stem",
    "option_label",
    "option_text",
    "stimulus",
    "section",
    "explanation"
]

def get_tag_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    tag_to_id = {"O": 0}
    for tag in BASE_TAGS:
        tag_to_id[f"B-{tag}"] = len(tag_to_id)
        tag_to_id[f"I-{tag}"] = len(tag_to_id)
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag

def resolve_stimulus_anchors(xml_content: str) -> str:
    """
    Resolves self-closing <stimulus id="..." start_anchor="..." end_anchor="..." /> tags
    by locating start_anchor and end_anchor in the surrounding text and wrapping the target
    span with explicit <stimulus>...</stimulus> tags. Also converts legacy <context> tags to <stimulus>.
    """
    if not xml_content:
        return xml_content

    # 1. Convert legacy <context> tags to <stimulus>
    xml_content = re.sub(r"<\s*context\s*>", "<stimulus>", xml_content, flags=re.IGNORECASE)
    xml_content = re.sub(r"<\s*/\s*context\s*>", "</stimulus>", xml_content, flags=re.IGNORECASE)

    # 2. Match self-closing stimulus anchor tag
    pattern = re.compile(
        r"<stimulus\s+[^>]*?start_anchor=([\"\x27])(.*?)\1[^>]*?end_anchor=([\"\x27])(.*?)\3[^>]*?/?>",
        re.DOTALL | re.IGNORECASE
    )

    matches = list(pattern.finditer(xml_content))
    if not matches:
        return xml_content

    # Extract anchors in order
    anchors = []
    for m in matches:
        start_a = m.group(2)
        end_a = m.group(4)
        anchors.append((start_a, end_a))

    # Remove all self-closing stimulus tags
    modified = pattern.sub("", xml_content)

    # Wrap each start_anchor ... end_anchor with <stimulus>...</stimulus>
    cursor = 0
    for start_a, end_a in anchors:
        if not start_a or not end_a:
            continue

        # Clean HTML entities if present
        import html
        start_clean = html.unescape(start_a).strip()
        end_clean = html.unescape(end_a).strip()

        # Find start_anchor starting from cursor
        s_idx = modified.find(start_a, cursor)
        if s_idx == -1 and start_clean != start_a:
            s_idx = modified.find(start_clean, cursor)
            if s_idx != -1:
                start_a = start_clean

        if s_idx != -1:
            # Find end_anchor starting after start_anchor
            e_search_start = s_idx + len(start_a)
            e_idx = modified.find(end_a, e_search_start)
            if e_idx == -1 and end_clean != end_a:
                e_idx = modified.find(end_clean, e_search_start)
                if e_idx != -1:
                    end_a = end_clean

            if e_idx != -1:
                e_end = e_idx + len(end_a)
                before = modified[:s_idx]
                stim_text = modified[s_idx:e_end]
                after = modified[e_end:]

                # Only wrap if not already wrapped
                if not (stim_text.startswith("<stimulus>") and stim_text.endswith("</stimulus>")):
                    wrapped = f"<stimulus>{stim_text}</stimulus>"
                    modified = before + wrapped + after
                    cursor = len(before) + len(wrapped)
                else:
                    cursor = e_end

    return modified

def spans_to_xml(raw_text: str, spans: List[Dict[str, Any]]) -> str:
    """
    Converts raw_text + ground-truth character-level spans into an inline-tagged
    XML string that matches the format produced by annotate_ocr.py:

        <question_label>Câu 1.</question_label> <stem>Nội dung...</stem>

    Spans are sorted by start offset. Untagged gaps between spans (page headers,
    separators, etc.) are preserved verbatim outside any tag.
    """
    if not spans:
        return raw_text

    # Sort spans by start position; resolve overlaps by taking first occurrence
    sorted_spans = sorted(spans, key=lambda s: s["start"])

    result = []
    cursor = 0

    for span in sorted_spans:
        start = span["start"]
        end = span["end"]
        label = span["label"]
        if label == "context":
            label = "stimulus"

        # Skip malformed or already-passed spans
        if end <= start or start < cursor:
            continue

        # Untagged gap before this span
        if start > cursor:
            result.append(raw_text[cursor:start])

        # Tagged span content
        span_text = raw_text[start:end]
        result.append(f"<{label}>{span_text}</{label}>")
        cursor = end

    # Trailing untagged text
    if cursor < len(raw_text):
        result.append(raw_text[cursor:])

    return "".join(result)

def sanitize_nested_explanation_tags(text: str) -> str:
    """
    Strips any entity tags accidentally nested inside <explanation>...</explanation>
    (e.g., <option_label>a.</option_label> or <stem> inside explanation),
    ensuring that the explanation block remains a single continuous span.
    """
    if not text or "<explanation>" not in text:
        return text

    def _clean_inner(match):
        inner = match.group(1)
        cleaned = re.sub(r'</?(?:stem|option_label|option_text|question_label|stimulus|section)>', '', inner)
        return f"<explanation>{cleaned}</explanation>"

    return re.sub(r'<explanation>(.*?)</explanation>', _clean_inner, text, flags=re.DOTALL)

def parse_xml_annotations(tagged_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parses an inline XML-tagged string into raw untagged text and character span dictionaries.
    Resolves self-closing stimulus anchors into <stimulus> tags before span computation.
    Only recognized entity tags are stripped and recorded as spans; other tags are preserved verbatim.
    """
    tagged_text = sanitize_nested_explanation_tags(tagged_text)
    tagged_text = resolve_stimulus_anchors(tagged_text)

    allowed_tags = {"question_label", "stem", "option_label", "option_text", "stimulus", "section", "explanation"}
    raw_chars = []
    spans = []

    tag_pattern = re.compile(r"<(/)?([a-zA-Z_0-9]+)>")

    pos = 0
    current_open_tag = None
    tag_start_idx = -1

    for match in tag_pattern.finditer(tagged_text):
        start, end = match.span()
        text_before = tagged_text[pos:start]
        raw_chars.append(text_before)

        is_closing = bool(match.group(1))
        raw_tag_name = match.group(2)
        tag_name = "stimulus" if raw_tag_name == "context" else raw_tag_name

        if tag_name in allowed_tags:
            if not is_closing:
                current_open_tag = tag_name
                tag_start_idx = len("".join(raw_chars))
            else:
                if current_open_tag == tag_name and tag_start_idx != -1:
                    tag_end_idx = len("".join(raw_chars))
                    span_text = "".join(raw_chars)[tag_start_idx:tag_end_idx]
                    spans.append(
                        {
                            "start": tag_start_idx,
                            "end": tag_end_idx,
                            "label": tag_name,
                            "text": span_text,
                        }
                    )
                current_open_tag = None
                tag_start_idx = -1
        else:
            raw_chars.append(match.group(0))

        pos = end

    raw_chars.append(tagged_text[pos:])
    raw_text = "".join(raw_chars)

    return raw_text, spans

SUBJECT_KEYWORD_MAP = {
    "toan": "math_algebra",
    "số phức": "math_algebra",
    "so phuc": "math_algebra",
    "bất đẳng thức": "math_algebra",
    "bat dang thuc": "math_algebra",
    "hàm số": "math_algebra",
    "ham so": "math_algebra",
    "hình học": "math_geometry",
    "hinh hoc": "math_geometry",
    "vat_ly": "physics",
    "vat_li": "physics",
    "vật lý": "physics",
    "vật lí": "physics",
    "hoa_hoc": "chemistry",
    "hóa học": "chemistry",
    "sinh_hoc": "biology",
    "sinh học": "biology",
    "lich_su": "history",
    "lịch sử": "history",
    "dia_ly": "geography",
    "dia_li": "geography",
    "địa lý": "geography",
    "địa lí": "geography",
    "tieng_anh": "english",
    "tiếng anh": "english",
    "ngu_van": "literature",
    "ngữ văn": "literature"
}

def infer_metadata_from_path(file_path: Path, input_root: Path) -> Dict[str, Any]:
    """
    Infers subject, grade, category, and exam_id from relative directory and file name.
    """
    try:
        rel = file_path.relative_to(input_root)
    except ValueError:
        rel = file_path
    rel_str = str(rel).lower()

    subject = "general"
    for k, v in SUBJECT_KEYWORD_MAP.items():
        if k in rel_str:
            subject = v
            break

    grade = 12
    if "g10" in rel_str or "grade_10" in rel_str or "lop_10" in rel_str or "lop10" in rel_str:
        grade = 10
    elif "g11" in rel_str or "grade_11" in rel_str or "lop_11" in rel_str or "lop11" in rel_str:
        grade = 11
    elif any(kw in rel_str for kw in ["g12", "grade_12", "lop_12", "lop12", "thpt", "dgnl", "tsa"]):
        grade = 12

    exam_id = "_".join(rel.parts[:-1]) if len(rel.parts) > 1 else rel.stem
    category = rel.parts[0] if len(rel.parts) > 1 else "root"

    return {
        "exam_id": exam_id,
        "subject": subject,
        "grade": grade,
        "category": category,
        "is_real": True
    }

def align_tokens_to_spans(
    offset_mapping: List[Tuple[int, int]], 
    spans: List[Dict[str, Any]], 
    tag_to_id: Dict[str, int],
    raw_text: Optional[str] = None
) -> List[int]:
    """
    Aligns tokenizer offset mapping with character-level spans to assign token-level labels
    using the V2 Character-Anchor Lookup method.
    """
    # 1. Clean spans (strip whitespaces/tabs)
    clean_spans = []
    for span in spans:
        span_text = span.get("text", "")
        start = span["start"]
        end = span["end"]
        
        if span_text:
            stripped = span_text.strip()
            if not stripped:
                continue
            leading = len(span_text) - len(span_text.lstrip())
            trailing = len(span_text) - len(span_text.rstrip())
            clean_start = start + leading
            clean_end = end - trailing
        else:
            if raw_text is not None:
                raw_span_text = raw_text[start:end]
                stripped = raw_span_text.strip()
                if not stripped:
                    continue
                leading = len(raw_span_text) - len(raw_span_text.lstrip())
                trailing = len(raw_span_text) - len(raw_span_text.rstrip())
                clean_start = start + leading
                clean_end = end - trailing
            else:
                clean_start = start
                clean_end = end
                
        clean_spans.append({
            "start": clean_start,
            "end": clean_end,
            "label": span["label"]
        })
        
    span_starts = [s["start"] for s in clean_spans]
    labels = []
    
    # 2. Map tokens based on first non-whitespace character offset lookup (O(log S) binary search)
    import bisect
    for start, end in offset_mapping:
        if start == 0 and end == 0:
            labels.append(-100)
            continue
            
        non_space_char_idx = -1
        if raw_text is not None:
            for char_idx in range(start, end):
                if char_idx < len(raw_text) and not raw_text[char_idx].isspace():
                    non_space_char_idx = char_idx
                    break
        else:
            # Fallback if raw_text is not provided
            non_space_char_idx = start
            
        if non_space_char_idx == -1:
            # Whitespace-only token. Check if it falls entirely within some span.
            matched_span = None
            span_idx = bisect.bisect_right(span_starts, start) - 1
            if span_idx >= 0:
                s = clean_spans[span_idx]
                if s["start"] <= start and end <= s["end"]:
                    matched_span = s
            if matched_span is not None:
                labels.append(tag_to_id.get(f"I-{matched_span['label']}", tag_to_id["O"]))
            else:
                labels.append(tag_to_id["O"])
            continue
            
        matched_span = None
        span_idx = bisect.bisect_right(span_starts, non_space_char_idx) - 1
        if span_idx >= 0:
            s = clean_spans[span_idx]
            if s["start"] <= non_space_char_idx < s["end"]:
                matched_span = s
                
        if matched_span is None:
            labels.append(tag_to_id["O"])
        else:
            span_label = matched_span["label"]
            if non_space_char_idx == matched_span["start"]:
                tag = f"B-{span_label}"
            else:
                tag = f"I-{span_label}"
            labels.append(tag_to_id.get(tag, tag_to_id["O"]))
            
    return labels

def mask_latex_in_real_data(
    raw_text: str,
    spans: List[Dict[str, Any]],
    placeholder: str,
    mask_prob: float,
    rng: random.Random
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Finds LaTeX formulas ($...$ and $$...$$) in raw_text using validated spans,
    masks them with placeholder with probability mask_prob, and shifts span offsets accordingly.
    """
    if mask_prob <= 0.0 or not raw_text:
        return raw_text, spans

    latex_spans = get_latex_spans(raw_text)
    if not latex_spans:
        return raw_text, spans
        
    current_text = raw_text
    new_spans = [dict(s) for s in spans]
    
    # Process from back to front to avoid shifting indices of earlier matches
    for m_start, m_end in reversed(latex_spans):
        if rng.random() > mask_prob:
            continue
            
        diff = len(placeholder) - (m_end - m_start)
        
        # Replace in text
        current_text = current_text[:m_start] + placeholder + current_text[m_end:]
        
        # Adjust spans
        updated_spans = []
        for span in new_spans:
            s_start = span["start"]
            s_end = span["end"]
            
            # If the span starts after the replaced segment, shift it
            if s_start >= m_end:
                span["start"] += diff
                span["end"] += diff
            # If the span starts before but ends after/during
            elif s_start < m_start and s_end > m_end:
                span["end"] += diff
            # If the span is entirely within the masked LaTeX
            elif s_start >= m_start and s_end <= m_end:
                span["start"] = m_start
                span["end"] = m_start + len(placeholder)
            # If the span overlaps the beginning but not the end
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
    Legacy method for single question parsing (keeps original question-level split layout).
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
    import copy
    q_copy = copy.deepcopy(q_data)
    
    def process_field(val):
        if isinstance(val, str):
            return re.sub(r"\$\$.*?\$\$|\$.*?\$", placeholder, val)
        elif isinstance(val, list):
            return [process_field(x) for x in val]
        return val

    if q_copy.get("is_group", False):
        if "stimulus" in q_copy:
            q_copy["stimulus"] = process_field(q_copy["stimulus"])
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

def parse_args():
    parser = argparse.ArgumentParser(description="XLM-RoBERTa Sequence Labelling Dataset Preparer")
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default="output",
        help="Directory containing the input question JSON files (default: 'output')"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="output/dataset",
        help="Directory to save the output dataset splits (default: 'output/dataset')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="jhu-clsp/mmBERT-base",
        help="Hugging Face model / tokenizer name (default: 'jhu-clsp/mmBERT-base')"
    )
    parser.add_argument(
        "--latex-placeholder",
        type=str,
        default="[LATEX]",
        help="Special token placeholder for LaTeX equations (default: '[LATEX]')"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training set (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of validation set (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting (default: 42)"
    )
    
    # Advanced data prep features
    parser.add_argument(
        "--exam-level",
        action="store_true",
        help="Process datasets at the exam level with multi-scale sliding windows (default: False)"
    )
    parser.add_argument(
        "--max-len",
        type=str,
        default="512,768,1024,2048",
        help="Comma-separated sequence lengths for tokenization (default: '512,768,1024,2048')"
    )
    parser.add_argument(
        "--stride",
        type=str,
        default="128,192,256,512",
        help="Comma-separated strides for tokenization (default: '128,192,256,512')"
    )
    
    # Advanced Data Augmentations
    parser.add_argument(
        "--typo-rate",
        type=float,
        default=0.02,
        help="Spelling mistake typo injection rate (default: 0.02)"
    )
    parser.add_argument(
        "--space-noise-rate",
        type=float,
        default=0.15,
        help="Spacing noise injection rate (default: 0.15)"
    )
    parser.add_argument(
        "--latex-mask-prob",
        type=float,
        default=0.5,
        help="Probability of masking LaTeX formulas (default: 0.5)"
    )
    parser.add_argument(
        "--enable-permutations",
        action="store_true",
        help="Enable random permutations of questions and options (default: False)"
    )
    parser.add_argument(
        "--option-drop-prob",
        type=float,
        default=0.05,
        help="Probability of dropping 1 to 3 options to simulate OCR cuts (default: 0.05)"
    )
    parser.add_argument(
        "--casing-noise-prob",
        type=float,
        default=0.10,
        help="Probability of random casing/capitalization noise (default: 0.10)"
    )
    parser.add_argument(
        "--synonym-swap-prob",
        type=float,
        default=0.10,
        help="Probability of random prefix synonym swap (default: 0.10)"
    )
    parser.add_argument(
        "--formatting-noise-prob",
        type=float,
        default=0.10,
        help="Probability of wrapping labels in random Markdown/HTML formatting (default: 0.10)"
    )
    parser.add_argument(
        "--inline-option-prob",
        type=float,
        default=0.0,
        help="Probability of formatting options inline (default: 0.0)"
    )
    parser.add_argument(
        "--min-inline-spaces",
        type=int,
        default=5,
        help="Minimum random spaces to inject between inline options (default: 5)"
    )
    parser.add_argument(
        "--max-inline-spaces",
        type=int,
        default=30,
        help="Maximum random spaces to inject between inline options (default: 30)"
    )
    parser.add_argument(
        "--min-inline-tabs",
        type=int,
        default=1,
        help="Minimum random tabs to inject between inline options (default: 1)"
    )
    parser.add_argument(
        "--max-inline-tabs",
        type=int,
        default=3,
        help="Maximum random tabs to inject between inline options (default: 3)"
    )
    parser.add_argument(
        "--grid-2x2-prob",
        type=float,
        default=0.0,
        help="Probability of formatting options in a 2x2 grid layout (default: 0.0)"
    )
    parser.add_argument(
        "--same-line-stem-options-prob",
        type=float,
        default=0.0,
        help="Probability of placing stem and first option on the same line (default: 0.0)"
    )
    parser.add_argument(
        "--flatten-newlines-prob",
        type=float,
        default=0.0,
        help="Probability of flattening all newlines and tabs into single spaces (default: 0.0)"
    )
    parser.add_argument(
        "--collapse-whitespace-prob",
        type=float,
        default=0.0,
        help="Probability of collapsing repeated spaces/tabs into a single space (default: 0.0)"
    )
    parser.add_argument(
        "--only-passed",
        action="store_true",
        default=True,
        help="Only include documents that passed quality audit if audit_report.json exists (default: True)"
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all documents regardless of audit status"
    )
    return parser.parse_args()

def consolidate_raw_exams(
    input_dir: Union[str, Path] = "output",
    output_file: Union[str, Path] = "output/dataset/raw_exams.jsonl",
    filter_passed: bool = True,
    include_synthetic: bool = True,
    include_real: bool = True
) -> List[Dict[str, Any]]:
    """
    Scans input_dir for synthetic mock exams and audited real OCR exams.
    Filters out unpassed audits and consolidates all clean documents into a single raw_exams.jsonl.
    """
    input_path = Path(input_dir)
    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    records = []
    seen_ids = set()

    for dirpath, _, filenames in os.walk(input_path, followlinks=True):
        dp = Path(dirpath)
        rel_str = str(dp)
        if "output/dataset" in rel_str or "dataset/xml" in rel_str:
            continue

        # 1. Real annotated XML files
        if include_real and ("merged.xml" in filenames or any(fn.endswith("_annotated.xml") for fn in filenames)):
            xml_name = "merged.xml" if "merged.xml" in filenames else next(fn for fn in filenames if fn.endswith("_annotated.xml"))
            fp = dp / xml_name
            
            if filter_passed:
                audit_file = dp / "audit_report.json"
                if audit_file.exists():
                    try:
                        audit = json.loads(audit_file.read_text(encoding="utf-8"))
                        if audit.get("decision", "").strip().upper() != "PASS" or audit.get("is_malfunctioned", False):
                            continue
                    except Exception:
                        continue
            try:
                content = fp.read_text(encoding="utf-8")
                raw_text, spans = parse_xml_annotations(content)
                if spans:
                    meta = infer_metadata_from_path(fp, input_path)
                    exam_id = meta.get("exam_id", fp.stem)
                    if exam_id not in seen_ids:
                        seen_ids.add(exam_id)
                        records.append({
                            "exam_id": exam_id,
                            "is_real": True,
                            "subject": meta.get("subject", "general"),
                            "grade": meta.get("grade", 12),
                            "category": meta.get("category", "root"),
                            "raw_text": raw_text,
                            "spans": spans,
                            "raw_xml": spans_to_xml(raw_text, spans),
                            "source_file": str(fp.relative_to(input_path)) if fp.is_relative_to(input_path) else fp.name
                        })
            except Exception:
                pass

        # 2. JSON files (synthetic exams, questions, or real JSONs)
        for fn in filenames:
            if not fn.endswith(".json") or fn.startswith("chunk_") or fn in ["label_mapping.json", "audit_report.json", "train_stats.json", "val_stats.json", "test_stats.json", "raw_exams.jsonl"]:
                continue
            fp = dp / fn
            
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            if isinstance(data, dict):
                # Real JSON exam
                if include_real and data.get("is_real", False) and "raw_text" in data and "spans" in data:
                    exam_id = data.get("exam_id", fp.stem)
                    if exam_id not in seen_ids:
                        seen_ids.add(exam_id)
                        meta = infer_metadata_from_path(fp, input_path)
                        records.append({
                            "exam_id": exam_id,
                            "is_real": True,
                            "subject": data.get("subject", meta.get("subject", "general")),
                            "grade": data.get("grade", meta.get("grade", 12)),
                            "category": data.get("category", meta.get("category", "root")),
                            "raw_text": data["raw_text"],
                            "spans": data["spans"],
                            "raw_xml": data.get("raw_xml") or spans_to_xml(data["raw_text"], data["spans"]),
                            "source_file": str(fp.relative_to(input_path)) if fp.is_relative_to(input_path) else fp.name
                        })

                # Synthetic compiled exam
                elif include_synthetic and "sections" in data:
                    exam_id = data.get("exam_id", fp.stem)
                    if exam_id not in seen_ids:
                        seen_ids.add(exam_id)
                        rec = reconstruct_exam(data, ReconstructorConfig())
                        records.append({
                            "exam_id": exam_id,
                            "is_real": False,
                            "subject": data.get("subject", "general"),
                            "grade": data.get("grade", 12),
                            "category": "synthetic",
                            "raw_text": rec["raw_text"],
                            "spans": rec["spans"],
                            "raw_xml": spans_to_xml(rec["raw_text"], rec["spans"]),
                            "sections": data.get("sections"),
                            "source_file": str(fp.relative_to(input_path)) if fp.is_relative_to(input_path) else fp.name
                        })

                # Synthetic single question
                elif include_synthetic and (data.get("stem") or data.get("stimulus") or data.get("questions")):
                    q_id = data.get("question_id", fp.stem)
                    if q_id not in seen_ids:
                        seen_ids.add(q_id)
                        rec = reconstruct_question(data, ReconstructorConfig())
                        records.append({
                            "exam_id": q_id,
                            "is_real": False,
                            "subject": data.get("subject", "general"),
                            "grade": data.get("grade", 12),
                            "category": "synthetic",
                            "raw_text": rec["raw_text"],
                            "spans": rec["spans"],
                            "raw_xml": spans_to_xml(rec["raw_text"], rec["spans"]),
                            "question_data": data,
                            "source_file": str(fp.relative_to(input_path)) if fp.is_relative_to(input_path) else fp.name
                        })

    with open(output_file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_real = sum(1 for r in records if r.get("is_real", False))
    n_synth = len(records) - n_real
    print(f"Successfully consolidated {len(records)} raw exams ({n_real} real, {n_synth} synthetic) -> '{output_file_path}'")
    return records

def run_prepare_dataset(args):
    if args.train_ratio + args.val_ratio > 1.0 or args.train_ratio < 0.0 or args.val_ratio < 0.0:
        print("Error: train-ratio and val-ratio must sum to <= 1.0 and be non-negative.")
        sys.exit(1)
        
    test_ratio = 1.0 - (args.train_ratio + args.val_ratio)
    
    # Setup paths
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
        
    def _is_excluded(p: Path) -> bool:
        p_str = str(p.resolve())
        out_str = str(output_path.resolve())
        if p_str.startswith(out_str):
            return True
        if "/dataset/" in p_str or "\\dataset\\" in p_str:
            return True
        if "/.cache/" in p_str or "\\.cache\\" in p_str:
            return True
        if p.name.startswith("chunk_"):
            return True
        return False

    def _find_files(root: Path, pattern: str) -> List[Path]:
        matches = []
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            dp = Path(dirpath)
            for fn in filenames:
                fp = dp / fn
                if (fn.endswith(pattern) or Path(fn).match(pattern)) and not _is_excluded(fp):
                    matches.append(fp)
        return matches

    json_files = _find_files(input_path, "question_*.json")
    exam_files = _find_files(input_path, "exam_*.json")
    real_exam_files = _find_files(input_path, "real_exam_*.json")
    
    # Also find XML annotations (e.g. merged.xml, *_annotated.xml, or any XML files in source directories)
    xml_files = _find_files(input_path, "merged.xml")
    if not xml_files:
        xml_files = [f for f in _find_files(input_path, "*_annotated.xml") if not _is_excluded(f)]
    if not xml_files:
        xml_files = [f for f in _find_files(input_path, "*.xml") if not _is_excluded(f)]

    if not json_files and not exam_files and not real_exam_files and not xml_files:
        print(f"No question JSON files, exam JSON files, real exam JSON files, or annotated XML files found in '{args.input_dir}'. Please generate data first.")
        sys.exit(1)
    # Combine real exams into exam_files list
    exam_files.extend(real_exam_files)
        
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: 'transformers' is not installed. Please install it or use the active Pixi environment.")
        sys.exit(1)
        
    print(f"Loading tokenizer '{args.model}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
        
    # Process LaTeX placeholder and special tokens
    latex_placeholder = args.latex_placeholder if args.latex_placeholder and args.latex_placeholder.strip() else "[LATEX]"
    special_tokens = ["<blank />", "<blank/>", "[BLANK]"]
    if latex_placeholder:
        special_tokens.append(latex_placeholder)
    
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
    tag_to_id, id_to_tag = get_tag_mappings()
    
    label_mapping = {
        "tag_to_id": tag_to_id,
        "id_to_tag": id_to_tag
    }
    with open(output_path / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        
    # Parse window configurations
    max_lens = [int(x.strip()) for x in getattr(args, "max_len", "512").split(",") if x.strip()]
    strides = [int(x.strip()) for x in getattr(args, "stride", "128").split(",") if x.strip()]
    while len(strides) < len(max_lens):
        strides.append(strides[-1] if strides else 128)
    strides = strides[:len(max_lens)]
    window_configs = list(zip(max_lens, strides))
    
    # Build the shared ReconstructorConfig
    reconstructor_config = ReconstructorConfig(
        typo_rate=getattr(args, "typo_rate", 0.02),
        space_noise_rate=getattr(args, "space_noise_rate", 0.15),
        latex_mask_prob=getattr(args, "latex_mask_prob", 0.5),
        latex_placeholder=latex_placeholder,
        enable_permutations=getattr(args, "enable_permutations", False),
        option_drop_prob=getattr(args, "option_drop_prob", 0.05),
        casing_noise_prob=getattr(args, "casing_noise_prob", 0.10),
        synonym_swap_prob=getattr(args, "synonym_swap_prob", 0.10),
        formatting_noise_prob=getattr(args, "formatting_noise_prob", 0.10),
        inline_option_prob=getattr(args, "inline_option_prob", 0.0),
        grid_2x2_prob=getattr(args, "grid_2x2_prob", 0.0),
        same_line_stem_options_prob=getattr(args, "same_line_stem_options_prob", 0.0),
        flatten_newlines_prob=getattr(args, "flatten_newlines_prob", 0.0),
        collapse_whitespace_prob=getattr(args, "collapse_whitespace_prob", 0.0),
    )
    
    xml_output_path = output_path / "xml"
    xml_output_path.mkdir(parents=True, exist_ok=True)

    def _save_xml(stem: str, raw_text: str, spans: List[Dict[str, Any]]) -> None:
        """Write ground-truth inline-tagged XML for a source file."""
        try:
            xml_content = spans_to_xml(raw_text, spans)
            xml_file = xml_output_path / f"{stem}_annotated.xml"
            with open(xml_file, "w", encoding="utf-8") as xf:
                xf.write(xml_content)
        except Exception as xe:
            print(f"Warning: Could not write XML for '{stem}': {xe}")

    # Collect all source documents into a registry for document-level partitioning
    all_documents: List[Dict[str, Any]] = []
    
    for f in json_files:
        all_documents.append({
            "doc_id": f.stem,
            "doc_type": "question",
            "file_path": f
        })
        
    for f in exam_files:
        all_documents.append({
            "doc_id": f.stem,
            "doc_type": "exam",
            "file_path": f
        })
        
    filter_passed = not getattr(args, "include_all", False)
    for f in xml_files:
        if filter_passed:
            audit_file = f.parent / "audit_report.json"
            if audit_file.exists():
                try:
                    audit = json.loads(audit_file.read_text(encoding="utf-8"))
                    decision = str(audit.get("decision", "")).strip().upper()
                    is_malfunctioned = audit.get("is_malfunctioned", False)
                    if decision != "PASS" or is_malfunctioned:
                        continue
                except Exception:
                    pass
        meta = infer_metadata_from_path(f, input_path)
        all_documents.append({
            "doc_id": meta["exam_id"],
            "doc_type": "xml",
            "file_path": f,
            "meta": meta
        })

    print(f"Collected {len(all_documents)} total source documents (Questions: {len(json_files)}, Exams: {len(exam_files)}, XMLs: {len(all_documents) - len(json_files) - len(exam_files)}).")

    # Document-level seeded shuffle and split to strictly eliminate data leakage
    rng_split = random.Random(args.seed)
    rng_split.shuffle(all_documents)

    n_total_docs = len(all_documents)
    n_train_docs = int(n_total_docs * args.train_ratio)
    n_val_docs = int(n_total_docs * args.val_ratio)

    doc_splits = {
        "train": all_documents[:n_train_docs],
        "val": all_documents[n_train_docs:n_train_docs + n_val_docs],
        "test": all_documents[n_train_docs + n_val_docs:]
    }

    def process_doc_to_samples(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_type = doc["doc_type"]
        file_path = doc["file_path"]
        samples = []
        try:
            if doc_type == "question":
                with open(file_path, "r", encoding="utf-8") as f:
                    q_data = json.load(f)
                if args.exam_level:
                    doc_samples = process_question_as_exam_level(q_data, tokenizer, tag_to_id, id_to_tag, window_configs, reconstructor_config)
                    for s in doc_samples:
                        s["metadata"]["source_file"] = file_path.name
                        samples.append(s)
                else:
                    sample = process_single_question_legacy(q_data, tokenizer, tag_to_id, id_to_tag, reconstructor_config)
                    if sample:
                        sample["metadata"]["source_file"] = file_path.name
                        samples.append(sample)
                try:
                    q_rec = reconstruct_question(q_data, ReconstructorConfig())
                    _save_xml(file_path.stem, q_rec["raw_text"], q_rec["spans"])
                except Exception:
                    pass

            elif doc_type == "exam":
                with open(file_path, "r", encoding="utf-8") as f:
                    exam_data = json.load(f)
                if args.exam_level:
                    doc_samples = process_exam_level(exam_data, tokenizer, tag_to_id, id_to_tag, window_configs, reconstructor_config)
                    for s in doc_samples:
                        s["metadata"]["source_file"] = file_path.name
                        samples.append(s)
                else:
                    sections = exam_data.get("sections", {})
                    for section_title, questions in sections.items():
                        for idx, q_data in enumerate(questions):
                            q_copy = dict(q_data)
                            if "subject" not in q_copy and "subject" in exam_data:
                                q_copy["subject"] = exam_data["subject"]
                            if "grade" not in q_copy and "grade" in exam_data:
                                q_copy["grade"] = exam_data["grade"]
                            sample = process_single_question_legacy(q_copy, tokenizer, tag_to_id, id_to_tag, reconstructor_config)
                            if sample:
                                sample["metadata"]["source_file"] = f"{file_path.name}::{section_title}::q_{idx}"
                                samples.append(sample)
                try:
                    if exam_data.get("is_real", False) and "raw_text" in exam_data and "spans" in exam_data:
                        _save_xml(file_path.stem, exam_data["raw_text"], exam_data["spans"])
                    else:
                        exam_rec = reconstruct_exam(exam_data, ReconstructorConfig())
                        _save_xml(file_path.stem, exam_rec["raw_text"], exam_rec["spans"])
                except Exception:
                    pass

            elif doc_type == "xml":
                content = file_path.read_text(encoding="utf-8")
                raw_text, spans = parse_xml_annotations(content)
                if spans:
                    meta = doc.get("meta") or infer_metadata_from_path(file_path, input_path)
                    rel_source = str(file_path.relative_to(input_path)) if file_path.is_relative_to(input_path) else file_path.name
                    exam_data = {
                        "exam_id": meta["exam_id"],
                        "is_real": True,
                        "raw_text": raw_text,
                        "spans": spans,
                        "subject": meta["subject"],
                        "grade": meta["grade"],
                        "category": meta["category"],
                        "source_file": rel_source
                    }
                    if args.exam_level:
                        doc_samples = process_exam_level(exam_data, tokenizer, tag_to_id, id_to_tag, window_configs, reconstructor_config)
                        for s in doc_samples:
                            s["metadata"]["source_file"] = rel_source
                            samples.append(s)
                    else:
                        sample = process_single_question_legacy(exam_data, tokenizer, tag_to_id, id_to_tag, reconstructor_config)
                        if sample:
                            sample["metadata"]["source_file"] = rel_source
                            samples.append(sample)
                    _save_xml(meta["exam_id"], raw_text, spans)
        except Exception as e:
            print(f"Warning: Failed to process document {file_path.name}: {e}")

        return samples

    total_samples = 0
    split_counts = {}

    for split_name, docs in doc_splits.items():
        print(f"\nProcessing {split_name} split ({len(docs)} documents)...")
        split_samples = []
        for d_idx, doc in enumerate(docs):
            s_list = process_doc_to_samples(doc)
            split_samples.extend(s_list)
            if (d_idx + 1) % 50 == 0 or (d_idx + 1) == len(docs):
                print(f"  [{split_name}] Processed {d_idx + 1}/{len(docs)} documents ({len(split_samples)} window samples generated)...")
        
        # Shuffle window chunks within the split
        rng_split.shuffle(split_samples)
        split_counts[split_name] = len(split_samples)
        total_samples += len(split_samples)

        split_file = output_path / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for s in split_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Saved {len(split_samples)} samples to '{split_file}'")

    # Also export consolidated raw_exams.jsonl for on-the-fly cloud training
    print("\nConsolidating raw exams to 'raw_exams.jsonl'...")
    try:
        consolidate_raw_exams(
            input_dir=input_path,
            output_file=output_path / "raw_exams.jsonl",
            filter_passed=not getattr(args, "include_all", False)
        )
    except Exception as ce:
        print(f"Warning: Could not consolidate raw_exams.jsonl: {ce}")

    print("\nDataset preparation completed successfully!")
    print(f"Total documents: {n_total_docs} (Train: {len(doc_splits['train'])}, Val: {len(doc_splits['val'])}, Test: {len(doc_splits['test'])})")
    print(f"Total window samples: {total_samples} (Train: {split_counts.get('train', 0)}, Val: {split_counts.get('val', 0)}, Test: {split_counts.get('test', 0)})")
    print(f"Label mapping saved to '{output_path / 'label_mapping.json'}'")
    print(f"Consolidated raw dataset saved to '{output_path / 'raw_exams.jsonl'}'")

def main():
    args = parse_args()
    run_prepare_dataset(args)

if __name__ == "__main__":
    main()

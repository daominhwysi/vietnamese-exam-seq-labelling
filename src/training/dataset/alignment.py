from typing import Dict, Any, List, Tuple, Optional

# Define base tags
BASE_TAGS = [
    "question_label",
    "stem",
    "option_label",
    "option_text",
    "context",
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

def spans_to_xml(raw_text: str, spans: List[Dict[str, Any]]) -> str:
    """
    Converts raw_text + ground-truth character-level spans into an inline-tagged
    XML string:
        <question_label>Câu 1.</question_label> <stem>Nội dung...</stem>
    """
    if not spans:
        return raw_text

    sorted_spans = sorted(spans, key=lambda s: s["start"])
    result = []
    cursor = 0

    for span in sorted_spans:
        start = span["start"]
        end = span["end"]
        label = span["label"]

        if end <= start or start < cursor:
            continue

        if start > cursor:
            result.append(raw_text[cursor:start])

        span_text = raw_text[start:end]
        result.append(f"<{label}>{span_text}</{label}>")
        cursor = end

    if cursor < len(raw_text):
        result.append(raw_text[cursor:])

    return "".join(result)

def align_tokens_to_spans(
    offset_mapping: List[Tuple[int, int]], 
    spans: List[Dict[str, Any]], 
    tag_to_id: Dict[str, int],
    raw_text: Optional[str] = None
) -> List[int]:
    """
    Aligns tokenizer offset mapping with character-level spans to assign token-level labels
    using the Character-Anchor Lookup method.
    """
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
        
    labels = []
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
            non_space_char_idx = start
            
        if non_space_char_idx == -1:
            matched_span = None
            for span in clean_spans:
                if span["start"] <= start and end <= span["end"]:
                    matched_span = span
                    break
            if matched_span is not None:
                labels.append(tag_to_id.get(f"I-{matched_span['label']}", tag_to_id["O"]))
            else:
                labels.append(tag_to_id["O"])
            continue
            
        matched_span = None
        for span in clean_spans:
            if span["start"] <= non_space_char_idx < span["end"]:
                matched_span = span
                break
                
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
